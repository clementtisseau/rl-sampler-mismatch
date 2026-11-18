"""REINFORCE with importance-sampling corrections and a sampler-centered score.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.

The score of each sampled token is centered under the sampler (REPORT.md):
log p(y) - sum_v pi(v) log p(v) with pi = q w / <q, w> detached, so that its
gradient has zero mean under y ~ q. Plain log p(y) has mean gradient
-grad KL(q || p) under q, which the advantage turns into a systematic drift
of the trainer away from (or toward) the sampler. The sampler's distribution
comes from sampler_topk_ids / sampler_topk_logprobs (sampler.vocab_logprobs=k)
or sampling_vocab_logprobs (=full) when recorded, else from the sampled token.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

sg = jax.lax.stop_gradient

# advantage of a (solved, unsolved) rollout; group_centered also subtracts the group mean
REWARD_MODES = {"plus_minus": (1.0, -1.0), "plus_zero": (1.0, 0.0),
                "zero_minus": (0.0, -1.0), "group_centered": (1.0, -1.0)}


def compute_advantages(rewards, group_size, reward_mode):
    solved, unsolved = REWARD_MODES[reward_mode]
    advantages = np.where(np.asarray(rewards) > 0, solved, unsolved).astype(np.float32)
    if reward_mode == "group_centered":
        groups = advantages.reshape(-1, group_size)
        advantages = (groups - groups.mean(1, keepdims=True)).reshape(-1)
    return advantages


DUAL_CLIP = 3.0  # PPO's ratio bound for negative advantages (arXiv:1912.09729)


class ISCorrection(NamedTuple):
    """One point on the IS-correction grid. TIS, MIS, PPO, DAPO, GSPO, TOPR and
    DPPO are all spellings of it (table in README.md)."""
    level: str = "token"   # token | seq_mean (mean log r) | seq_sum (sum log r: trajectory ratio)
    stat: str = "ratio"    # band tests the ratio r, or the TV distance |q - p| (DPPO; high = delta)
    low: float = 0.0       # band [low, high] on r; 0 / inf leave a side open
    high: float = math.inf
    outside: str = "drop"  # clamp: w = nearest bound | drop: w = 0
    sign: str = "both"     # which advantages get corrected: both | neg | pos (others w = 1)
                           # | ppo: block only moves away from q; A>0: (0, high], A<0: [low, DUAL_CLIP]

    def check(self):
        choices = dict(level=("token", "seq_mean", "seq_sum"), stat=("ratio", "tv"), outside=("clamp", "drop"),
                       sign=("both", "neg", "pos", "ppo"))
        for field, allowed in choices.items():
            assert getattr(self, field) in allowed, f"rl.is.{field} must be one of {allowed}"
        assert self.stat == "ratio" or self.level != "seq_sum", "seq_sum requires stat=ratio"
        return self


def log_mass(log_p, log_q, positive, c, tv=None):
    """log(q * w(p/q)) under correction c (-inf where w = 0, log q for no
    correction). Log space keeps the common in-band case q * (p/q) = p exact
    and overflow-free. is_weight evaluates it at p = r, q = 1."""
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else -math.inf
    hi = log_q + math.log(c.high) if c.high < math.inf else math.inf
    if c.stat == "tv":
        in_band = tv <= c.high
        if c.sign == "ppo":
            in_band |= jnp.where(positive, log_p <= log_q, log_p >= log_q)
    elif c.sign == "ppo":
        in_band = jnp.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + math.log(DUAL_CLIP)))
    else:
        in_band = (log_p >= lo) & (log_p <= hi)
    log_m = jnp.where(in_band, log_p, jnp.clip(log_p, lo, hi) if c.outside == "clamp" else -jnp.inf)
    if c.sign == "neg":
        log_m = jnp.where(positive, log_q, log_m)
    if c.sign == "pos":
        log_m = jnp.where(positive, log_m, log_q)
    return log_m


def is_weight(log_ratio, positive, c, tv=None):
    """Detached IS weight w(r), with r = exp(log_ratio) clipped to e^±20."""
    return sg(jnp.exp(log_mass(jnp.clip(log_ratio, -20.0, 20.0), 0.0, positive, c, tv)))


def take_vocab(logp, ids):
    """take_along_axis over a [B, C, V] vocab axis that may be TP-sharded (the
    gather drops that axis, so its output sharding must be explicit)."""
    spec = jax.typeof(logp).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(logp, ids, -1)
    b = jnp.arange(ids.shape[0])[:, None, None]
    c = jnp.arange(ids.shape[1])[None, :, None]
    return logp.at[b, c, ids].get(out_sharding=P(*spec[:-1], None))


def sampler_center(logp_vocab, logp, logq, targets, positive, c=None, topk_ids=None, topk_logq=None,
                   vocab_logq=None):
    """Centering term m = sum_v pi(v) log p(v), pi detached, such that the
    corrected score w(y) (log p(y) - m) has zero mean under y ~ q whatever
    the advantage: pi(v) = q(v) w(v) / sum_u q(u) w(u). With no correction
    (w = 1), pi = q and the score's logit gradient is e_y - q.

    q is the sampler's next-token distribution, as far as it is known:
      - vocab_logq (full sampler logprobs): exactly;
      - otherwise on a set S = recorded top-k ids plus the sampled token y.
        Off S the remaining sampler mass 1 - q(S) is spread in proportion to
        the trainer's p: q(v) = p(v) (1 - q(S)) / (1 - p(S)), i.e. the
        sampler is assumed to agree with the trainer up to scale on the tail.
    """
    if vocab_logq is not None:
        logq_s, logp_s, known = vocab_logq.astype(jnp.float32), logp_vocab, None
    elif topk_ids is not None:
        topk_ids = topk_ids.astype(targets.dtype)
        logp_k = take_vocab(logp_vocab, topk_ids)
        y_new = ~(topk_ids == targets[..., None]).any(-1, keepdims=True)  # y not already among the top-k
        logq_s = jnp.concatenate([topk_logq.astype(jnp.float32), logq[..., None]], -1)
        logp_s = jnp.concatenate([logp_k, logp[..., None]], -1)
        known = jnp.concatenate([jnp.ones_like(logp_k, bool), y_new], -1)
    else:
        logq_s, logp_s, known = logq[..., None], logp[..., None], None
    q_s = sg(jnp.exp(logq_s))
    p_s = sg(jnp.exp(logp_s))
    if known is not None:
        q_s, p_s = jnp.where(known, q_s, 0.0), jnp.where(known, p_s, 0.0)
    pos = positive[..., None]
    w_s = is_weight(sg(logp_s) - logq_s, pos, c, jnp.abs(q_s - p_s)) if c else 1.0
    num = (q_s * w_s * logp_s).sum(-1)
    den = (q_s * w_s).sum(-1)
    if vocab_logq is None:  # tail: q(v) = scale * p(v), ratio p/q = 1/scale on every tail token
        tail_q = jnp.clip(1.0 - q_s.sum(-1), 0.0, 1.0)
        tail_p = jnp.clip(1.0 - p_s.sum(-1), 0.0, 1.0)
        has_tail = tail_p > 1e-6
        scale = jnp.where(has_tail, tail_q / jnp.maximum(tail_p, 1e-6), 0.0)
        log_scale = jnp.log(jnp.maximum(scale, 1e-30))
        w_t = is_weight(-log_scale, positive, c, jnp.zeros_like(scale)) if c else 1.0
        plogp_tail = (sg(jnp.exp(logp_vocab)) * logp_vocab).sum(-1) - (p_s * logp_s).sum(-1)
        num = num + jnp.where(has_tail, w_t * scale, 0.0) * plogp_tail
        den = den + jnp.where(has_tail, w_t * tail_q, 0.0)
    return jnp.where(den > 0, num / jnp.maximum(den, 1e-30), logp)  # all weights 0: score 0


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """Sampler-centered REINFORCE loss on sampled rollouts, with an optional
    ISCorrection. Returns (loss, metrics).

    The score of a sampled token y ~ q is centered under the sampler:
        s(y) = log p(y) - sum_v sg(q(v)) log p(v),   logit gradient e_y - q,
    instead of log p(y) (logit gradient e_y - p, whose mean under q is q - p
    = -grad KL(q || p): a hidden term that, multiplied by the advantage,
    pushes the trainer away from the sampler when advantages are negative).
    See REPORT.md. Uses the sampler's full or top-k logprobs when the batch
    has them (sampler.vocab_logprobs), and otherwise only the sampled token's.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens;
    optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K] or
    sampling_vocab_logprobs [B, T, V].
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs", "sampling_vocab_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        center = sampler_center(logp_vocab, logp, logq, b["targets"], positive, token_c, b.get("sampler_topk_ids"),
                                b.get("sampler_topk_logprobs"), b.get("sampling_vocab_logprobs"))
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * (logp - center), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv}
        return {k: (v * b["mask"]).sum(-1) for k, v in terms.items()}

    chunks = jax.lax.map(jax.checkpoint(chunk_sums), jnp.arange(0, seq_len, chunk))
    sums = {k: v.sum(0) for k, v in chunks.items()}  # per-sequence sums [B]

    n_tokens = jnp.maximum(mask.sum(), 1)
    advantages = batch["advantages"]
    frac_masked = sums["masked"].sum() / n_tokens
    if seq_c:
        seq_tokens = jnp.maximum(mask.sum(1), 1)
        log_ratio = -sums["kl_samp_train"]
        if seq_c.level == "seq_mean":
            log_ratio = log_ratio / seq_tokens
        w = is_weight(log_ratio, advantages >= 0, seq_c, sums["tv"] / seq_tokens)
        advantages = advantages * w
        frac_masked = (w == 0).mean()  # fraction of sequences
    loss = -(advantages * sums["score"]).sum() / n_tokens
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
