"""REINFORCE with importance-sampling corrections.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.
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


TAIL_FLOOR = 1e-6  # floor on the trainer's tail mass p_R (float32 resolution of 1 - sum p_K)


def sampler_centered_terms(logp_vocab, logp, logq, ids, logq_k, positive, c):
    """Per-token REINFORCE term, centered under the sampler.

    The plain term w_y log p(y) has, under y ~ q, the expected logit gradient
    q*w - (sum q*w) p, which for w = 1 is q - p != 0: the baseline no longer
    cancels. We subtract its expectation, the control variate
        CV = sum_v sg(c_v) log p(v),   c_v = q(v) w(p(v)/q(v)),
    so that d/dz [w_y log p(y) - CV] = w_y e_y - c has zero mean under q.

    q is known on K (the sampler's top-k, or only the sampled token); the
    rest of the vocabulary R is one bucket of mass q_R, spread like p:
    c_u = q_R w(p_R/q_R) p(u)/p_R for u in R. Since sum_v sg(p_v) log p(v) has
    zero gradient, sum_{u in R} sg(p_u) log p(u) = H_sum - sum_K sg(p) log p.
    Returns (centered term, plain term w_y log p(y)).
    """
    pos = positive[..., None]
    lp_k = take_vocab(logp_vocab, ids)                                  # trainer logprobs on K
    lq_k = logq_k
    tv_k = sg(jnp.abs(jnp.exp(lq_k) - jnp.exp(lp_k)))
    w_k = is_weight(sg(lp_k) - lq_k, pos, c, tv_k)
    c_k = sg(jnp.exp(lq_k) * w_k)                                       # [B, C, K]
    q_r = sg(jnp.clip(-jnp.expm1(jnp.minimum(jax.nn.logsumexp(lq_k, -1), 0.0)), 0.0, 1.0))
    p_r = sg(jnp.clip(-jnp.expm1(jnp.minimum(jax.nn.logsumexp(lp_k, -1), 0.0)), TAIL_FLOOR, 1.0))
    log_ratio_r = jnp.log(p_r) - jnp.log(jnp.maximum(q_r, 1e-30))
    tv_r = sg(jnp.abs(q_r - p_r))
    w_r = jnp.where(q_r > 0, is_weight(log_ratio_r, positive, c, tv_r), 0.0)
    plogp_all = (sg(jnp.exp(logp_vocab)) * logp_vocab).sum(-1)          # zero gradient, value only
    plogp_k = (sg(jnp.exp(lp_k)) * lp_k).sum(-1)
    tail = sg(q_r * w_r / p_r) * (plogp_all - plogp_k)                  # sum_R c_u log p(u)
    cv = (c_k * lp_k).sum(-1) + tail
    w = is_weight(logp - logq, positive, c, sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp))))
    score = w * logp
    return score - cv, score, w


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection, and
    with every token's score centered under the sampler distribution q
    (see sampler_centered_terms and REPORT.md). Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K], the
    sampler's top-k logprobs (without them, K is the sampled token alone).
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    has_topk = "sampler_topk_ids" in batch and "sampler_topk_logprobs" in batch
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        if has_topk:
            ids, logq_k = b["sampler_topk_ids"], b["sampler_topk_logprobs"].astype(jnp.float32)
        else:
            ids, logq_k = b["targets"][..., None], logq[..., None]
        centered, score, w = sampler_centered_terms(logp_vocab, logp, logq, ids, logq_k, positive, token_c)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": centered, "raw_score": sg(score), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv}
        m = b["mask"]
        return {k: jnp.where(m, v, 0).sum(-1) for k, v in terms.items()}

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
    metrics["centering"] = (sums["raw_score"] - sums["score"]).sum() / n_tokens  # mean control variate
    return loss, metrics | {"frac_masked": frac_masked}
