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


def sampler_baseline(logp_vocab, logp_y, logq_y, top_ids, top_logq, positive, c):
    """Per-token score baseline that makes the score zero-mean under the
    SAMPLER q instead of the trainer p (REPORT.md).

    With weights w_v (w = 1 without a token-level correction) the score of the
    sampled token y is  w_y * (log p_y - sum_v b_v log p_v),  b = q*w / <q, w>
    (b detached).  In logit space its gradient is  w_y (e_y - b),  and
    E_{y~q}[w_y (e_y - b)] = q*w - <q, w> b = 0  exactly, for any p.

    q is known on the sampler's top-k ids; the unreported tail mass 1 - S is
    spread like p over the tail (q_v = (1 - S) p_v / (1 - P_K)), which is
    exact when the top-k covers the vocabulary and a second-order error
    otherwise.  Returns (detached weight w_y, baseline sum_v b_v log p_v).
    """
    logp_top = take_vocab(logp_vocab, top_ids)                      # [B, C, K]
    q_top = jnp.exp(top_logq.astype(jnp.float32))
    p_vocab = sg(jnp.exp(logp_vocab))
    p_top = sg(jnp.exp(logp_top))
    q_tail = jnp.clip(1.0 - q_top.sum(-1), 0.0, 1.0)                 # [B, C]
    p_tail = jnp.clip(1.0 - p_top.sum(-1), 0.0, 1.0)
    # sum_{v not in K} p_v log p_v  (detached weights, gradient through log p)
    plogp_all = (p_vocab * logp_vocab).sum(-1)
    plogp_tail = plogp_all - (p_top * logp_top).sum(-1)
    if c is None:
        w_top, w_tail = jnp.ones_like(q_top), jnp.ones_like(q_tail)
        w_y = jnp.ones_like(logp_y)
    else:
        pos = positive[..., None]
        w_top = is_weight(sg(logp_top) - top_logq, pos, c, sg(jnp.abs(q_top - p_top)))
        # in the tail p_v / q_v is the same for every v: (1 - P_K) / (1 - S)
        tail_lr = jnp.log(jnp.maximum(p_tail, 1e-30)) - jnp.log(jnp.maximum(q_tail, 1e-30))
        w_tail = is_weight(sg(tail_lr), positive, c, sg(jnp.abs(q_tail - p_tail)))
        w_y = is_weight(sg(logp_y) - logq_y, positive, c, sg(jnp.abs(jnp.exp(logq_y) - jnp.exp(logp_y))))
    mass_top = q_top * w_top
    mass_tail = q_tail * w_tail                                       # total tail mass of q*w
    norm = mass_top.sum(-1) + mass_tail
    tail_coef = mass_tail / jnp.maximum(p_tail, 1e-30)                # q_v w_v / p_v on the tail
    baseline = ((mass_top * logp_top).sum(-1) + jnp.where(p_tail > 1e-30, tail_coef, 0.0) * plogp_tail)
    baseline = sg(jnp.where(norm > 0, 1.0 / jnp.maximum(norm, 1e-30), 0.0)) * baseline
    # (mass_top, tail_coef, norm are built from detached quantities only)
    return sg(w_y), baseline


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts with a sampler-centered score, and
    an optional ISCorrection. Returns (loss, metrics).

    The score of each sampled token is centered by its expectation under the
    sampler q:  log p(y_t) - E_{v~q}[log p(v)]  (REPORT.md). Its gradient is
    zero-mean under q at every prefix, as the plain score log p(y_t) is under
    p, so the advantage-independent drift  E_q[A | prefix] * (q - p)  that the
    plain score produces when q != p is removed. On-policy (q = p) this is
    exactly REINFORCE.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B],
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens, and (used when present) sampler_topk_ids / sampler_topk_logprobs
    [B, T, K], the sampler's top-k next-token logprobs. Without the top-k the
    centering falls back to the importance-weighted form, whose logit
    gradient  (q_y/p_y)^-1 e_y - p  is also zero-mean under q.
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    have_topk = "sampler_topk_ids" in batch and "sampler_topk_logprobs" in batch
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logits = head(b["hidden"]).astype(jnp.float32)
        logp_vocab = jax.nn.log_softmax(logits)
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        pos = jnp.broadcast_to(positive, logp.shape)
        if have_topk:
            w, baseline = sampler_baseline(logp_vocab, logp, logq, b["sampler_topk_ids"],
                                           b["sampler_topk_logprobs"], pos, token_c)
            score = w * (logp - baseline)
        else:
            # w_y log p_y + (1 - w_y) logsumexp-part: gradient  w_y e_y - p  with
            # w = p/q, zero-mean under q;  log p_y = z_y - lse, so this is
            # w_y z_y - lse = w_y log p_y - (1 - w_y) lse. A correction then
            # bounds w as usual.
            ratio_c = token_c or ISCorrection()
            w = is_weight(logp - logq, pos, ratio_c, tv)
            lse = jax.nn.logsumexp(logits, -1)
            score = w * logp - (1.0 - w) * lse
            score = score - sg(score) + sg(w * logp)  # same value as the IS score, centered gradient
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": score, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
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
