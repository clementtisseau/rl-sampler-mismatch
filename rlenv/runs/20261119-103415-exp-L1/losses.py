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


def sampler_baseline(logp_vocab, logp, logq, top_ids, top_logq, positive, c, tv):
    """Detached weights a_v (on the sampler's top-k ids) such that
    grad sum_v a_v log p_v = mu - p in logit space, where mu is the
    distribution of the corrected sampling mass q_v w(p_v/q_v), normalized.

    The tail outside the top-k is not observed; it is modelled as carrying
    the sampler's remaining mass Q_tail spread proportionally to p:
    mu_v = c * p_v on the tail. Since sum_v sg(p_v) grad log p_v = 0, the
    tail's contribution moves onto the top-k ids as -c * p_v."""
    logp_k = take_vocab(logp_vocab, top_ids)                      # [B, C, K]
    logq_k = jnp.maximum(top_logq.astype(jnp.float32), -1e4)       # -inf (masked vocab) -> mass 0
    p_k, q_k = sg(jnp.exp(logp_k)), jnp.exp(logq_k)
    p_tail = jnp.clip(1.0 - p_k.sum(-1), 1e-6, 1.0)                # p mass off the sampler's top-k
    q_tail = jnp.clip(1.0 - q_k.sum(-1), 0.0, 1.0)                 # sampler mass off its top-k
    pos = positive[..., None]
    tv_k = None if tv is None else jnp.abs(q_k - p_k)
    m_k = jnp.exp(log_mass(sg(logp_k), logq_k, pos, c, tv_k))    # q_v w(p_v / q_v) on the top-k
    # w is a function of p/q, constant over the tail under the model q_v = p_v Q/P
    m_tail = jnp.exp(log_mass(jnp.log(p_tail), jnp.log(jnp.maximum(q_tail, 1e-30)), positive, c,
                              None if tv is None else jnp.abs(q_tail - p_tail)))
    m_tail = jnp.where(q_tail > 0, m_tail, 0.0)
    total = m_k.sum(-1) + m_tail
    ok = total > 1e-12
    mu_k = jnp.where(ok[..., None], m_k / jnp.where(ok, total, 1.0)[..., None], p_k)
    mu_tail = jnp.where(ok, m_tail / jnp.where(ok, total, 1.0), p_tail)
    ratio_tail = mu_tail / p_tail                                  # c: tail mu_v = c * p_v
    a_k = sg(mu_k - ratio_tail[..., None] * p_k)
    return (a_k * logp_k).sum(-1), sg(mu_k.sum(-1))


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    The score of every sampled token is centered under the distribution the
    token was actually drawn from (the sampler's, reweighted by the IS
    correction), not under the trainer's: in logit space the per-token
    gradient is w(y) A (e_y - mu) with mu = normalize(q * w) instead of
    w(y) A (e_y - p). REINFORCE's implicit "E[score] = 0" only holds for
    the distribution that generated the token; with p != q the uncentered
    score adds a reward-independent drift E_q[A|s] (q - p) that pulls the
    trainer toward (or pushes it away from) the sampler at every step. See
    REPORT.md.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens,
    and (optionally) sampler_topk_ids / sampler_topk_logprobs [B, T, K], the
    sampler's top-k next-token logprobs. Without the top-k, the score is
    centered under p as in plain REINFORCE.
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
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if has_topk:
            base, q_cover = sampler_baseline(logp_vocab, logp, logq, b["sampler_topk_ids"],
                                             b["sampler_topk_logprobs"], positive, token_c,
                                             tv if token_c is not None and token_c.stat == "tv" else None)
        else:
            base, q_cover = jnp.zeros_like(logp), jnp.zeros_like(logp)
        base = jnp.where(b["mask"], base, 0.0)  # rows past an early stop hold stale top-k
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * (logp - base), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv, "q_cover": q_cover}
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
