"""REINFORCE with importance-sampling corrections and a sampler-centered score.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.

The score of each sampled token is centered under the sampler: the loss
term is A * (log p(y) - sum_v q~(v) log p(v)), whose logit gradient is
A * (e_y - q~) instead of A * (e_y - p). q~ is the sampler distribution
reconstructed from its top-k logprobs (and the sampled token's), with the
remaining mass spread like p. Under y ~ q this score has (near) zero mean,
so a constant advantage no longer moves the policy: the systematic drift
-E[A|s] * grad KL(q || p) of plain REINFORCE under a mismatched sampler is
removed. With q = p it reduces exactly to REINFORCE. See REPORT.md.
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


def sampler_baseline(logp_vocab, logp, b, positive, c):
    """sum_v m(v) log p(v), with m(v) = q~(v) w(p(v)/q~(v)) detached: the
    expectation of the per-token score w(y) log p(y) over y ~ q, so that
    w(y) log p(y) minus it has zero-mean gradient under the sampler. With no
    correction m = q~, and the logit gradient of the centered score is
    e_y - q~ instead of e_y - p.

    q~ is exact on the known support K (the sampler's top-k tokens plus the
    sampled token y, deduplicated) and puts the leftover sampler mass
    q_rest = 1 - q(K) on the rest of the vocabulary in proportion to p:
        sum_v q~(v) log p(v) = sum_{v in K} q(v) log p(v) + q_rest log p(not K).
    Without top-k logprobs K = {y}. With q = p, q~ = p and the baseline has
    zero gradient: plain REINFORCE."""
    y = b["targets"][..., None]
    logq_y = b["sampling_token_logprobs"][..., None].astype(jnp.float32)
    if "sampler_topk_ids" in b:
        ids, logq_k = b["sampler_topk_ids"], b["sampler_topk_logprobs"].astype(jnp.float32)
        # y joins K only if the sampler's top-k does not already contain it
        fresh = ~jnp.any(ids == y, -1, keepdims=True)
        ids = jnp.concatenate([ids, y], -1)
        logq_k = jnp.concatenate([logq_k, logq_y], -1)
        keep = jnp.concatenate([jnp.ones(ids.shape[:-1] + (ids.shape[-1] - 1,), bool), fresh], -1)
        # drop duplicate ids inside K (ties or padding in top-k): keep the first
        k = ids.shape[-1]
        dup = jnp.any((ids[..., :, None] == ids[..., None, :]) & jnp.tril(jnp.ones((k, k), bool), -1), -1)
        keep = keep & ~dup
        logp_k = take_vocab(logp_vocab, ids)
    else:
        ids, logq_k, logp_k, keep = y, logq_y, logp[..., None], jnp.ones_like(y, bool)
    logq_k = jnp.minimum(logq_k, 0.0)
    q_rest = sg(jnp.clip(1.0 - jnp.where(keep, jnp.exp(logq_k), 0.0).sum(-1), 0.0, 1.0))
    # log p(not K) as a logsumexp over the vocabulary with K masked out (exact,
    # no 1 - p(K) cancellation); its logit gradient is p|notK / p(notK) - p
    vocab = jnp.arange(logp_vocab.shape[-1], dtype=ids.dtype)
    in_k = jnp.zeros(logp_vocab.shape, bool)
    for j in range(ids.shape[-1]):
        in_k = in_k | ((vocab == ids[..., j:j + 1]) & keep[..., j:j + 1])
    in_k = in_k & ~in_k.all(-1, keepdims=True)  # K = whole vocab: q_rest = 0, keep it finite
    log_p_rest = jax.nn.logsumexp(jnp.where(in_k, -jnp.inf, logp_vocab), -1)
    log_q_rest = jnp.log(jnp.maximum(q_rest, 1e-30))
    # detached masses m = q * w(p/q) under the correction (m = q without one)
    tv_k = sg(jnp.abs(jnp.exp(logq_k) - jnp.exp(logp_k)))
    m_k = sg(jnp.exp(log_mass(jnp.clip(sg(logp_k), logq_k - 20.0, logq_k + 20.0), logq_k, positive[..., None], c, tv_k)))
    m_k = jnp.where(keep, m_k, 0.0)
    tv_rest = sg(jnp.abs(q_rest - jnp.exp(log_p_rest)))
    m_rest = sg(jnp.exp(log_mass(jnp.clip(sg(log_p_rest), log_q_rest - 20.0, log_q_rest + 20.0), log_q_rest,
                                 positive, c, tv_rest)))
    m_rest = jnp.where(q_rest > 0, m_rest, 0.0)
    return (m_k * jnp.where(keep, logp_k, 0.0)).sum(-1) + m_rest * log_p_rest


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens.
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        baseline = sampler_baseline(logp_vocab, logp, b, positive, token_c)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp - baseline, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
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
