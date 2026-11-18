"""REINFORCE with a sampler-centered score and importance-sampling corrections.

The score of each sampled token is centered under the sampler's next-token
distribution: log p(y_t) - sum_v m_t(v) log p_t(v) (see sampler_baseline and
REPORT.md). On-policy its gradient is plain REINFORCE's; under a sampler /
trainer mismatch it removes the bias A * E_q[grad log p] != 0.

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


def sampler_baseline(logp_vocab, targets, logq, positive, c, extra):
    """sum_v m(v) log p(v): the score's expected value under the sampler.

    m = q * w(p/q), normalized, is the distribution the (possibly IS-weighted)
    sampled score averages over: E_{y~q}[w(y) grad log p(y)] = Z * sum_v m(v)
    grad log p(v), with Z = sum_v q(v) w(v). Subtracting it centers the
    per-token score under q, so a reward-independent advantage has zero
    expected update, as in the on-policy case (where it is sum_v p grad log p = 0).

    q is known exactly on a set K of tokens: the sampled token (always), the
    sampler's top-k head (sampler.vocab_logprobs=k) or the whole vocabulary
    (sampler.vocab_logprobs=full). Off K its remaining mass 1 - q(K) is spread
    in proportion to p, the maximum-entropy completion that equals p when q = p.
    Every quantity except log p is detached.
    """
    if "sampling_vocab_logprobs" in extra:  # exact q on the whole vocabulary
        logq_vocab = extra["sampling_vocab_logprobs"].astype(jnp.float32)
        tv_vocab = jnp.abs(jnp.exp(logq_vocab) - jnp.exp(logp_vocab))
        log_m = log_mass(logp_vocab, logq_vocab, positive[..., None], c, tv_vocab)
        log_m = jnp.where(jnp.isfinite(logq_vocab), log_m, -jnp.inf)
        m = jax.nn.softmax(log_m, axis=-1)
        return (sg(m) * logp_vocab).sum(-1)

    # known set K: top-k ids, plus the sampled token when it is not among them
    lp_y = take_vocab(logp_vocab, targets[..., None])  # [B, C, 1]
    if "sampler_topk_ids" in extra:
        ids = extra["sampler_topk_ids"]
        lq_k = extra["sampler_topk_logprobs"].astype(jnp.float32)
        lp_k = take_vocab(logp_vocab, ids)
        y_new = ~(ids == targets[..., None]).any(-1, keepdims=True)
        lp_k = jnp.concatenate([lp_k, lp_y], -1)
        lq_k = jnp.concatenate([lq_k, logq[..., None]], -1)
        keep = jnp.concatenate([jnp.ones_like(ids, bool), y_new], -1)
    else:
        lp_k, lq_k, keep = lp_y, logq[..., None], jnp.ones_like(lp_y, bool)
    lq_k = jnp.where(keep, lq_k, -jnp.inf)
    sp_k, sq_k = sg(lp_k), sg(lq_k)
    slp = sg(logp_vocab)
    in_k = jnp.zeros(slp.shape, bool)  # membership of K over the vocabulary, built without a scatter
    spec = jax.typeof(slp).sharding.spec
    for j in range(lp_k.shape[-1]):
        ids_j = targets if j == lp_k.shape[-1] - 1 else extra["sampler_topk_ids"][..., j]
        hit = jax.nn.one_hot(ids_j, slp.shape[-1], dtype=jnp.bool_)
        if spec[-1] is not None:
            hit = jax.sharding.reshard(hit, spec)
        in_k |= hit
    # tail: q(v) = rho p(v) off K, rho = (1 - q(K)) / p(V \ K)
    q_k = jnp.exp(jax.nn.logsumexp(sq_k, -1))
    log_p_tail = jnp.log(jnp.maximum(jnp.where(in_k, 0.0, jnp.exp(slp)).sum(-1), 1e-30))
    log_rho = jnp.log(jnp.maximum(1.0 - q_k, 1e-30)) - log_p_tail  # [B, C]
    pos = positive[..., None]  # [B, C, 1]
    log_q_tail = slp + log_rho[..., None]
    tv_tail = jnp.abs(jnp.exp(log_q_tail) - jnp.exp(slp))
    m_tail = jnp.where(in_k, 0.0, jnp.exp(log_mass(slp, log_q_tail, pos, c, tv_tail)))  # [B, C, V]
    tv_k = jnp.abs(jnp.exp(sq_k) - jnp.exp(sp_k))
    m_k = jnp.where(keep, jnp.exp(log_mass(sp_k, sq_k, pos, c, tv_k)), 0.0)  # [B, C, |K|]
    z = jnp.maximum(m_tail.sum(-1) + m_k.sum(-1), 1e-30)
    num = (m_tail * logp_vocab).sum(-1) + (m_k * lp_k).sum(-1)
    return num / sg(z)


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts with a sampler-centered score, and
    an optional ISCorrection. Returns (loss, metrics).

    The score of each completion token is log p(y_t) - sum_v m_t(v) log p_t(v),
    with m_t the sampler's next-token distribution (reweighted by the
    correction, see sampler_baseline). On-policy (q = p) its gradient equals
    plain REINFORCE's; under a mismatch it removes the term
    A * E_q[grad log p] = -A * grad KL(q || p), which otherwise drives the
    trainer along the sampler's error every step (REPORT.md).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally sampler_topk_ids / sampler_topk_logprobs [B, T, k] or
    sampling_vocab_logprobs [B, T, V] (sampler.vocab_logprobs=k | full).
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
        pos_t = jnp.broadcast_to(positive, logq.shape)
        baseline = sampler_baseline(logp_vocab, b["targets"], logq, pos_t, token_c, b)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * (logp - baseline), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
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
