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


def xlogy(m, logp):
    """m * logp with 0 where m = 0 (masked vocabulary entries may have logp = -inf)."""
    return jnp.where(m > 0, m * jnp.where(m > 0, logp, 0.0), 0.0)


def sampler_measure(logp_vocab, b, positive, c):
    """Per-position centering of the score: returns sum_v m_v log p_v / sum_v m_v,
    with m_v = q_v w(p_v/q_v) the effective measure that generates the weighted
    sample (m = q without a correction). Its gradient in the logits is
    E_m[e_v] - p, exactly the drift E_q[w (e_y - p)] / E_q[w] that a constant
    advantage produces under the sampler (REPORT.md). None when the sampler's
    distribution was not recorded. p-weights are detached: only log p carries
    gradient."""
    if "sampling_vocab_logprobs" in b:  # full sampler distribution
        logq_vocab = b["sampling_vocab_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq_vocab) - jnp.exp(logp_vocab)))
        log_m = log_mass(sg(logp_vocab), logq_vocab, positive[..., None], c, tv)
        m = sg(jnp.exp(log_m))
        return xlogy(m, logp_vocab).sum(-1) / jnp.maximum(m.sum(-1), 1e-30)
    if "sampler_topk_logprobs" in b:
        # Sampler head K only. Head tokens keep the measure q w; a sampled tail
        # token gets the exact ratio p/q (applied in loss_fn_rl), so the tail's
        # measure is p, which is known: the centering stays exact.
        logq_k = b["sampler_topk_logprobs"].astype(jnp.float32)
        logp_k = take_vocab(logp_vocab, b["sampler_topk_ids"])
        p_vocab, p_k = sg(jnp.exp(logp_vocab)), sg(jnp.exp(logp_k))
        tail_plogp = xlogy(p_vocab, logp_vocab).sum(-1) - xlogy(p_k, logp_k).sum(-1)  # sum over the tail of p log p
        p_tail = jnp.clip(1.0 - p_k.sum(-1), 0.0, 1.0)
        tv_k = sg(jnp.abs(jnp.exp(logq_k) - p_k))
        m_k = sg(jnp.exp(log_mass(sg(logp_k), logq_k, positive[..., None], c, tv_k)))
        total = jnp.maximum(m_k.sum(-1) + p_tail, 1e-30)
        return (xlogy(m_k, logp_k).sum(-1) + tail_plogp) / total
    return None


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts with a sampler-centered score, and an
    optional ISCorrection. Returns (loss, metrics).

    Each token contributes A * w * (log p(y) - E_m[log p]), m = q w the
    measure that produced the weighted sample. Under a mismatched sampler the
    plain score log p(y) has nonzero mean E_q[grad log p] = -grad KL(q || p),
    so every constant part of the advantage drives p towards (A > 0) or away
    from (A < 0) the sampler, and the drift compounds as the sampler follows
    the trainer. Centering at the sampler makes the score zero-mean under the
    sampling distribution, as it is on-policy; with q = p it is plain
    REINFORCE. Without recorded sampler distributions (sampler.vocab_logprobs)
    the centering term is not computable, and the same zero-mean condition is
    met by the token ratio w = p/q (m = p, whose centering term vanishes).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally sampling_vocab_logprobs [B, T, V] or sampler_topk_ids /
    sampler_topk_logprobs [B, T, K].
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    has_vocab = "sampling_vocab_logprobs" in batch or "sampler_topk_logprobs" in batch
    if not has_vocab and token_c is None and seq_c is None:
        token_c = ISCorrection()  # no centering available: unbiased token ratio p/q
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if "sampler_topk_ids" in b:  # sampled token outside the sampler's head: ratio p/q
            in_head = (b["sampler_topk_ids"] == b["targets"][..., None]).any(-1)
            w = jnp.where(in_head, w, is_weight(logp - logq, positive, ISCorrection(), tv))
        center = sampler_measure(logp_vocab, b, positive, token_c)
        score = logp if center is None else logp - center
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * score, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv,
                 "is_weight": w}
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
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "is_weight") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked, "sampler_centered": jnp.asarray(has_vocab, jnp.float32)}
