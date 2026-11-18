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


def sampler_baseline(logp_vocab, logp, logq, targets, positive, c, b, ratio_cap=100.0):
    """Detached-weight surrogate whose gradient is E_{y'~q}[w(y') grad log p(y')]
    at every position: the mean of the (weighted) score under the sampler.

    q is known on the sampler's recorded head (`sampler_topk_*` or the full
    `sampling_vocab_logprobs`) and at the sampled token; the rest of its mass,
    Q_tail, is spread over the unrecorded tokens in proportion to p. Uses
    sum_v p_v grad log p_v = 0, so the tail needs no full-vocabulary sum:
    sum_{v in tail} p_v grad log p_v = -sum_{v known} p_v grad log p_v.
    Returns (surrogate [B, C], tail mass of q [B, C])."""
    p_y = jnp.exp(sg(logp))
    if "sampling_vocab_logprobs" in b:  # full q: exact, no tail
        logq_vocab = b["sampling_vocab_logprobs"].astype(jnp.float32)
        lp = sg(logp_vocab)
        tv = jnp.abs(jnp.exp(logq_vocab) - jnp.exp(lp))
        m = jnp.exp(log_mass(lp, logq_vocab, positive[..., None], c, tv))
        return (sg(m) * logp_vocab).sum(-1), jnp.zeros_like(logp)
    if "sampler_topk_ids" in b:
        ids = b["sampler_topk_ids"]
        logq_h = b["sampler_topk_logprobs"].astype(jnp.float32)
        logp_h = take_vocab(logp_vocab, ids)                            # [B, C, k]
        in_head = (ids == targets[..., None]).any(-1)                   # sampled token recorded?
    else:  # only the sampled token is known
        logq_h = logp_h = jnp.zeros(logp.shape + (0,), jnp.float32)
        in_head = jnp.zeros(logp.shape, bool)
    extra = (~in_head).astype(jnp.float32)                              # add y to the known set
    p_h, q_h = jnp.exp(sg(logp_h)), jnp.exp(logq_h)
    q_y = jnp.exp(logq)
    P_known = p_h.sum(-1) + extra * p_y
    Q_known = q_h.sum(-1) + extra * q_y
    P_tail = jnp.clip(1.0 - P_known, 1e-6, 1.0)
    Q_tail = jnp.clip(1.0 - Q_known, 0.0, 1.0)
    # sampler mass q_v w(p_v / q_v) on each known token
    m_h = jnp.exp(log_mass(sg(logp_h), logq_h, positive[..., None], c, jnp.abs(q_h - p_h)))
    m_y = jnp.exp(log_mass(sg(logp), logq, positive, c, jnp.abs(q_y - p_y)))
    known = (sg(m_h) * logp_h).sum(-1) + extra * sg(m_y) * logp
    # tail: q_v = Q_tail p_v / P_tail, ratio P_tail / Q_tail shared by all tail tokens
    log_Pt, log_Qt = jnp.log(P_tail), jnp.log(jnp.maximum(Q_tail, 1e-30))
    m_tail = jnp.where(Q_tail > 0, jnp.exp(log_mass(log_Pt, log_Qt, positive, c, jnp.abs(Q_tail - P_tail))), 0.0)
    coef = jnp.minimum(m_tail / P_tail, ratio_cap)                     # tail weight per unit of p
    tail = -coef * ((p_h * logp_h).sum(-1) + extra * p_y * logp)       # = coef * sum_tail p_v log p_v (gradient)
    return known + tail, Q_tail


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts with a sampler-centered score, and
    an optional ISCorrection. Returns (loss, metrics).

    Per completion token the score is centered under the sampler q:
        w(y) log p(y) - sum_v sg(q_v w(p_v/q_v)) log p_v,
    whose gradient has zero mean over y ~ q for every prefix. Plain
    REINFORCE's score grad log p(y) has mean sum_v q_v grad log p_v
    = -grad KL(q || p) != 0 when q != p, which turns every advantage offset
    into a drift of p toward (A > 0) or away from (A < 0) the sampler. See
    REPORT.md. If q = p the subtracted term has exactly zero gradient.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally the sampler's head sampler_topk_ids/_logprobs [B, T, k]
    (sampler.vocab_logprobs=k) or sampling_vocab_logprobs [B, T, V] (full).
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask, "positive": jnp.broadcast_to(positive, mask.shape)}
    per_token |= {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, b["positive"], token_c, tv)
        base, q_tail = sampler_baseline(logp_vocab, logp, logq, b["targets"], b["positive"], token_c, b)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp - base, "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv, "q_tail": sg(q_tail)}
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
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "q_tail") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
