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


def sampler_center(logp_vocab, logq_tok, logp_tok, top_ids, top_logq, positive, token_c):
    """Detached centering distribution c over the vocabulary such that, for a
    token y ~ q at this position, E_q[w(y) (e_y - c)] = 0 (in logit space).

    c_v = q(v) w(v) / sum_u q(u) w(u): the sampler's distribution reweighted by
    the correction's weight (c = q without a correction). It is exact on the
    sampler's top-k (ids and logprobs given by the sampler) and the remaining
    sampler mass q_T is spread like the trainer's tail, q(v) ~ q_T p(v)/p_T.

    Returns sum_v c_v log p(v) (differentiable in log p, weights detached)."""
    logp_k = take_vocab(logp_vocab, top_ids)                                   # [B, C, K]
    q_k = jnp.exp(top_logq)
    q_T = jnp.clip(1.0 - q_k.sum(-1), 0.0, 1.0)                                # sampler tail mass
    log_p_T = jnp.log(jnp.clip(1.0 - jnp.exp(logp_k).sum(-1), 1e-30, 1.0))   # trainer tail mass
    # trainer's distribution restricted to the tail, renormalized (stable log-softmax)
    in_top = jnp.zeros(logp_vocab.shape, bool)
    in_top = in_top.at[jnp.arange(top_ids.shape[0])[:, None, None],
                       jnp.arange(top_ids.shape[1])[None, :, None], top_ids].set(True)
    tail_logp = jnp.where(in_top, -1e30, logp_vocab)  # finite: an empty tail stays NaN-free
    tail_dist = sg(jax.nn.softmax(tail_logp, -1))
    tail_mean_logp = (tail_dist * jnp.where(in_top, 0.0, logp_vocab)).sum(-1)  # E_{p_tail}[log p]
    if token_c is not None:
        pos = positive[..., None]
        tv_k = sg(jnp.abs(q_k - jnp.exp(logp_k)))
        w_k = is_weight(sg(logp_k) - top_logq, pos, token_c, tv_k)
        log_q_T = jnp.log(jnp.maximum(q_T, 1e-30))
        tv_T = sg(jnp.abs(q_T - jnp.exp(log_p_T)))
        w_T = is_weight(sg(log_p_T) - log_q_T, positive, token_c, tv_T)
    else:
        w_k, w_T = jnp.ones_like(q_k), jnp.ones_like(q_T)
    m_k, m_T = sg(q_k * w_k), sg(q_T * w_T)
    Z = m_k.sum(-1) + m_T
    safe_Z = jnp.maximum(Z, 1e-20)
    centre = ((m_k * logp_k).sum(-1) + m_T * tail_mean_logp) / safe_Z
    return jnp.where(Z > 0, centre, logp_tok)  # all mass dropped: no update


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection, and
    with the score of every token centred under the sampler's (reweighted)
    next-token distribution (see REPORT.md). Returns (loss, metrics).

    Per token, the score log p(y) is replaced by log p(y) - sum_v c_v log p(v)
    with c a detached distribution: c = q (the sampler's next-token
    distribution) without correction, c ~ q w(p/q) with a token correction.
    Its logit gradient is e_y - c instead of e_y - p, whose expectation
    under y ~ q is zero for any advantage that does not depend on y: the
    mismatch no longer produces a systematic drift E_q[A (e_y - p)] = A (q - p).
    When q = p it is exactly REINFORCE (the extra term has zero gradient).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B],
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens, and (optionally) sampler_topk_ids / sampler_topk_logprobs [B, T, K],
    the sampler's top-k next-token logprobs. Without the top-k, the centre
    falls back to the sampled token alone (plain REINFORCE).
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
    has_topk = "sampler_topk_ids" in per_token and "sampler_topk_logprobs" in per_token

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if has_topk:
            centre = sampler_center(logp_vocab, logq, logp, b["sampler_topk_ids"],
                                    b["sampler_topk_logprobs"].astype(jnp.float32), positive, token_c)
        else:
            centre = sg(logp)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * (logp - centre), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
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
