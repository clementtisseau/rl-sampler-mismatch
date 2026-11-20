"""Sampler-centered REINFORCE with importance-sampling corrections.

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


def sampler_centering(logp_vocab, logp, logq, topk_logq, topk_ids, positive, c):
    """Detached weights of the sampler-centered score (REPORT.md).

    Returns (w_y, terms): w_y is the IS weight of the sampled token and
    `terms` a [B, C] quantity whose gradient is  w_y * sum_v m_v grad log p_v,
    with m = q w / sum(q w) the measure under which the per-token score
    w_y grad log p_y has expectation  sum(q w) * sum_v m_v grad log p_v.
    The surrogate score is  w_y * log p_y - terms,  whose gradient
    w_y (grad log p_y - E_m[grad log p]) has zero mean under the sampler for
    any advantage that is constant given the prefix.

    q is known on the sampler's top-k tokens (a set that depends on the prefix
    only, not on the sampled token); on the remaining tail it is approximated
    by p rescaled to the remaining sampler mass. Without top-k logprobs the
    known set is the sampled token alone.
    """
    p_vocab = sg(jnp.exp(logp_vocab))
    pos3 = positive[..., None]
    # tokens with known sampler probability
    if topk_ids is not None:
        logp_k = take_vocab(logp_vocab, topk_ids)                        # [B, C, K]
        logq_k = topk_logq.astype(jnp.float32)
        y_extra = jnp.zeros_like(logp)
    else:  # fallback without top-k: the sampled token is the only one whose q is known
        logp_k = jnp.zeros(logp.shape + (0,), jnp.float32)
        logq_k = logp_k
        y_extra = jnp.ones_like(logp)
    P_known = (sg(jnp.exp(logp_k))).sum(-1) + y_extra * sg(jnp.exp(logp))
    Q_known = jnp.exp(logq_k).sum(-1) + y_extra * jnp.exp(logq)
    P_tail = jnp.maximum(1.0 - P_known, 0.0)
    Q_tail = jnp.clip(1.0 - Q_known, 0.0, 1.0)
    has_tail = P_tail > 1e-6
    # IS weights on the vocabulary (w = 1 without a correction)
    if c is None:
        w_k = jnp.ones_like(logq_k)
        w_y = jnp.ones_like(logp)
        w_tail = jnp.ones_like(logp)
    else:
        w_k = is_weight(sg(logp_k) - logq_k, pos3, c, sg(jnp.abs(jnp.exp(logq_k) - jnp.exp(logp_k))))
        w_y = is_weight(sg(logp) - logq, positive, c, sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp))))
        tail_log_ratio = jnp.log(jnp.maximum(P_tail, 1e-30)) - jnp.log(jnp.maximum(Q_tail, 1e-30))
        w_tail = is_weight(sg(tail_log_ratio), positive, c, jnp.zeros_like(logp))
    # unnormalized measure q*w on known tokens and on the tail (tail: q_v = p_v Q_tail / P_tail)
    qw_k = jnp.exp(logq_k) * w_k                                         # [B, C, K]
    qw_y = y_extra * jnp.exp(logq) * w_y
    tail_scale = jnp.where(has_tail, Q_tail / jnp.maximum(P_tail, 1e-6), 0.0) * w_tail
    qw_tail = tail_scale * P_tail
    Z = qw_k.sum(-1) + qw_y + qw_tail
    # sum_v (q w)_v log p_v; tail part via the full sum of p log p minus its known part
    plogp_all = (p_vocab * logp_vocab).sum(-1)
    plogp_known = (sg(jnp.exp(logp_k)) * logp_k).sum(-1) + y_extra * sg(jnp.exp(logp)) * logp
    s = (sg(qw_k) * logp_k).sum(-1) + sg(qw_y) * logp + sg(tail_scale) * (plogp_all - plogp_known)
    coef = sg(jnp.where(Z > 0, w_y / jnp.maximum(Z, 1e-30), 0.0))
    return sg(w_y), coef * s


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """Sampler-centered REINFORCE loss on sampled rollouts, with an optional
    ISCorrection. Returns (loss, metrics).

    The per-token score  w grad log p(y)  is centered by its expectation
    under the sampler (REPORT.md): the gradient is
        A * w_y * (grad log p(y) - E_{v ~ m}[grad log p(v)]),   m ∝ q * w(p/q),
    which has zero mean under the sampler q when A is constant given the
    prefix, so a mismatch q != p no longer adds an advantage-baseline drift
    along (q - p). With q == p it equals plain REINFORCE (E_p grad log p = 0).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B],
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens, and optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K].
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
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        valid = b["mask"]
        logq_safe = jnp.where(valid, logq, sg(logp))
        topk_ids = b.get("sampler_topk_ids")
        topk_logq = b.get("sampler_topk_logprobs")
        if topk_ids is not None:
            # outside the completion the sampler's top-k is padding: use p there (masked out anyway)
            topk_logq = jnp.where(valid[..., None], topk_logq.astype(jnp.float32),
                                  sg(take_vocab(logp_vocab, topk_ids)))
        w, center = sampler_centering(logp_vocab, logp, logq_safe, topk_logq, topk_ids,
                                      positive, token_c)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp - center, "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv}
        return {k: jnp.where(valid, v, 0).sum(-1) for k, v in terms.items()}

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
