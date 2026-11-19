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


def sampler_center_logmass(logp_vocab, logp, b, positive, c):
    """Log of the (unnormalized) centering mass m_v = q_v * w(p_v / q_v) over the
    vocabulary, returned in a factored, gather-only form (no [B, C, V] scatter):

      * on the sampler's top-k ids K: log q_k + log w_k   (exact q from the sampler)
      * off K: q is approximated by the leftover sampler mass spread like p,
        q_v ~= tau * p_v with tau = (1 - Q_K) / (1 - P_K), so the ratio p/q = 1/tau
        is constant there and m_v = tau * w(1/tau) * p_v.

    Returns (ids [B,C,K], log m on K [B,C,K], log of the tail factor
    tau * w(1/tau) [B,C]); m_v = exp(tail factor) * p_v off K.
    """
    ids = b["sampler_topk_ids"]
    log_q_k = b["sampler_topk_logprobs"].astype(jnp.float32)
    log_p_k = take_vocab(logp_vocab, ids)                                   # trainer logprobs on K
    q_tail = jnp.clip(1.0 - jnp.exp(log_q_k).sum(-1), 0.0, 1.0)
    p_tail = jnp.clip(1.0 - jnp.exp(log_p_k).sum(-1), 0.0, 1.0)
    log_tau = jnp.log(jnp.maximum(q_tail, 1e-30)) - jnp.log(jnp.maximum(p_tail, 1e-30))
    log_tau = jnp.where((q_tail <= 1e-7) | (p_tail <= 1e-7), -jnp.inf, log_tau)  # negligible tail
    if c is None or c.level != "token":
        log_m_k, log_tail = log_q_k, log_tau
    else:
        pos = positive[..., None]
        tv_k = jnp.abs(jnp.exp(log_q_k) - jnp.exp(log_p_k))
        log_m_k = log_mass(jnp.clip(log_p_k - log_q_k, -20.0, 20.0), 0.0, pos, c, tv_k) + log_q_k
        tv_tail = jnp.abs(q_tail - p_tail)
        log_w_tail = log_mass(jnp.clip(-log_tau, -20.0, 20.0), 0.0, positive, c, tv_tail)
        log_tail = jnp.where(jnp.isfinite(log_tau), log_tau + log_w_tail, -jnp.inf)
    return ids, sg(log_m_k), sg(log_tail)


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss with a sampler-centered score, on sampled rollouts, with an
    optional ISCorrection. Returns (loss, metrics).

    Per token the surrogate is  A * w(y) * (log p(y) - sum_v m_v log p_v),
    m = q w / E_q[w] (m = q without correction), with m detached. Its logit
    gradient is A w (e_y - m), whose expectation over y ~ q vanishes for a
    constant advantage: unlike plain REINFORCE (expected logit update
    A (q - p) when q != p), it adds no drift toward or away from the sampler.
    On-policy (q = p, w = 1) the extra term has exactly zero gradient, since
    sum_v p_v grad log p_v = 0. See REPORT.md.

    q over the vocabulary comes from the sampler's top-k logprobs
    (sampler_topk_ids / sampler_topk_logprobs), the leftover mass spread like
    p. Without them the loss falls back to plain REINFORCE.

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
    centered = "sampler_topk_ids" in batch and "sampler_topk_logprobs" in batch
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        pos = jax.lax.dynamic_slice_in_dim(jnp.broadcast_to(positive, mask.shape), start, chunk, axis=1)
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, pos, token_c, tv)
        p_vocab = sg(jnp.exp(logp_vocab))
        plogp_all = (p_vocab * logp_vocab).sum(-1)  # differentiable; sg(p) * log p
        score = logp
        drift = jnp.zeros_like(logp)
        if centered:
            # Sampler-centered score: log p(y) - sum_v m_v log p_v, m = q w / E_q[w].
            ids, log_m_k, log_tail = sampler_center_logmass(logp_vocab, logp, b, pos, token_c)
            logp_k = take_vocab(logp_vocab, ids)
            m_k = jnp.exp(log_m_k)
            tail = jnp.exp(log_tail)                                    # m_v = tail * p_v off K
            p_k = sg(jnp.exp(logp_k))
            z = jnp.maximum(m_k.sum(-1) + tail * jnp.clip(1.0 - p_k.sum(-1), 0.0, 1.0), 1e-30)
            plogp_tail = plogp_all - (p_k * logp_k).sum(-1)              # sum_{v not in K} p_v log p_v
            center = ((m_k * logp_k).sum(-1) + tail * plogp_tail) / sg(z)
            score = logp - center
            # E_y~q of the plain-REINFORCE logit update per unit advantage is q - p:
            # report its TV size |q - p|_1 / 2 restricted to K (+ tail), a drift diagnostic.
            drift = sg(0.5 * (jnp.abs(jnp.exp(b["sampler_topk_logprobs"]) - p_k).sum(-1)))
        terms = {"score": w * score, "entropy": -sg(plogp_all), "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv, "drift": drift}
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
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "drift") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    metrics["tv_samp_train_topk"] = metrics.pop("drift")
    return loss, metrics | {"frac_masked": frac_masked}
