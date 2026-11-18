"""REINFORCE with importance-sampling corrections and a sampler-score baseline.

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


def sampler_mass(logp_vocab, logp, targets, b, positive, c):
    """Detached per-token mass m(v) = q(v|h) w_v of the expected score
    E_q[w_y grad log p(y|h) | h] = sum_v m(v) grad log p(v|h), and the
    surrogate sum_v m(v) log p(v|h) whose gradient is that expected score.

    q is the sampler's next-token distribution. Exact with the full sampler
    vocabulary (`sampling_vocab_logprobs`). With only its top-k head
    (`sampler_topk_ids` / `sampler_topk_logprobs`), or with nothing but the
    sampled token, q is known exactly on K = head + {sampled token}, and its
    remaining mass q(T) = 1 - q(K) is spread over the tail T in proportion to
    p, the trainer's distribution: q_hat(v) = q(T) p(v) / p(T) for v in T.
    Returns (surrogate [b, C], mass outside the band [b, C] for metrics)."""
    pos = positive[..., None]
    if "sampling_vocab_logprobs" in b:
        logq_vocab = b["sampling_vocab_logprobs"].astype(jnp.float32)
        p_vocab = sg(jnp.exp(logp_vocab))
        tv = sg(jnp.abs(jnp.exp(logq_vocab) - p_vocab))
        mass = sg(jnp.exp(log_mass(sg(logp_vocab), logq_vocab, pos, c, tv)))
        return jnp.where(mass > 0, mass * logp_vocab, 0.0).sum(-1)

    # Known support K: the sampler's top-k head (if recorded) plus the sampled token.
    logq_y = b["sampling_token_logprobs"]
    y = targets[..., None]
    if "sampler_topk_ids" in b:
        ids = b["sampler_topk_ids"].astype(y.dtype)
        logq_k = b["sampler_topk_logprobs"].astype(jnp.float32)
        y_new = ~(ids == y).any(-1, keepdims=True)  # sampled token outside the head
        ids = jnp.concatenate([ids, y], -1)
        logq_k = jnp.concatenate([logq_k, logq_y[..., None]], -1)
        valid = jnp.concatenate([jnp.ones(ids.shape[:-1] + (ids.shape[-1] - 1,), bool), y_new], -1)
        logp_k = take_vocab(logp_vocab, ids)
    else:
        logq_k, valid, logp_k = logq_y[..., None], jnp.ones_like(y, bool), logp[..., None]
    q_k = jnp.where(valid, jnp.exp(logq_k), 0.0)
    p_k = sg(jnp.where(valid, jnp.exp(logp_k), 0.0))
    tv_k = jnp.abs(q_k - p_k)
    mass_k = jnp.where(valid, sg(jnp.exp(log_mass(sg(logp_k), logq_k, pos, c, tv_k))), 0.0)
    head_term = jnp.where(mass_k > 0, mass_k * logp_k, 0.0).sum(-1)
    # Tail T: q_hat / p is constant on T, so its band test uses q(T), p(T).
    q_t = jnp.clip(1.0 - q_k.sum(-1), 0.0, 1.0)
    p_t = jnp.clip(1.0 - p_k.sum(-1), 0.0, 1.0)
    log_q_t, log_p_t = jnp.log(jnp.maximum(q_t, 1e-30)), jnp.log(jnp.maximum(p_t, 1e-30))
    mass_t = jnp.where(q_t > 0, jnp.exp(log_mass(log_p_t, log_q_t, positive, c, jnp.abs(q_t - p_t))), 0.0)
    coef_t = sg(mass_t / jnp.maximum(p_t, 1e-4))  # floor: float32 1 - p(K) is noise below ~1e-6
    p_vocab = sg(jnp.exp(logp_vocab))
    tail_term = (p_vocab * logp_vocab).sum(-1) - jnp.where(valid, p_k * logp_k, 0.0).sum(-1)
    return head_term + coef_t * tail_term


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection, and
    the sampler's expected score as a per-token baseline. Returns (loss, metrics).

    Per completion token y at prefix h the surrogate is
        w_y log p(y|h) - sum_v sg(q(v|h) w_v) log p(v|h),
    so the update is A (w_y s_y - E_q[w s | h]), with s_v = grad log p(v|h).
    Its expectation over y ~ q(.|h) is the covariance of A and w_y s_y under
    q: no update unless the advantage depends on the token. Plain REINFORCE
    adds E_q[A|h] E_q[w s|h], which is 0 on-policy but under a mismatched
    sampler is a drift with no reward signal (REPORT.md). For q = p and no
    correction the extra term has zero gradient: plain REINFORCE.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally its next-token distribution, sampling_vocab_logprobs
    [B, T, V] (sampler.vocab_logprobs=full) or its top-k head sampler_topk_ids /
    sampler_topk_logprobs [B, T, k] (sampler.vocab_logprobs=k).
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = jnp.broadcast_to(batch["advantages"][:, None] >= 0, targets.shape)
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask, "positive": positive} | {
        k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, b["positive"], token_c, tv)
        baseline = sampler_mass(logp_vocab, logp, b["targets"], b, b["positive"], token_c)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp - baseline, "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv}
        return {k: jnp.where(b["mask"], v, 0).sum(-1) for k, v in terms.items()}

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
