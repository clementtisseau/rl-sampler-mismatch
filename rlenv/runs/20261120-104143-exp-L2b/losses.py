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


def sampler_centering(logp_vocab, logp, logq, targets, top_ids, top_logq, positive, c):
    """Coefficients a [B, C, K+1] on log p at the ids [B, C, K+1] such that

        grad_z sum_k a_k log p(ids_k) = pi - p,

    where pi is the sampler distribution q reweighted by the token-level
    correction, pi_v ∝ q_v w(p_v / q_v) (pi = q without a correction).
    Subtracting w(y) times this term from w(y) log p(y) turns the per-token
    logit update w(y) (e_y - p) into w(y) (e_y - pi), which has zero mean
    under y ~ q (see REPORT.md).

    q is known on the sampler's top-k ids and on the sampled token y (the
    set S). Its remaining mass 1 - Q_S is spread over the other tokens in
    proportion to p (q_v ≈ p_v (1 - Q_S) / (1 - P_S) off S), so the
    estimate is exact whenever the top-k covers the vocabulary and
    reduces to plain REINFORCE when q = p."""
    in_top = (top_ids == targets[..., None]).any(-1)
    ids = jnp.concatenate([top_ids, targets[..., None]], -1)
    lq = jnp.concatenate([top_logq, logq[..., None]], -1)
    valid = jnp.concatenate([jnp.ones(top_ids.shape, bool), ~in_top[..., None]], -1)
    lq = jnp.where(valid & jnp.isfinite(lq), jnp.minimum(lq, 0.0), -jnp.inf)
    lp = sg(take_vocab(logp_vocab, ids))
    q_s, p_s = jnp.exp(lq), jnp.where(valid, jnp.exp(lp), 0.0)
    q_tail = jnp.clip(1.0 - q_s.sum(-1), 0.0, 1.0)
    p_tail = jnp.clip(1.0 - p_s.sum(-1), 1e-6, 1.0)
    # sampler mass on S and on the tail, each times its correction weight
    log_ratio_s = jnp.clip(lp - lq, -20.0, 20.0)
    m_s = jnp.where(valid, q_s * is_weight(log_ratio_s, positive[..., None], c, jnp.abs(q_s - p_s)), 0.0)
    w_tail = is_weight(jnp.clip(jnp.log(p_tail) - jnp.log(jnp.maximum(q_tail, 1e-30)), -20.0, 20.0),
                       positive, c, jnp.zeros_like(p_tail))
    m_tail = q_tail * w_tail
    z = m_s.sum(-1) + m_tail
    ok = z > 1e-12
    z = jnp.where(ok, z, 1.0)
    pi_s, pi_tail = m_s / z[..., None], m_tail / z
    kappa = jnp.clip(pi_tail / p_tail, 0.0, 100.0)  # pi = kappa * p off S
    a = jnp.where(valid, pi_s - kappa[..., None] * p_s, 0.0)
    a = jnp.where(ok[..., None] & jnp.isfinite(a), a, 0.0)
    return sg(a), ids


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with the score centered under the
    sampler and an optional ISCorrection. Returns (loss, metrics).

    Per token, the plain score log p(y) has logit gradient e_y - p, whose mean
    under the sampler's y ~ q is q - p != 0: any part of the advantage that
    does not depend on y then drives p systematically toward q (A > 0) or
    away from it (A < 0). The loss uses instead

        w(y) [log p(y) - sum_v pi_v log p_v],   pi ∝ q w(p/q) (detached),

    whose logit gradient w(y) (e_y - pi) has zero mean under q: only the
    part of the advantage that depends on the sampled token moves p.
    With q = p (on-policy) the extra term has zero gradient.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens; optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K],
    the sampler's top-k logprobs per position.
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
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if has_topk:
            top_ids, top_logq = b["sampler_topk_ids"].astype(b["targets"].dtype), b["sampler_topk_logprobs"]
        else:  # only the sampled token's q is known
            top_ids = jnp.zeros(b["targets"].shape + (0,), b["targets"].dtype)
            top_logq = jnp.zeros(b["targets"].shape + (0,), jnp.float32)
        a, ids = sampler_centering(logp_vocab, sg(logp), sg(logq.astype(jnp.float32)), b["targets"], top_ids,
                                   top_logq.astype(jnp.float32), positive, token_c)
        center = (a * take_vocab(logp_vocab, ids)).sum(-1)  # grad_z = pi - p
        score = jnp.where(b["mask"], w * (logp - center), 0.0)
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
