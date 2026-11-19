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


def centering_logp(logp_vocab, logp_k, logq_k, positive, c):
    """E_{v~mu}[log p(v)] with mu detached: the score's baseline.

    mu is the normalised effective sampling distribution of one position,
    mu(v) ∝ q(v) w(p(v)/q(v)), i.e. the distribution under which the weighted
    score w (e_y - mu) of the sampled token y has exactly zero mean. q is
    known on the sampler's top-k ids (logq_k, [B, C, K]); the remaining tail
    mass Q_tail is spread proportionally to p, and is weighted by the ratio
    P_tail / Q_tail of the two tails. Returns ([B, C] differentiable in log p,
    [B, C] detached sampler tail mass).
    """
    logq_k = jnp.where(jnp.isfinite(logq_k), logq_k, -1e30)
    sp_k, sq_k = sg(jnp.exp(logp_k)), jnp.exp(logq_k)
    p_tail = jnp.clip(1.0 - sp_k.sum(-1), 0.0, 1.0)
    q_tail = jnp.clip(1.0 - sq_k.sum(-1), 0.0, 1.0)
    pos = positive[..., None] if positive.ndim == logp_k.ndim - 1 else positive
    tv_k = jnp.abs(sq_k - sp_k)
    m_k = jnp.exp(log_mass(sg(logp_k), logq_k, pos, c, tv_k))  # q w on the top-k
    lp_tail, lq_tail = jnp.log(jnp.maximum(p_tail, 1e-30)), jnp.log(jnp.maximum(q_tail, 1e-30))
    m_tail = jnp.where(q_tail > 0, jnp.exp(log_mass(lp_tail, lq_tail, positive, c, jnp.abs(q_tail - p_tail))), 0.0)
    z = m_k.sum(-1) + m_tail
    safe_z = jnp.where(z > 0, z, 1.0)
    mu_k = jnp.where(z[..., None] > 0, m_k / safe_z[..., None], 0.0)
    alpha = jnp.where((z > 0) & (p_tail > 0), m_tail / safe_z / jnp.maximum(p_tail, 1e-30), 0.0)  # mu = alpha p on the tail
    p_vocab = sg(jnp.exp(logp_vocab))
    # sum_v mu_v log p_v = alpha * sum_v p_v log p_v + sum_k (mu_k - alpha p_k) log p_k
    full = (p_vocab * logp_vocab).sum(-1)
    center = sg(alpha) * full + (sg(mu_k - alpha[..., None] * sp_k) * logp_k).sum(-1)
    return center, q_tail


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection and a
    sampler-centered score. Returns (loss, metrics).

    Per token the score is w * (log p(y) - E_{v~mu} log p(v)), mu ∝ q w(p/q)
    over the vocabulary (centering_logp; see REPORT.md). Its gradient on the
    logits is w (e_y - mu), which has zero mean under the sampler q, so the
    mismatch no longer adds a value-proportional drift V(s) (q - p) to the
    update. When q = p and w = 1 it is plain REINFORCE.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B],
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled
    tokens, and optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K],
    its top-k logprobs per position (without them only the sampled token's q
    is used and the rest of q is assumed proportional to p).
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
    sampler_keys = ("sampling_token_logprobs",) + (("sampler_topk_ids", "sampler_topk_logprobs") if has_topk else ())
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if has_topk:
            ids, logq_k = b["sampler_topk_ids"], b["sampler_topk_logprobs"].astype(jnp.float32)
        else:
            ids, logq_k = b["targets"][..., None], logq[..., None]
        ids = jnp.clip(ids, 0, logp_vocab.shape[-1] - 1)
        logp_k = take_vocab(logp_vocab, ids)
        center, q_tail = centering_logp(logp_vocab, logp_k, logq_k, positive, token_c)
        m = b["mask"]
        score = jnp.where(m, w * (logp - center), 0.0)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": score, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv,
                 "q_tail": sg(q_tail)}
        return {k: jnp.where(m, v, 0).sum(-1) for k, v in terms.items()}

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
    metrics["sampler_tail_mass"] = metrics.pop("q_tail")
    return loss, metrics | {"frac_masked": frac_masked}
