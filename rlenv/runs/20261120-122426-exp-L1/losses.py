"""REINFORCE with importance-sampling corrections and a sampler-centered score.

The score of every token is centered under the distribution it is actually
sampled from: w(y) (grad log p(y) - E_{v~mbar}[grad log p(v)]) with
mbar ∝ q w(p/q), q the sampler (top-k logprobs from the rollout, tail spread
like p). Its expectation under the sampler is zero, so a mismatched sampler no
longer turns the reward baseline into a systematic push of p along q - p (see
REPORT.md). Without a correction mbar = q; with the full ratio w = p/q, mbar = p
and this is plain naive IS.

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


def sampler_baseline(logp_vocab, plogp_sg, topk_ids, topk_logq, positive, c):
    """Per-token score baseline b = sum_v mbar_v log p(v), with detached
    coefficients mbar ∝ q(v) w(p(v)/q(v)): the distribution the corrected score
    w(y) grad log p(y) is actually averaged under when y ~ q. Subtracting it
    makes E_{y~q}[w(y) (grad log p(y) - grad b)] = 0 exactly, for any w.

    q is known on the sampler's top-k; its remaining mass Q_tail is spread over
    the other tokens in proportion to p (the tail is treated as one outcome with
    ratio P_tail / Q_tail). Returns b [B, C], differentiable in log p."""
    logp_k = take_vocab(logp_vocab, topk_ids)                         # [B, C, K]
    logq_k = topk_logq.astype(jnp.float32)
    p_k, q_k = sg(jnp.exp(logp_k)), jnp.exp(logq_k)
    log_p_tail = sg(jnp.log(jnp.clip(1.0 - p_k.sum(-1), 1e-30, 1.0)))   # [B, C]
    log_q_tail = jnp.log(jnp.clip(1.0 - q_k.sum(-1), 1e-30, 1.0))
    pos = jnp.broadcast_to(positive, log_p_tail.shape)
    log_m_k = log_mass(jnp.clip(sg(logp_k), logq_k - 20.0, logq_k + 20.0), logq_k, pos[..., None], c,
                       jnp.abs(q_k - p_k))
    # The tail shares one ratio P_tail / Q_tail, so ratio bands treat it exactly as one outcome. For the
    # TV band (DPPO) each tail token's |q - p| = p_v |Q_tail / P_tail - 1| is below the smallest top-k
    # probability times that factor: taken as 0 (in band), as for the many tiny tokens of an LLM vocabulary.
    log_m_tail = log_mass(jnp.clip(log_p_tail, log_q_tail - 20.0, log_q_tail + 20.0), log_q_tail, pos, c,
                          jnp.zeros_like(log_p_tail))
    log_z = jnp.logaddexp(jax.nn.logsumexp(log_m_k, -1), log_m_tail)
    log_z = jnp.where(jnp.isfinite(log_z), log_z, 0.0)  # every weight zero: no baseline needed
    mbar_k = sg(jnp.exp(log_m_k - log_z[..., None]))
    # coefficient of sg(p_v) log p_v on the tail: mbar_tail / P_tail (P_tail floored for float32 safety)
    mbar_tail = sg(jnp.exp(log_m_tail - log_z - jnp.maximum(log_p_tail, math.log(1e-6))))
    tail_plogp = plogp_sg - (p_k * logp_k).sum(-1)                    # sum_{v not in top-k} sg(p_v) log p_v
    return (mbar_k * logp_k).sum(-1) + mbar_tail * tail_plogp


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens,
    and (optional) sampler_topk_ids / sampler_topk_logprobs [B, T, K], the
    sampler's top-k next-token logprobs, used to center the score under q.
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
        plogp_sg = (sg(jnp.exp(logp_vocab)) * logp_vocab).sum(-1)  # grad: sum_v p_v grad log p_v
        plogp = sg(plogp_sg)  # policy entropy metric
        if "sampler_topk_ids" in b and seq_c is None:
            baseline = sampler_baseline(logp_vocab, plogp_sg, b["sampler_topk_ids"],
                                        b["sampler_topk_logprobs"], positive, token_c)
        else:  # sequence-level weights, or no sampler distribution: center under p (the plain score)
            baseline = 0.0
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
