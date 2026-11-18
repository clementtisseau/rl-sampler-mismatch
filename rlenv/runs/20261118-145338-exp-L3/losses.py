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


def _centering(logp_vocab, logp, logq, b, positive, token_c):
    """Detached masses (m_v / Z) of the sampler's next-token distribution, and
    the trainer logprobs they multiply, for the control variate

        s_bar(h) = sum_v m_v(h) grad log p(v|h) / Z(h),   m_v = q(v|h) w(p_v/q_v)

    (m = q without correction). Returns (coef, logp_terms), both [B, C, n]:
    the loss subtracts sum_n coef * logp_terms from log p(y|h).

    What the sampler recorded decides the support:
      full vocabulary   exact s_bar;
      top-k head K      exact on K, the tail mass Q_T = 1 - sum_K q is spread
                        like p on the tail: its score is grad log P_T,
                        P_T = 1 - sum_K p (exact if q|tail is proportional to p|tail);
      sampled token     the same with K = {y} (only q(y) is known).
    """
    if "sampling_vocab_logprobs" in b:
        logq_v = b["sampling_vocab_logprobs"].astype(jnp.float32)
        tv = jnp.abs(jnp.exp(logq_v) - jnp.exp(sg(logp_vocab)))
        log_m = log_mass(sg(logp_vocab), logq_v, positive[..., None], token_c, tv)
        coef = sg(jax.nn.softmax(log_m, axis=-1))
        return coef, logp_vocab
    if "sampler_topk_ids" in b:
        logp_k = take_vocab(logp_vocab, b["sampler_topk_ids"])
        logq_k = b["sampler_topk_logprobs"].astype(jnp.float32)
    else:
        logp_k, logq_k = logp[..., None], logq[..., None]
    tv_k = jnp.abs(jnp.exp(logq_k) - jnp.exp(sg(logp_k)))
    log_m_k = log_mass(sg(logp_k), logq_k, positive[..., None], token_c, tv_k)
    q_tail = jnp.clip(1.0 - jnp.exp(logq_k).sum(-1), 0.0, 1.0)
    p_tail = 1.0 - jnp.exp(logp_k).sum(-1)
    ok = p_tail > 1e-4  # below this the tail score is numerically meaningless: drop it
    logp_tail = jnp.log(jnp.where(ok, p_tail, 1.0))
    log_m_tail = jnp.where(ok & (q_tail > 0), jnp.log(jnp.where(q_tail > 0, q_tail, 1.0)), -jnp.inf)
    log_m = jnp.concatenate([log_m_k, log_m_tail[..., None]], -1)
    coef = sg(jax.nn.softmax(log_m, axis=-1))
    coef = jnp.where(jnp.isfinite(log_m).any(-1, keepdims=True), coef, 0.0)
    return coef, jnp.concatenate([logp_k, logp_tail[..., None]], -1)


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """Drift-free REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    Plain REINFORCE uses the score grad log p(y|h). Under a sampler q != p
    its conditional mean s_bar(h) = E_q[grad log p(y|h)] is not zero, so
    E_q[A s_y | h] = E_q[A|h] s_bar(h) + Cov_q(A, s_y|h) carries a drift term
    unrelated to the reward signal. This loss uses the centered score
    s_y - s_bar(h) instead (see REPORT.md): per token
        A * (log p(y|h) - sum_v sg(q(v|h)) log p(v|h)),
    whose logit gradient is A (e_y - q): its expectation under q is exactly
    the covariance term, for any advantage. With a token-level correction w,
    q is replaced by the normalized corrected mass q w / sum q w.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens.
    Optional (sampler.vocab_logprobs): sampler_topk_ids / sampler_topk_logprobs
    [B, T, k] or sampling_vocab_logprobs [B, T, V], used for s_bar.
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
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"].astype(jnp.float32)
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        coef, logp_terms = _centering(logp_vocab, logp, logq, b, positive, token_c)
        baseline = (coef * logp_terms).sum(-1)  # value: E_m[log p]; gradient: s_bar
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * (logp - baseline), "entropy": -plogp, "kl_samp_train": sg(logq - logp),
                 "masked": w == 0, "tv": tv, "ctrl": sg(1.0 - coef.max(-1))}
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
