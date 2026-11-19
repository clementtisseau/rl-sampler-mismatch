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


def sampler_centering(logp_vocab, logp, logq, b, positive, token_c):
    """Per-token centering term m_t = sum_v sg(c_t(v)) log p_t(v) such that the
    corrected score  w(y_t) log p_t(y_t) - m_t  has zero mean under the
    sampler q_t, whatever q_t is (REPORT.md). c_t(v) = q_t(v) w(v) is the
    sampler's (IS-weighted) next-token distribution, reconstructed from

    - the sampler's top-k logprobs (exact on the head), and
    - its remaining mass spread over the rest of the vocabulary in proportion
      to p (the tail of q is not observed).

    Without top-k logprobs only the sampled token's q is known: the head is
    then that single token. With q = p everywhere, c = p and the gradient of
    m_t is exactly zero, so the on-policy update is unchanged."""
    if "sampler_topk_ids" in b:
        ids, log_q_head = b["sampler_topk_ids"], b["sampler_topk_logprobs"]
        # the sampled token's logprob is exact even when it lies outside the head
        in_head = (ids == b["targets"][..., None]).any(-1, keepdims=True)
        ids = jnp.concatenate([ids, b["targets"][..., None]], -1)
        log_q_head = jnp.concatenate([log_q_head, jnp.where(in_head, -jnp.inf, logq[..., None])], -1)
    else:
        ids, log_q_head = b["targets"][..., None], logq[..., None]
    log_p_head = take_vocab(logp_vocab, ids)
    head_valid = log_q_head > -jnp.inf
    p_head = jnp.where(head_valid, jnp.exp(log_p_head), 0.0)
    q_head = jnp.where(head_valid, jnp.exp(log_q_head), 0.0)
    # sum_{v in tail} sg(p(v)) log p(v), as all minus head (p is a detached weight)
    plogp_all = (sg(jnp.exp(logp_vocab)) * logp_vocab).sum(-1)
    plogp_head = (sg(p_head) * jnp.where(head_valid, log_p_head, 0.0)).sum(-1)
    p_rest = sg(jnp.clip(1.0 - p_head.sum(-1), 1e-12, 1.0))
    q_rest = sg(jnp.clip(1.0 - q_head.sum(-1), 0.0, 1.0))
    # tail: q(v) ~= s p(v), s = q_rest / p_rest, ratio p/q = 1/s
    log_s = jnp.log(jnp.maximum(q_rest, 1e-30)) - jnp.log(p_rest)
    pos_k = jnp.broadcast_to(positive[..., None], log_p_head.shape)
    tv_head = jnp.abs(q_head - p_head)
    log_c_head = log_mass(sg(log_p_head), jnp.where(head_valid, log_q_head, 0.0), pos_k, token_c, sg(tv_head))
    c_head = jnp.where(head_valid, sg(jnp.exp(log_c_head)), 0.0)
    tail_ok = p_rest > 1e-6  # below that the tail is negligible (and q_rest/p_rest ill-conditioned)
    log_s = jnp.where(tail_ok, log_s, -jnp.inf)
    if token_c is None:
        c_tail = jnp.exp(log_s)             # per unit of p in the tail
    else:
        log_r_tail = jnp.clip(jnp.where(tail_ok, -log_s, 0.0), -20.0, 20.0)
        tv_tail = sg(jnp.abs(q_rest - p_rest))
        w_tail = is_weight(log_r_tail, jnp.broadcast_to(positive, log_s.shape), token_c, tv_tail)
        c_tail = jnp.where(tail_ok & (q_rest > 0), sg(jnp.exp(log_s)) * w_tail, 0.0)
    m = (c_head * jnp.where(head_valid, log_p_head, 0.0)).sum(-1) + c_tail * (plogp_all - plogp_head)
    return m


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    The score of every token is centered under the sampler's next-token
    distribution q (sampler_centering): w log p(y) - sum_v sg(q w)(v) log p(v).
    Its expectation under q is exactly zero, so tokens that do not affect the
    reward receive no systematic push, as in the on-policy case. On-policy
    (q = p) the extra term has zero gradient. See REPORT.md.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens;
    optionally sampler_topk_ids / sampler_topk_logprobs [B, T, k], the sampler's
    top-k next-token logprobs (used to reconstruct q for the centering).
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
    pos_bt = jnp.broadcast_to(positive, targets.shape)

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        pos = jax.lax.dynamic_slice_in_dim(pos_bt, start, chunk, axis=1)
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, pos, token_c, tv)
        center = sampler_centering(logp_vocab, logp, logq, b, pos, token_c)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp - center, "entropy": -plogp, "kl_samp_train": sg(logq - logp),
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
