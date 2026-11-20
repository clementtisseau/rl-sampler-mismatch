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


def sampler_expected_logp(logp_vocab, logp_target, batch_chunk, positive=None, c=None):
    """sum_v q~_v log p_v with q~ (detached) the sampler's next-token distribution,
    rebuilt from what the sampler returns.

    With the sampler's top-k logprobs: q~ is q on its top-k tokens, and the
    remaining mass m = 1 - sum_k q_k is spread over the other tokens in
    proportion to p (exact when q = p). Its gradient w.r.t. the logits is
    q~ - p. The full-vocabulary sum over the tail is replaced by its value
    with a stop-gradient on sum_v p_v log p_v, whose gradient is identically 0.

    Without top-k logprobs only q(y) of the sampled token y is known: q~ is then
    q(y) on y and 1 - q(y) spread over the others in proportion to p.
    With a token-level correction c, each token v is also weighted by its
    detached IS weight w(p_v / q~_v), so the result is sum_v q~_v w_v log p_v.
    Returns (expected logp [B, C], logprob mass of q~ that is exact, i.e. not
    spread over the tail [B, C])."""
    p_vocab = sg(jnp.exp(logp_vocab))
    s_all = sg((p_vocab * logp_vocab).sum(-1))  # value of sum_v p log p; gradient is 0 anyway
    if "sampler_topk_ids" in batch_chunk:
        ids = batch_chunk["sampler_topk_ids"].astype(jnp.int32)
        q_k = jnp.exp(batch_chunk["sampler_topk_logprobs"].astype(jnp.float32))
        logp_k = take_vocab(logp_vocab, ids)
        p_k = sg(jnp.exp(logp_k))
    else:  # only the sampled token
        q_k = jnp.exp(batch_chunk["sampling_token_logprobs"].astype(jnp.float32))[..., None]
        logp_k = logp_target[..., None]
        p_k = sg(jnp.exp(logp_k))
    q_k = sg(q_k)
    q_top = q_k.sum(-1)
    m = jnp.clip(1.0 - q_top, 0.0, 1.0)                 # sampler mass outside the known tokens
    p_tail = jnp.clip(1.0 - p_k.sum(-1), 1e-6, 1.0)     # trainer mass there
    c_tail = m / p_tail
    s_top = (p_k * logp_k).sum(-1)
    w_k = w_tail = 1.0
    if c is not None:
        log_q_k = jnp.log(jnp.maximum(q_k, 1e-30))
        w_k = is_weight(sg(logp_k) - log_q_k, positive[..., None], c, jnp.abs(q_k - p_k))
        w_tail = is_weight(jnp.log(p_tail) - jnp.log(jnp.maximum(m, 1e-30)), positive, c, jnp.abs(m - p_tail))
    expected = (w_k * q_k * logp_k).sum(-1) + w_tail * c_tail * (s_all - s_top)
    return expected, q_top


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts with a sampler-centered score, and an
    optional ISCorrection. Returns (loss, metrics).

    Each token's score log p(y_t) is centered under the sampler distribution q
    of that position: log p(y_t) - sum_v q_v log p_v (q detached). On-policy
    (q = p) the subtracted term has zero gradient (sum_v p_v grad log p_v = 0),
    so the update is plain REINFORCE. Off-policy it removes the term
    E[A | prefix] * E_q[grad log p] = -E[A | prefix] * grad KL(q || p), a
    systematic drift that pushes p away from q wherever the expected advantage
    is negative (see REPORT.md). In logit space the per-token update becomes
    A (e_y - q) instead of A (e_y - p): zero-mean under the distribution that
    actually sampled y.

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens;
    optionally sampler_topk_ids / sampler_topk_logprobs [B, T, K], the sampler's
    top-k next-token logprobs (same alignment).
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
        w = is_weight(logp - logq, positive, token_c, tv)
        # centered score: w(y) grad log p(y) minus its expectation under q
        expected, q_known = sampler_expected_logp(logp_vocab, logp, b, positive, token_c)
        score = w * logp - expected
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": score, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv,
                 "q_known": q_known}
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
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "q_known") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
