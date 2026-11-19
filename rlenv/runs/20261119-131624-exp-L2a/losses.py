"""REINFORCE with importance-sampling corrections and a sampler-centered score.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.

The score of every token is centered under the sampler (always on, see
REPORT.md): the per-token term  w(y) log p(y)  becomes

    w(y) log p(y) - sum_y' sg(q(y') w(y')) log p(y'),

whose gradient has zero mean under y ~ q for any w, so a nonzero mean
advantage no longer turns into a drift of p along +/- grad KL(q || p).
On-policy (q = p, w = 1) the subtracted term has zero gradient and the loss
is plain REINFORCE. q over the vocabulary comes from the sampler's top-k
logprobs; the remaining mass is spread like p (see `sampler_mass`).
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


MAX_TAIL_SCALE = 100.0  # cap on c = q(tail) / p(tail) in sampler_mass


def sampler_mass(logp_vocab, logp_head, logq_head):
    """Estimate of the sampler's next-token distribution q_hat over the vocab.

    The sampler returns K logprobs (logq_head) of tokens whose trainer
    logprobs are logp_head; on those tokens q_hat = q. The missing mass
    1 - sum q_head is spread over the other tokens in proportion to p:
    q_hat(y) = c p(y) with c = (1 - Q_K) / (1 - P_K). Returns log c [B, C, 1]
    so that log q_hat = log c + log p off the head. With K = vocab this is q.
    """
    q_head = jnp.where(jnp.isfinite(logq_head), jnp.exp(logq_head), 0.0)
    q_rest = jnp.clip(1.0 - q_head.sum(-1, keepdims=True), 0.0, 1.0)
    p_rest = jnp.clip(1.0 - jnp.exp(logp_head).sum(-1, keepdims=True), 0.0, 1.0)
    # The tail sums are full-vocab sums minus head sums, exact only to ~1e-7:
    # cap c so that rounding error cannot be amplified. When the cap binds
    # (p puts ~no mass off the head but q does), q_hat sums to < 1 and the
    # missing mass costs a residual of at most q_rest (REPORT.md, sec. 4).
    log_c = jnp.log(jnp.maximum(q_rest, 1e-30)) - jnp.log(jnp.maximum(p_rest, 1e-30))
    return jnp.where(q_rest > 0, jnp.minimum(log_c, math.log(MAX_TAIL_SCALE)), -jnp.inf)


def centered_baseline(logp_vocab, logp_head, logq_head, positive, c):
    """B(theta) = sum_y sg(q_hat(y) w(y)) log p(y) per position [B, C].

    grad B = E_{y~q_hat}[w(y) grad log p(y)], the mean of the (weighted)
    score under the sampler; subtracting it makes the per-token score
    zero-mean under q. Computed as a full-vocab sum with q_hat = c p,
    corrected on the K head tokens (no scatter on the sharded vocab axis).
    """
    lp_v, lp_h = sg(logp_vocab), sg(logp_head)
    log_c = sampler_mass(lp_v, lp_h, logq_head)

    def mass(lp, lq):  # sg(q w(p/q)), zero where q = 0 or w = 0
        tv = jnp.abs(jnp.exp(lq) - jnp.exp(lp))
        m = jnp.exp(log_mass(lp, lq, positive[..., None], c, tv))
        return sg(jnp.where(jnp.isfinite(lq), m, 0.0))

    def dot(m, lp):  # sum m log p, with 0 * -inf = 0
        return jnp.where(m > 0, m * lp, 0.0).sum(-1)

    rest_all = dot(mass(lp_v, log_c + lp_v), logp_vocab)
    rest_head = dot(mass(lp_h, log_c + lp_h), logp_head)
    head_true = dot(mass(lp_h, logq_head), logp_head)
    return rest_all - rest_head + head_true


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

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
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        if "sampler_topk_ids" in b:  # the sampler's top-k next-token logprobs
            head_ids, logq_head = b["sampler_topk_ids"], b["sampler_topk_logprobs"].astype(jnp.float32)
        else:  # only the sampled token's q is known: K = 1
            head_ids, logq_head = b["targets"][..., None], logq[..., None]
        logp_head = take_vocab(logp_vocab, head_ids)
        baseline = centered_baseline(logp_vocab, logp_head, logq_head, positive, token_c)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        # centered score: zero-mean gradient under the sampler q (REPORT.md)
        score = w * logp - baseline
        terms = {"score": score, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
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
