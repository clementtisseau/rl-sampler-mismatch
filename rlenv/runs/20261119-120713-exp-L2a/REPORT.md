# Sampler-mismatch instability in REINFORCE: a q-centered score

## What is wrong with the current loss

Notation: at a position with prefix s, p = p_theta(. | s) is the trainer, q is the sampler,
y ~ q is the sampled token, A is the (group-centered) advantage of the sequence, and
z are the logits, so that grad_z log p(y) = e_y - p.

With correction=None, `loss_fn_rl` returns -sum_t A * log p(y_t) / N. Its expected gradient
per position is taken under the sampling distribution q, not p:

    E_q[A (e_y - p)]_v = q_v * Abar_q(v) - p_v * Abar_q,

where Abar_q(v) = E_q[A | s, y=v] and Abar_q = E_q[A | s]. Write it as

    G_v = q_v (Abar_q(v) - Abar_q)  +  (q_v - p_v) * Abar_q.

**On-policy (q = p).** The second term vanishes and G_v = p_v (Abar_p(v) - Abar_p), which is the
policy-gradient advantage form. The score has zero mean under p (E_p[grad log p] = 0), so any
advantage-independent offset contributes nothing in expectation.

**Mismatch (q != p).** The score no longer has zero mean under the distribution it is sampled
from: E_q[grad log p] = sum_v q_v (e_v - p) = q - p != 0. Consequently a constant (or
prefix-dependent) advantage Abar_q produces a systematic update c (q - p). Group centering
removes the mean over whole sequences, not the conditional mean Abar_q(s) given a prefix, so
the bias term (q - p) Abar_q is generically non-zero. Its direction is fixed: for c > 0,
ascent on E_q[c log p] = c * sum_v q_v log p_v moves p toward q (it minimizes KL(q || p) up to
terms in H(q)). Since the sampler is refreshed from the trainer at every step, and the
mismatch (noise, quantization, staleness) is a fixed offset rather than zero-mean noise, this
drift accumulates over training. The reward collapse follows: tokens that the sampler
over-weights (for example never-correct tail tokens, which the sampler puts mass on and the
trainer does not) receive a systematic push, and the trainer's distribution drifts into that
tail.

Two observations from the mini setup support this. With `MINI_EXTRA_TOKENS=200` the
sampler puts about 0.13-0.18 of its mass on never-correct tokens at step 0, and the current
loss drives the reward to 0 within about 25 steps. The bias term is linear in q - p, so it
does not vanish as the rollouts get better. Importance-sampling weights w = p/q make the
expected update exact, since E_q[w grad log p] = E_p[grad log p] = 0 for constant A, but
they are high-variance, and clipping them reintroduces a bias proportional to the clipped
mass. That is consistent with IS and clipping slowing the drift without removing it.

## The modification: centering the score under q

Replace the score log p(y) by the score minus its q-expectation, a baseline that depends only on
the prefix:

    score(s, y) = log p(y | s) - b(s),     b(s) = sum_v q(v | s) log p(v | s).

b is computed from the sampler's top-k logprobs at each position (q renormalized over those k
tokens and detached). Because b depends only on s, and E_q[grad log p] = grad b (with q detached),
the surrogate's gradient in logit space is

    grad_z score = (e_y - p) - (q - p) = e_y - q.

Its expectation under the sampler is

    E_q[A (e_y - q)]_v = q_v (Abar_q(v) - Abar_q).

So the bias term (q - p) Abar_q is gone. For a constant advantage the expected update is exactly
zero, whatever the mismatch. What remains is the on-policy advantage form with the sampler's
distribution q in place of p, a per-token reweighting of order q/p and a difference between
Abar_q and Abar_p. These differences do not have a fixed direction, so they do not accumulate in
the way the constant offset did. The baseline is a valid control variate (it is independent of the
sampled token), so the estimator stays unbiased for the q-expected advantage update. It is
also a deterministic function of the sampler's logprobs that the trainer already receives.

Why it is not applied with an IS weight: with w = p/q the weighted score already has zero
mean under q (E_q[w grad log p] = E_p[grad log p] = 0), so subtracting b(s) would add a term
E_q[w grad b] that is not zero in general. `losses.py` therefore centers only when
`correction is None`, which is the default.

Residual error: the sampler gives only top-k (k = 20) logprobs. The renormalized top-k
baseline differs from the full sum by O(tail mass of q). The residual bias is then proportional
to the tail mass rather than to the KL between q and p, and it does not compound, because the
tail mass beyond the top-k is not pushed in a fixed direction by this term.

## Implementation (losses.py)

- `baseline(logp_vocab, b)`: sum_k q_k log p(id_k) over the sampler's top-k ids, with
  q = softmax(topk_logprobs), detached. Uses `take_vocab`, so TP sharding is preserved.
- `chunk_sums` uses `score = logp - baseline` when `correction is None` and the batch has
  `sampler_topk_ids` and `sampler_topk_logprobs`. Otherwise it is the original loss.
- Entropy, KL, TV and masking metrics are unchanged.

## Empirical check (dev/mini.py, CPU)

These runs are short and are a sanity check, not evidence for the hidden settings.
Default `noise=0.3, lr=0.2, 600 steps`, `reward_mode=plus_minus`.

| setting | original (mean of last 10 reward) | centered (mean of last 10 reward) |
| --- | --- | --- |
| extra tokens 200, seed 0 | 0.000 (peak 0.33, collapse) | 0.848 |
| extra tokens 200, seed 1 | 0.089 (peak 0.32, collapse) | 0.804 |
| extra tokens 400, seed 0 | 0.000 (peak 0.34, collapse) | 0.629 |
| extra tokens 200, noise 0.6 | 0.000 (peak 0.12) | 0.000 (peak 0.04) |

The noise-0.6 run fails to learn under either loss, so it does not show stability. Token-level
exact IS (w = p/q, unclipped) gave 0.615 and 0.597 on the extra-200 seeds 0 and 1, so the
centered loss is at least as stable there. The default mismatch without extra tokens did not
collapse with the original loss (seed 0); the centered loss gave 0.821 on seed 1 (seed 0 was not
run with the centered loss).

No `## Configuration` section is needed: the change is on by default and uses no `rl.is.*` keys.
