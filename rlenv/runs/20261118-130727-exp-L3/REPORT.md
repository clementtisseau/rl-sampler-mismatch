# Drift-free REINFORCE under sampler mismatch

## What is wrong with the current loss

`loss_fn_rl` with `correction=None` minimizes `-(1/N) sum_t A_t w_t log p(y_t | h_t)`
with `w_t = 1`, where `y_t` is the sampled token, `h_t` its prefix, `A_t` the advantage
of the completion, and `s_v(h) = grad_theta log p(v|h)` the score of token `v`.
The gradient is `g = sum_t A_t s_{y_t}(h_t)`.

Fix a prefix `h`. The token `y` is drawn from the sampler `q(.|h)`, not from the
trainer `p(.|h)`. Then

    E_q[A s_y | h] = E_q[A | h] * s_bar_q(h) + Cov_q(A, s_y | h),
    s_bar_q(h) = sum_v q(v|h) s_v(h).

(This is the decomposition in the task: `E_q[s_y] = s_bar_q` and `E_q[A s_y] - E_q[A] E_q[s_y]`
is the covariance.)

On-policy, `s_bar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0` identically,
so only the covariance remains, and the covariance is what policy gradient needs.
Under a mismatch, `s_bar_q(h) != 0` and the first term is a **drift**: it is proportional
to the expected advantage `E_q[A|h]` at that prefix, which is determined by the rewards only
through the group-centering, not by which token is better. It does not vanish in
expectation and it is not zero-mean noise.

The drift has a concrete form. For a softmax head parameterized by logits `z(h)`,
`s_v = e_v - p` and therefore `s_bar_q = q - p` (as a vector in logit space). The expected
update at prefix `h` is therefore `E_q[A|h] (q(.|h) - p(.|h))`: the policy's logits are pushed
toward the sampler's distribution when the expected advantage is positive, and away from it when
it is negative. Each step, the sampler is refreshed from the trainer, but the mismatch
(fixed weight noise, quantization, tail mass of `q`) is systematically in the same direction, so the
bias accumulates instead of averaging out. Tokens that the sampler over-weights (for example
tail tokens that `p` almost never emits, or quantization-induced mass) are reinforced for no reason
related to the reward.

Importance weights `w = p/q` remove this exactly when they are unclipped:
`E_q[(p/q) A s_y] = sum_v p(v) A_v s_v`, which is the on-policy gradient. Clipping (TIS, token-level
clamping) or dropping (MIS, DPPO band) makes `w` a biased function of the token, and the
drift returns on the tokens where `w` is clipped. This is why clipping only slows the degradation.

## The fix: subtract the sampler's expected score

Replace the score of the sampled token by its difference from the sampler's expected score:

    score_t = log p(y_t | h_t) - sum_v q(v | h_t) log p(v | h_t)      (uncorrected loss)

so the loss is `-(1/N) sum_t A_t * score_t`, with gradient `-A_t (s_{y_t} - s_bar_q(h_t))`.
The second term `sum_v q(v|h) log p(v|h)` is the cross-entropy of the sampler's distribution under
the trainer; its gradient is exactly `s_bar_q(h)`. It is stop-gradient on `q`, only `p` carries gradient.

**Expected update.** Under `q`,

    E_q[A (s_y - s_bar_q) | h] = E_q[A s_y | h] - E_q[A|h] s_bar_q(h) = Cov_q(A, s_y | h).

The drift term is removed exactly, for any `q`, at every prefix, and the expected update is the
covariance alone, as in the on-policy case.

**On-policy.** With `q = p`, `s_bar_p = 0` identically, so the subtracted term is zero and the
estimator equals the original one. Nothing changes when there is no mismatch.

**Why this is a baseline.** `s_bar_q(h)` depends on the prefix only, not on the sampled token,
so `A * s_bar_q(h)` is a control variate whose mean under `q` is exactly the drift `E_q[A|h] s_bar_q(h)`.
Subtracting it cancels the drift in expectation.

**With an importance correction.** If the token weight `w = p/q` is used (token-level IS), the estimator
`E_q[w A s_y] = E_p[A s_y]` is already unbiased for the trainer, and the matching baseline is
`s_bar_p = 0`. Subtracting `s_bar_q` would then reintroduce `E_p[A] s_bar_q`. The implementation
therefore applies the expected-score term only when no token-level correction is active (no correction,
or a sequence-level one). Sequence-level weights act on whole trajectories and do not change the per-token
argument.

**Remaining error.** The estimator's expectation is `Cov_q(A, s_y | h)` rather than `Cov_p(A, s_y | h)`.
The difference is a second-order effect of the mismatch in the covariance itself, not a bias
proportional to `E[A|h]` at every prefix. It does not accumulate in the same direction across steps.

## Implementation

In `losses.py`:

- `expected_logp_under_sampler(logp_vocab, batch)` computes `sum_v q(v|h) log p(v|h)` per position.
  With the full sampler distribution (`sampling_vocab_logprobs`, [B, T, V]) it is exact. With top-k
  (`sampler_topk_ids`, `sampler_topk_logprobs`) it renormalizes `q` over the kept tokens, which is an
  approximation with the tail mass dropped. If no sampler distribution is present it returns 0, so the
  score is the plain sampled-token logprob.
- In `chunk_sums`, the uncorrected score is `logp - expected_logp_under_sampler(...)`. With a token-level
  correction it stays `w * logp`.
- `sampler_keys` now forwards the three sampler-distribution keys to the chunked loss.

The loss uses the same chunking and checkpointing as before; the extra cost is one `[B, chunk, V]`
product per chunk.

## Configuration

sampler.vocab_logprobs=full
