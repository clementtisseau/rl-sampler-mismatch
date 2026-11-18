# Removing the sampler-mismatch drift from the REINFORCE loss

## What is wrong with the current loss

`loss_fn_rl` with `correction=None` minimises `-sum_t A * logp(y_t)` with
`y_t ~ q` (the sampler) and `A` the group-centred advantage of the completion.
Its expected gradient at a token position, conditional on the prefix h, is

    E_q[A s_y | h] = E_q[A | h] s_bar_q(h) + Cov_q(A, s_y | h),
    s_y = grad_theta log p(y|h),   s_bar_q(h) = sum_v q(v|h) s_v.

The on-policy gradient we want is the covariance term alone, because

    s_bar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0,

so the first term vanishes when q = p. Under a sampler mismatch q != p, so
s_bar_q(h) != 0 and the loss contains the extra term

    drift(h) = E_q[A | h] * s_bar_q(h).

This term is not reward signal. Its direction is the score of the q-weighted
log-likelihood, grad sum_v q(v|h) log p(v|h), scaled by the expected advantage
at that prefix. Where the prefix tends to lead to a solved completion
(E_q[A|h] > 0) it pushes the trainer toward the sampler's own next-token
distribution, and where it tends to lead to failure it pushes away. The sign
of the push is set by reward, but its direction is set by q, not by the
reward. Because the sampler is refreshed from the trainer at every step, the
drift is systematic: it does not average out over steps, and it compounds.
Clipping or token-level IS weights only rescale the sampled-token term. They
do not remove s_bar_q(h), so they slow the degradation but do not remove it.

## The modification

Subtract the drift with a control variate. The baseline is the q-expected
log-probability of the next token,

    b(h) = sum_v q(v|h) log p(v|h),     grad_theta b(h) = s_bar_q(h)

(q is stop-gradiented, so the gradient is exactly s_bar_q(h); checked
numerically on a softmax toy). The loss uses the per-token score

    s~_t = logp(y_t|h_t) - b(h_t)   instead of   logp(y_t|h_t),

so the loss is `-sum_t A * s~_t` and the objective is unchanged in form.

Expected update. Let A be the full-sequence advantage, so its conditional
expectation depends on the prefix h. Then

    E_q[A (s_y - s_bar_q(h)) | h]
      = E_q[A s_y | h] - E_q[A | h] s_bar_q(h)
      = Cov_q(A, s_y | h).

The drift term cancels exactly, and the remaining update is the covariance
between advantage and score under the sampler's distribution. This is the
form the hint identifies as the reward signal.

On-policy check. When q = p, s_bar_p = 0, so b(h) has zero gradient and the
estimator is `E_p[A s_y | h] = Cov_p(A, s_y | h)`, the unchanged REINFORCE
gradient. The modification therefore changes nothing in the on-policy limit
(its only effect is on variance, and it is a control variate).

Remaining bias. Under mismatch the update is Cov_q(A, s_y | h), not
Cov_p(A, s_y | h). That difference is O(q - p) and is reward-directed: it is
the covariance of the reward with the score under the distribution that
generated the data. It has no component that pushes the trainer toward q
independently of the reward, so there is no drift to compound.

Verified numerically on a 6-way softmax with random A and a perturbed q:
`E_q[A(s_y - s_bar_q)] == Cov_q(A, s_y)` and `grad b == s_bar_q`.

## Implementation (losses.py)

- `sampling_vocab_logprobs` ([B, T, V]) gives q(.|h) over the full vocabulary.
  `b(h)` is computed exactly from it, in the same rematerialised chunks as
  the head, so only [B, chunk, V] is live.
- `sampler_topk_ids` / `sampler_topk_logprobs` (top-k) give the approximation
  `b(h) ~ sum_{v in top-k} q(v|h) log p(v|h)`, with the head of q not
  renormalised. This is approximate, since the omitted tail of q is dropped.
- With no sampler distribution recorded, b(h) = 0 and the loss reduces to the
  original estimator. The loss then cannot remove the drift, because s_bar_q
  depends on q over the whole vocabulary and a single sampled logprob does
  not determine it.
- The default-argument call (`correction=None`) uses the control variate.
  The drift removal requires the sampler's distribution, which is the
  `sampler.vocab_logprobs=full` configuration below.

Caveat: the exact cancellation is for the unweighted estimator. With a
token-level IS weight w, the baseline is multiplied by the same w, and the
cancellation is exact only when w = 1. The recommended setting is therefore
`correction=None`, which is the default.

## Empirical sanity check (not required; for the reader)

On `dev/mini.py` (CPU, sampler noise 0.3, group-centred rewards, 600 steps),
the eval reward over the last five evaluations was about 0.82 to 0.84 with
the control variate (3 seeds), versus about 0.67 to 0.69 for the original
loss. The original loss plateaued near 0.6 to 0.7 and fluctuated with the sampler
noise, while the control variate kept rising to about 0.85. The no-vocab fallback reproduced the
original run exactly. The top-k path runs.

## Configuration

sampler.vocab_logprobs=full
