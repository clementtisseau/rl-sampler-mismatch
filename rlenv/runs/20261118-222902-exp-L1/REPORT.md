# Sampler mismatch in REINFORCE: diagnosis and fix

## What is wrong with the current loss
Rollouts y ~ q (sampler), but the loss is -(1/N) sum_i A_i sum_t log p(y_t | y_<t), differentiated at p (trainer).
Its expected gradient is, per token, E_{y_t~q}[ A ∇log p(y_t) ] and not E_{y_t~p}[...].
Write A = A_t^{signal} + b with b the part of the advantage not depending on y_t (group centering, length, prefix).
Since sum_v p_v ∇log p_v = 0, a baseline is harmless under p. Under q it is not:

  E_q[∇log p] = sum_v q_v ∇log p_v = -∇_p CE(q, p) = -∇_p KL(q||p) + const  ≠ 0.

So every token carries a term b_t * ∇(-KL(q||p)) with sign set by the advantage. Group-centering only cancels b across
sequences, not across tokens: sequences have different lengths and failing ones are typically longer and carry
mass on tails, so sum_t A is not zero. This gives a systematic drift of p toward (A>0) or away from (A<0) the
sampler q, i.e. toward the sampler's errors (noise, quantization tails). It is a bias, not zero-mean noise.
Since q is rebuilt from p each step (q_k = noisy/quantized(p_k)), the drift acts every step and compounds.
It is also why hard token-level clipping can hurt: dropping tokens with large p/q removes exactly those whose
weight corrects the bias (the tokens where q and p disagree most), while keeping the rest uncorrected.

## Fix
Use the unclipped token-level importance weight r_t = p_t/q_t (detached) on the score, so that
  E_q[ r_t ∇log p_t ] = sum_v p_v ∇log p_v = 0   for the baseline part,
  E_q[ r_t A_t ∇log p_t ] = E_p[ A_t ∇log p_t ]  for the signal part,
i.e. the per-token conditional expectation is the on-policy one and the KL(q||p) pull cancels exactly.
No clipping or dropping (which re-introduces the bias). Variance is controlled instead by self-normalisation:
the loss is divided by sum_t r_t (= N * mean r) rather than N. E_q[r]=1 per token, so this is consistent
(a ratio estimator, O(1/N) bias) and removes the batch-scale fluctuation of the weights.
Log-ratios are clipped at ±20 only for numerical safety.
Residual: prefix mismatch (the distribution of y_<t is q's, not p's) is a second-order effect; exact correction would need
sequence-level ratios whose variance grows exponentially with length.

## Implementation
`loss_fn_rl` with `correction=None` (default) runs token-level IS with the above self-normalised weights.
An explicit correction behaves as before.

## Configuration
rl.is.level=null
