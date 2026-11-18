# Sampler-mismatch drift in REINFORCE, and a fix

## What is wrong
Fix a prefix h. y ~ q(.|h), score s_v = grad log p(v|h), advantage A. REINFORCE (also with IS weights that
are clipped/dropped) updates along A s_y, and
E_q[A s_y | h] = E_q[A|h] s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.
On-policy q=p: s_bar = grad sum_v p = 0, only the covariance (the reward signal) remains. With q != p,
s_bar = sum_v (q-p)(v) s_v != 0 (sum_v p s_v = 0), so there is a drift E_q[A|h] s_bar(h). It is a
systematic bias not zero-mean noise: it is nonzero whenever prefixes have non-zero mean advantage
(e.g. prefixes that lead to success), it pushes p along sum_v (q-p) s_v, i.e. toward q's quirks for
positive-advantage prefixes and away for negative ones, and since the sampler is re-derived from the
trainer every step the errors accumulate. Token-level IS clipping/dropping changes the weights per
token but a clipped w(p/q) no longer satisfies sum_v q w s_v = 0 (exact IS, w=p/q, gives sum_v p s_v = 0,
but only for unclipped weights and with huge variance), so it leaves a residual drift, and stronger clipping
leaves more.

## Fix: subtract the sampler-averaged score (control variate)
Per token use the surrogate  w * (log p(y|h) - b(h)),  b(h) = sum_v q(v|h) log p(v|h), with q detached.
Then grad b = sum_v q(v) s_v = s_bar(h), so the update direction is A (s_y - s_bar(h)) and
E_q[A (s_y - s_bar) | h] = E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h):
the drift term is removed exactly for any q, and on-policy (q=p) b has zero gradient, so the loss reduces
to plain REINFORCE (no cost when there is no mismatch). The covariance term, which carries the reward
signal, is unchanged (s_bar is constant given h, so it also does not add variance beyond the removed drift).
The correction is active by default (correction=None) and also composes with an IS weight w.

Implementation (losses.py, `sampler_baseline`): with full sampler logprobs, b is exact. With top-k logprobs,
b is exact on the top-k and the leftover q-mass 1-sum_topk q is assigned to the tail with p's conditional
distribution (an approximation; tail score mean under p is exact-zero-in-expectation). If no sampler
distribution is recorded, b=0 (falls back to plain REINFORCE; cannot be corrected, since only the sampled
token's q is known).

## Configuration
sampler.vocab_logprobs=full
