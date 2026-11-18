# Removing the mismatch drift from the REINFORCE gradient

## Problem
Fix a prefix h; y ~ q(.|h), A the advantage, s_v = grad log p(v|h). The loss -A log p(y|h) has expected gradient

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q=p) s_bar = grad sum_v p = 0, leaving only the covariance (the true policy gradient signal).
When q != p, s_bar = sum_v (q-p)(v) s_v != 0 (since sum_v p s_v = 0). The first term is a drift
proportional to E_q[A|h]: it pushes up/down the log-probs of tokens q favors, whatever the reward says. It is a
systematic bias (not zero-mean), and because the sampler is rebuilt from the trainer every step it compounds:
the drift changes p, which changes the mismatch, etc. With group-centered rewards the advantages average zero over
a group but E_q[A|h] is not zero at individual prefixes (prefixes that lead to success have positive mean A), so the drift is nonzero.
Token-level IS weights p/q or clipping rescale/zero tokens but do not remove this term: w(y) s_y with
y~q still has a mean-score component, and clipping makes the reweighted score even less balanced.

## Fix
Subtract the drift per position, using the sampler's next-token distribution (detached):

L = - sum_t A * [ log p(y_t|h_t) - sum_v q(v|h_t) log p(v|h_t) ].

The gradient of the bracket is s_y - s_bar(h), and
E_q[A (s_y - s_bar) | h] = E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h).
So the expected update equals the pure covariance term for any q, i.e. exactly the on-policy form
(when q=p it is unchanged in expectation, since s_bar=0). The drift is removed, not just damped; it also
reduces variance since s_bar is a control variate with known mean contribution. (An IS weight, if a correction is
given, multiplies both parts of the bracket.) The residual bias is that the covariance is under q rather than p,
which is the ordinary, non-compounding, reward-driven mismatch.

## Implementation
losses.py uses sampling_vocab_logprobs (exact, sum over full vocab) if present, else the top-k
(sampler_topk_ids/logprobs; the tail mass of q is ignored, a small error), else falls back to q:=p
(no correction of drift, since no q is available). Active by default, works with correction=None.

## Configuration
sampler.vocab_logprobs=full
