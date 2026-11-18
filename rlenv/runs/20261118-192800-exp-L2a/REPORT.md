# Why mismatch breaks REINFORCE, and the fix

## Problem
Let q be the sampler and p the trainer. At a context s the loss gradient w.r.t. the logits z is
A * (e_y - p(.|s)) for a sampled token y ~ q. Its expectation is
E_q[A (e_y - p)] = A (q - p) + Cov-term(A, y).
On-policy (q=p) the first term vanishes, which is why a baseline / group-centering is free:
E_p[grad log p] = sum_y p grad log p = grad sum_y p = 0.
Off-policy, E_q[grad log p(y)] = -grad_p KL(q||p) != 0 (it is the gradient of -CE(q,p)).
So any non-zero token-level mean advantage (e.g. sequence advantages are centered per group, but the loss sums
per token, so A_i is multiplied by the length L_i, sum_i A_i L_i != 0; and A correlates with which tokens are likely)
adds a term A (q-p): an unintended distillation of p toward the sampler's distribution q (positive A) or away from it
(negative A), i.e. a systematic bias, not noise. Because the sampler is rebuilt from p each step, the bias
is applied repeatedly and compounds (entropy/sharpness drifts, collapse). Token-level clipping/dropping of the
ratio makes it worse: truncating w removes exactly the mass that cancels this term.

## Fix
Weight each token score by the detached ratio w = p/q (token-level importance sampling), by default, with no
effective clipping (cap e^... = 20, only an overflow guard):
E_q[w grad log p] = sum_y q (p/q) grad log p = sum_y p grad log p = 0   (exactly, for each context),
so the zero-mean-score property that on-policy training relied on is restored: the baseline/group-centering
again contributes zero expected gradient, and the term A(q-p) becomes A (p-p) = 0. In general
E_q[A w grad log p] = sum_y p A grad log p = the on-policy policy gradient at that context (A as function of y
included), i.e. the expected update equals the p-on-policy update at the first-order (per-token context) level.
The sampled-token gradient is w grad log p = grad p / q, the gradient of the ratio surrogate E_q[(p/q) A].
Without correction the bias is E_q[A grad log p] - E_p[A grad log p], which is O(TV(p,q)) per token and
accumulates; with w the per-token bias vanishes, leaving only the prefix-ratio (sequence-level) mismatch, second order.
Since q is the sampler with logprobs recorded, no extra data are needed; works with correction=None.
An explicit `correction` argument still overrides the default.

## Configuration
