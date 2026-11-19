# q-centered score REINFORCE

## What is wrong
Per token the loss is `-A * log p(a_t | s_t)`, with `a_t ~ q(.|s_t)`; its gradient in expectation is
`E_q[A * grad log p(a)]`. Write the advantage at a state as `A = Abar + (A - Abar)`.

* On-policy (q = p): `E_p[grad log p(a)] = sum_a grad p(a) = 0` (score identity). Any term that does not depend on `a` (baseline, group mean, length-dependent weights) therefore contributes zero, and the update is the true policy gradient.
* Mismatched (q != p): `E_q[grad log p(a)] = grad sum_a q(a) log p(a) = -grad CE(q, p) = -grad KL(q||p)` (up to constants). This is nonzero. So every advantage component that is not `a`-dependent now contributes `Abar * (-grad KL(q||p))`, a systematic drift that pulls p toward the sampler q (mass-covering, towards q's errors/noise) with weight Abar and per-sequence token counts. Group-centering only makes the advantages sum to zero over the group, not per state and not weighted by sequence length. Wrong or long samples carry more tokens, so the weighted sum of these drift terms does not cancel. Because the sampler is refreshed from the trainer each step, the bias acts on every step and compounds: it is a bias in the objective, not zero-mean noise.
* Clipping/IS weights `w(p/q)` fix the identity only if unclipped (`E_q[(p/q) grad log p] = 0`). Token-level clipping or dropping breaks the identity again (`E_q[w grad log p] != 0`), and it is high variance, so it slows the problem but does not remove it. Stronger clipping removes more of the correction and can be worse.

## Fix
Subtract from the score a per-state baseline that depends only on the state and q, the cross-entropy `b(s) = E_{v~q}[log p(v|s)]`, treated as a function of p (gradient flows through p, q is constant):

`L = - sum_t A * ( log p(a_t|s_t) - b(s_t) )`, `b = sum_v q(v|s) log p(v|s)`.

Then `E_q[grad(log p(a) - b)] = grad b - grad b = 0` for every state, exactly as the score identity gives zero on-policy. An `a`-independent advantage component contributes zero for any q, p, and the baseline term `-Abar*(-grad KL)` drift is removed. The remaining update is `E_q[(A-Abar) grad log p(a)]` (the policy-gradient signal), with the bias that `A` depends on state distribution shift only (second order). The estimator has no clipping and no weights so it adds no variance of the importance ratio type, and is unchanged when q = p up to a zero-mean term (b = -H(p) gradient part vanishes).

Implementation: the sampler returns its top-20 logprobs per position (`sampler_topk_ids`, `sampler_topk_logprobs`). b is computed from them: `b = sum_{k in top20} q_k log p_k + q_tail * log(1 - sum_k p_k)` where the tail is one bucket (exact if q and p agree within the tail). If the top-k are missing from the batch, the loss falls back to plain REINFORCE. It is active with default arguments and `correction=None`; a correction, if passed, still multiplies the centered score.

On the dev setting (noise 0.3) the mean reward over the last 10 steps went from about 0.58 (plain, plateau and decline) to about 0.85 and still improving.

## Configuration
