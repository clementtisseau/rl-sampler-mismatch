# Sampler/trainer mismatch in REINFORCE

## What is wrong
Rollouts x ~ q (sampler), but the loss `-A * log p(x)` has gradient `A * grad log p(x)`. The policy gradient we want is
`g = E_p[A grad log p] = grad J(p)`. What the loss computes in expectation is, per token with context s:

E_q[A grad log p(x)] = sum_x q(x) A(x) grad log p(x).

Split q = p + (q - p). The p part is the true gradient. The remainder `sum_x (q-p)(x) A(x) grad log p(x)` is a bias. Even for A = const (a pure baseline shift, or any advantage not exactly mean-zero per context, e.g. token-count weighting of group-centered advantages, or plus/minus rewards) the gradient does not vanish:
`E_q[grad log p] = sum_x q grad log p = -grad_p CE(q,p) != 0`, whereas under p it is exactly 0 (`sum_x p grad log p = grad sum p = 0`). So the baseline no longer removes bias: the loss acts as a hidden SFT/cross-entropy term towards the sampler's own distribution q on top of the RL signal. The bias is systematic (same sign every step, set by the fixed perturbation of q: e.g. noise/quantization inflates mass on many unlikely tokens), not zero-mean noise. Because q is rebuilt from p each step, the induced drift moves p, which moves q: the errors compound and the policy can collapse (seen in `dev/mini.py` with a large tail vocabulary: plain PG goes to reward 0).

## Fix
Use the exact token-level importance weight w = p(x)/q(x) (detached) on the score term:
`E_q[ w A grad log p ] = sum_x q (p/q) A grad log p = sum_x p A grad log p = true gradient`,
and for constant A: `sum_x grad p = grad 1 = 0`, i.e. the baseline property is restored exactly. The gradient equals `A grad p / q`, the ratio-form surrogate, with the bias removed for any q with support covering p. (Remaining approximation: the token-level ratio ignores prefix mismatch, which is second order in the per-token mismatch.)

Why clipping does not help: a weight `min(w, C)` or masking gives `sum_x q min(p/q, C) A grad log p`, whose bias is `sum_{p/q>C}(p - Cq) A grad log p`, systematic and concentrated precisely on the tokens where mismatch is largest; tighter bands mean more bias. The remaining cost of exact IS is variance from rare tokens with p >> q, so the default uses only a very loose cap (w <= 10) which affects a negligible mass (bias bounded by the mass where p/q > 10) while removing the catastrophic-variance steps.

## Implementation
`loss_fn_rl` with `correction=None` now applies the token-level weight with high=10, outside=clamp (computed in log space, detached). Passing an explicit `ISCorrection` behaves as before. `rl.is.level=null` also maps to this default.

## Configuration
rl.is.level=token
rl.is.high=10.0
rl.is.outside=clamp
