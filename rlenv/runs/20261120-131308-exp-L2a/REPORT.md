# Report

## What is wrong with the current loss
The default loss (no correction) is REINFORCE with a sampler q: g = E_{y~q}[ A(y) ∇log p(y) ], with A group-centered sequence advantages.
Per token, at context s, the expected update is E_{a~q}[ A ∇logp(a|s) ].
Write A = Ā(s) + (A − Ā(s)), where Ā(s) = E_q[A|s] is the context-dependent mean advantage. The score function has zero mean under p but not under q:

  E_{a~q}[∇logp(a|s)] = Σ_a q(a|s) ∇logp(a|s) = Σ_a (q−p)(a|s) ∇logp(a|s) = −∇_θ KL(q(·|s) ‖ p_θ(·|s)) ≠ 0 when q ≠ p.

(On-policy q = p, so this is 0 exactly and any baseline/constant Ā drops out.) Hence the expected update contains the bias term

  b = Σ_s d_q(s) Ā(s) · (−∇_θ KL(q_s‖p_s)).

Group centering only removes the mean over sequences; Ā(s) is non-zero at most contexts (good prefixes have Ā>0, bad ones Ā<0, and length effects too). So for Ā(s)<0 the update *increases* KL(q‖p) (pushes p away from q), for Ā>0 it decreases it. The bias is systematic, proportional to the mismatch, not zero-mean noise. Since q is regenerated from p each step, the discrepancy and bias compound, giving drift and collapse. Importance weights with clipping/dropping (TIS, MIS, PPO-style, ...) multiply by w(r) ≠ r, so E_q[w ∇logp] = Σ q w ∇logp ≠ Σ p ∇logp = 0 still: the bias is only partly removed, and tighter token clipping makes w deviate more from r, so the zero-mean identity is broken more (hence worse).

## Fix
Restore the zero-mean identity with the exact per-token importance weight r = p/q:

  E_{a~q}[ (p/q) ∇logp ] = Σ_a p ∇logp = ∇Σ_a p = 0,

so with w = p/q the expected update is E_p-based, Σ_s d(s) E_{a~p}[A ∇logp], i.e. an (unbiased in the per-token action sampling) policy gradient for p with no bias term from Ā(s) times (q−p). The loss is now -Σ A · sg(p/q) · log p (equivalently A·∇p/q), by default and without config. Weights are computed in log space and only guarded against numerical overflow (clamp of ratio to [e^-4, e^4], which is outside the range of normal mismatch), never "dropped" or tightly clipped, because any substantial truncation re-introduces b. Passing an explicit `correction` still uses that correction. Variance is higher than for plain PG for large mismatch, but it is variance, not drift, and the mismatch here is small (per-token log-ratios of order 0.05).

Remaining (state-distribution) mismatch d_q(s) vs d_p(s) is second-order and is not compounding through the score-function bias.

## Configuration
rl.is.level=null
