# Sampler/trainer mismatch in REINFORCE

## What is wrong
Fix a prefix and let the next token y ~ q (sampler); p is the trainer policy, z the logits, A the advantage.
The current loss (no correction) has per-position expected gradient
g_q = E_q[A(y) ∇_θ log p(y)] = Σ_y q(y) A(y) ∇ log p(y).

On-policy (q=p) this is Σ_y A(y)∇p(y) = ∇ E_p[A], the policy gradient, and a baseline b drops out because
E_p[∇log p] = Σ ∇p = ∇1 = 0. Group centering relies on exactly this.

For q≠p, write A = Ā + (A−Ā) with Ā the mean advantage at this prefix. The constant part gives
b·Σ_y q(y)∇log p(y) = −b ∇_θ H(q,p) = −b ∇_θ [KL(q‖p)] (q held fixed),
which is no longer zero (in logit space: b(q−p) ≠ 0). So a baseline/constant advantage is not neutral: gradient descent on
−A·log p with A<0 (the majority of tokens: failed rollouts) *maximizes* cross-entropy H(q,p), pushing p away from q,
unboundedly (log p of sampled tokens → −∞); A>0 pulls p toward q (sharpening). This is a systematic, signed bias of order
|Ā|·|q−p|, not zero-mean noise. Since the sampler is refreshed from the trainer, q tracks p, the drift
accumulates and the policy collapses. The advantage-correlated part is also weighted by q/p instead of 1, i.e. it is the gradient of a
reweighted objective.
Clipping/dropping weights w(p/q) breaks the cancellation further: E_q[min(p/q,c) A ∇log p] = Σ min(p,cq) A ∇log p, which
has a nonzero constant-advantage part wherever clipping is active, so tighter token clipping can be worse.

## Fix
Use the exact token-level importance weight, w = sg(p(y)/q(y)), in the default loss:
E_q[w A ∇log p] = Σ_y q (p/q) A ∇log p = Σ_y A ∇p = ∇E_p[A].
Hence the conditional expected update equals the on-policy gradient for every prefix, the baseline term
Σ_y p ∇log p = 0 vanishes again, and the KL-pushing bias is gone. The weight is kept essentially unclipped
(clamp only at ratio 50, for float safety; a clamp is a bias that only appears on the rare tokens beyond it, and "clamp" rather than "drop" keeps
their gradient direction). Gradient is w·A·∇log p = A·∇p/q, with w detached.
The default (correction=None) now applies this; an explicit correction argument still overrides it.
Note the prefix distribution under q still differs from p, but the per-position bias, which compounds, is removed.

## Configuration
