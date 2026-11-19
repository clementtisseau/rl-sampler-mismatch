# Stable REINFORCE under sampler/trainer mismatch

## What is wrong with the current loss
Fix a prefix. Let p be the trainer's next-token distribution, q the sampler's, and A the advantage
(group-centered, so it is not zero for a given prefix). The default loss is -A log p(x) with x ~ q, so its expected gradient is

  g_PG = E_q[A ∇log p(x)] = Σ_x q(x) A(x) ∇log p(x).

Write A(x) = Ā + δA(x), where Ā is the q-mean of A at this prefix. The Ā part contributes

  Ā Σ_x q(x) ∇log p(x) = Ā ∇ Σ_x q(x) log p(x) = -Ā ∇ H(q, p) = Ā ∇(-KL(q||p)) (q is constant).

- On-policy (q = p): Σ_x p ∇log p = ∇Σ_x p = 0. The baseline term vanishes, so any prefix-constant baseline or centering is harmless and the gradient is the true policy gradient.
- Mismatched (q ≠ p): this term is non-zero. It is a cross-entropy/KL gradient that pulls p toward q when Ā>0 and pushes p away from q when Ā<0. It is a systematic bias, not zero-mean noise, and it scales with Ā. It cannot be removed by a baseline, because the baseline is exactly what multiplies it. Because the sampler is refreshed from the trainer, the bias feeds back: the push away from q changes p, which changes q at the next step. The error compounds, and the entropy and probabilities of sampled tokens drift until training collapses.

Token-level clipping or dropping of w(p/q) is not a fix. With w = clip(p/q), E_q[A w ∇log p] = Σ_x A(x) min(p, cq)∇log p (for an upper clamp), so Σ_x w q ∇log p ≠ 0 whenever any token is clipped. The baseline identity is broken again, and a tighter clip breaks it more. This is why stronger clipping can be worse.

## The modification
With no correction passed, `loss_fn_rl` now uses unclipped token-level importance weights, w_t = p(x_t)/q(x_t), detached. The gradient is A w ∇log p = A ∇(p/q). The only cap is the numerical clip of log-ratio to ±20.

Expected gradient at a prefix:

  E_q[A w ∇log p] = Σ_x q(x) (p(x)/q(x)) A(x) ∇log p(x) = Σ_x p(x) A(x) ∇log p(x),

which is exactly the on-policy gradient, assuming q has support wherever p does. For the prefix-constant part Ā the term is Ā Σ_x ∇p(x) = Ā ∇1 = 0, so the bias above is gone for any baseline or centering. The estimator is unbiased at every position conditional on the prefix, so no bias accumulates through the sampler-refresh loop. The price is higher variance (heavy-tailed ratios) instead of bias. Variance is zero-mean and does not compound, and it is the correct trade because bias is what causes collapse.

Note on the sequence level: the full-trajectory ratio is a product of token ratios and has exploding variance. The per-token form is already unbiased for the per-token policy gradient given that the advantage is a function of the sampled sequence. This holds in the usual way for the policy-gradient theorem with the prefix distribution under q. Prefix-distribution mismatch is a second-order effect compared to the next-token bias analysed above.

## Configuration
rl.is.level=null
