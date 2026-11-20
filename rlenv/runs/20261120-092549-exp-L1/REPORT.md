# Stable RL under sampler/trainer mismatch

## What is wrong with the current loss

With `correction=None`, `loss_fn_rl` is plain REINFORCE on sampler tokens:

    L(θ) = -(1/N) Σ_i A_i Σ_t log p_θ(x_it | x_i<t)        (weight w = 1)

where x ~ q (the sampler, weights θ + δ, with δ a fixed relative perturbation,
or a quantized / stale copy) but the score is taken under p = p_θ (the trainer).
The unbiased trainer gradient is E_p[A ∇log p]. The loss instead estimates
E_q[A ∇log p], so the bias is

    b_t = E_q[A ∇log p] - E_p[A ∇log p].

Take the baseline part of the advantage first. Any advantage can be written as
A = (A - b) + b, where b does not depend on the sample's own tokens (a batch
or group mean). For the score of one categorical position with logits z and
p = softmax(z), ∇_z log p(v) = e_v - p, so for a constant b

    E_p[b ∇_z log p] = b (Σ_v p_v (e_v - p)) = 0           (the baseline identity)
    E_q[b ∇_z log p] = b (q - p)                            (≠ 0 under mismatch).

The baseline identity holds only under the distribution that generated the
samples. The current loss therefore contains the spurious term b (q - p) at
every position. Group centering does not remove it: a group-mean baseline
includes the other members' rewards, so the term survives with coefficient
-((G-1)/G) E[R] (q - p). Non-centered advantages (`plus_minus`, A = 2r-1)
make it large: b ≈ 2μ - 1 is not small.

Why it compounds. Let q = softmax(z + δ) with δ the sampler's logit noise.
For a negative constant advantage, a gradient step gives
Δz ∝ -b (q - p) ≈ |b| p ∘ (δ - E_p δ): the trainer moves its logits in the
direction of -δ, i.e. away from the noise the sampler will apply. The sampler
is then refreshed from the trainer, so the offset between trainer and sampler
grows with every step (the bias is systematic, not zero-mean). The logs show it:
with the current loss, the mean sampler-trainer KL on sampled tokens
(`kl_samp_train` = E_q[log q - log p]) grows steadily, and in the non-centered
mini setting the reward goes to zero.

## The modification

Use the per-token trainer/sampler ratio as the score weight, and normalize it
by its batch mean:

    w_t = p(x_t | ·) / q(x_t | ·) = exp(log p_t - log q_t)     (detached)
    Z   = (1/N) Σ_{i,t} mask_it w_it                           (detached)
    L(θ) = -(1/(N Z)) Σ_i A_i Σ_t w_it log p_θ(x_it | x_i<t)

This is the default (`correction=None`). Explicit `rl.is.*` corrections are
unchanged.

## Derivation

1. Exact per-position unbiasedness. Conditional on the prefix h_t, the sampled
   token is drawn from q, so

       E_q[ (p/q)(x_t) ∇log p(x_t | h_t) | h_t ] = Σ_v q_v (p_v/q_v) ∇log p_v = Σ_v p_v ∇log p_v = 0.

   The importance weight therefore removes the spurious baseline term exactly
   for every constant (or leave-one-out) baseline, which is the term identified
   above. Under q alone the same expectation is b (q - p) ≠ 0.

2. Why self-normalization. Clipping or banding w (TIS, MIS, PPO-type bands)
   changes E_q[w ∇log p] from 0 to a nonzero clipping bias. Stronger clipping
   therefore reintroduces the bias, consistent with observations that tighter
   token clipping can make things worse. Dividing by the batch mean Z instead
   keeps the weights unclipped. Since E_q[w_t | h_t] = 1 for each position, Z
   is an unbiased estimate of 1, and the estimator is a ratio estimator whose
   bias is O(1/N) (self-normalized IS). The same normalization bounds the
   effective step size when log-ratios are large, which is the case for
   int8/fp8 or heavy KV quantization.

3. Remaining bias. The per-token weight corrects the sampled token but not the
   ratios of the later tokens that the advantage depends on. The exact sequence
   correction (`rl.is.level=seq_sum`) has variance growing with the sum of
   log-ratios over the completion, which is very large at the mismatch levels
   tested, so it was not used as the default. The residual bias is second-order
   in the mismatch, since it needs both a nonzero future log-ratio and a
   nonzero advantage covariance with it.

4. Sign of the correction. The weights are below 1 where the sampler
   over-samples relative to the trainer, which are the tokens that the constant
   negative advantage would otherwise over-penalize. This is the term that
   pushed the trainer away from the sampler.

## Evidence (dev/mini.py, CPU; not required, for orientation)

- Noise 0.45, extra tokens 30, plus_minus advantages, 300 to 400 steps, 3 seeds:
  the current loss (`pg`) collapses to eval reward 0 in all seeds. The
  self-normalized ratio loss keeps eval reward about 0.2 to 0.6 across seeds
  and has a smaller KL drift.
- Group-centered advantages (train_rl's default) already make the plain loss
  much more stable. The ratio loss reduces the KL drift (pg KL reached 1.3 to
  5.2, the ratio loss 1.0 to 2.2) at similar clean eval reward.
- Clipped token IS (`tis`) was worse than unclipped at the same settings.

## Limitations

- The advantage-correlated bias from future-token ratios is not removed; see 3.
- The mini results are small-scale and noisy, and the effect of the
  normalization is smaller in the centered setting than in the non-centered one.
