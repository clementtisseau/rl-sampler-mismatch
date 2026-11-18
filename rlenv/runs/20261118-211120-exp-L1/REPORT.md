# Stable REINFORCE under sampler/trainer mismatch

## What is wrong with the current loss

Let q be the sampler, p the trainer, tau a completion, A its advantage (group-centered), and
rho_t = p(x_t | h_t) / q(x_t | h_t) the per-token ratio. The current loss (correction=None) is
`-sum_t A * log p(x_t|h_t) / N`, with every token weighted 1. Its expectation over rollouts is

    g_none = E_q[ A * grad log p(tau) ]

while the quantity we want, the gradient of J = E_p[R], is

    g_true = E_p[ A * grad log p(tau) ] = E_q[ (p(tau)/q(tau)) * A * grad log p(tau) ].

So the uncorrected gradient is the gradient of a different objective, weighted by the wrong distribution.
The error is not zero-mean noise: it is a fixed functional of (q, p) at every step.

Two parts of the error matter.

1. Support mismatch (off-support tokens). At a position t with prefix h, the score of a sampled token has
   conditional mean under q that is not zero:

       E_q[ grad log p(x|h) | h ] = sum_x q(x|h) grad log p(x|h) = -grad H(q_h, p_h),

   the gradient of the cross-entropy from q to p. Under p it is exactly zero, since sum_x p grad log p = 0.
   For a token the trainer never emits (p ~ 0) that the noisy sampler does emit (q > 0), the uncorrected
   update contributes an O(1) score (e_x - p) that is not multiplied by any p-probability, so it is not
   suppressed by the trainer's own likelihood. In the dev setting with extra never-correct tokens the
   sampler puts mass ~0.15 on them at step 0 while the trainer puts ~0; this drives the collapse.

2. The constant-advantage part of the bias is exactly what IS removes. For constant advantage c,

       E_q[ rho_t grad log p(x_t|h_t) | h_t ] = sum_x p(x|h) grad log p(x|h) = 0,

   so token-level importance weights kill the pull toward q exactly. Any change to the weights that is
   not linear in rho breaks this identity.

## Why the usual corrections only slow the drift

- Clamping, w = min(rho, c) (TIS), gives E_q[min(rho,c) grad log p] = -E_q[(rho - c)_+ grad log p] != 0.
  The leftover bias is a pull on the tokens where p > c q (under-sampled tokens), with size growing as c
  shrinks. This is why a stronger clip can be worse: dev run with c = 1.2 collapsed on one of four seeds
  where c = 2 did not.
- Dropping outside a band (MIS, PPO-type masks) removes mass from one side and leaves a bias with
  the opposite sign. In dev, MIS let the off-support mass grow again on one seed (sampler mass on extra
  tokens rose to 0.67).
- Sequence-level weights (seq_sum, seq_mean) are unbiased for the sequence objective but the product of
  per-token ratios has huge variance over 500+ tokens. In dev they collapsed on almost every seed.
- Unclipped token weights are unbiased for the constant part but p/q has heavy outliers where the sampler
  under-samples a token the trainer likes. One of four dev seeds collapsed.

## The fix

For correction=None, `loss_fn_rl` now uses the token-level weight, bounded and self-normalized:

    w_t = stop_gradient( min( p(x_t|h_t) / q(x_t|h_t), c ) ),   c = DEFAULT_CLIP = 2,

    loss = - sum_{i,t} A_i * w_t * log p(x_{i,t}|h_{i,t})  /  sum_{i,t} w_t.

Why each part helps.

- The token weight rho = p/q is the only per-token weight with E_q[rho grad log p] = 0 (see above), so it
  removes the cross-entropy pull and turns off-support tokens into zero-weight tokens. This is the
  part that stops the uncorrected drift.
- The clamp at c bounds the variance of the rare large ratios. Its bias is E_q[(rho - c)_+ grad log p],
  which is small when mismatch is moderate and is the only remaining deviation from the unbiased
  estimator. Choosing c = 2 keeps this term small (c = 1.2 was measurably worse).
- Self-normalization divides by the batch sum of weights instead of the token count. The weights have
  mean about 1 when the sampler is close to the trainer, so the scale of the step is unchanged in that
  case. When the sampler is far from the trainer and many weights are near 0, the denominator shrinks
  and the step is not inflated. The estimator is a ratio estimator, so it is biased by O(1/N_eff),
  which vanishes as the batch grows; it is not a source of the systematic drift.

Limits. The per-token weight does not correct the future-token coupling: the reward-to-go Q_t is
evaluated on q-sampled continuations, so the bias E_q[rho_t (Q_t - b)] - E_p[...] is not removed by any
per-token weight. Sequence-level weights remove it but have prohibitive variance at this length. The
bounded token weight is therefore a bias-variance compromise, not an exact fix; the bias left is the
clamp term and the future term.

## Evidence (dev/mini.py, 600 steps, final eval reward, seeds 0-3, extra never-correct tokens = 200)

| Loss (correction=None unless noted) | seed 0 | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- | --- |
| plain PG (old default) | not run | 0.10 (collapse) | 0.00 (collapse) | 0.00 (collapse) |
| naive token IS, no clamp | not run | 0.75 | 0.21 | 0.46 |
| TIS, clamp 2 | 0.81 | 0.78 | 0.78 | 0.75 |
| MIS band [0.5, 5] | 0.05 | 0.62 | 0.56 | 0.67 |
| seq_sum / seq_mean, no clamp | ~0 | 0.10 | ~0 | ~0 |
| TIS, clamp 1.2 | 0.00 (collapse) | 0.75 | 0.62 | 0.74 |
| **new default (clamp 2, self-normalized)** | 0.77 | 0.77 | 0.77 | 0.77 |

Without the extra tokens the plain PG default is stable (about 0.66, four seeds), which fits the support
explanation. The same mismatch with the extra tokens is what makes it collapse.

I also checked the noise-absorption hypothesis: the projection of the trainer's drift onto the fixed
sampler perturbation stayed about 0 in the plain PG run, so the drift is not the trainer absorbing the
noise along its own direction. The mechanism above (off-support score and clamp bias) is what the
evidence supports.

Caveats. Only the mini proxy was run (weight noise plus extra tokens; no staleness, int8/fp8, or KV
quantization, and no GPU run). The clamp constant c = 2 was chosen from this proxy and is not tuned to
the hidden settings.

## Configuration

No configuration changes are required. The default `correction=None` uses the new loss; with an
explicit `rl.is.*` correction the previous code path is unchanged.
