# Sampler-trainer mismatch in REINFORCE: a biased score term, and its fix

## 1. What is wrong with the current loss

With `correction=None` the loss is `-sum_t A_i * sg(1) * log p(y_t)` / N, i.e. the
REINFORCE estimator evaluated on samples drawn from the sampler q, not the trainer p.
The on-policy gradient is

    g* = E_{y~p}[ A(y) grad log p(y) ],

while the code computes

    g_cur = E_{y~q}[ A(y) grad log p(y) ].

Take one position with logits z, p = softmax(z), and the sampler's next-token
distribution q. Since grad_z log p(y) = e_y - p,

    E_q[A (e_y - p)] = sum_y q(y) A(y) e_y - E_q[A] p,
    E_p[A (e_y - p)] = sum_y p(y) A(y) e_y - E_p[A] p.

For a constant advantage offset b (for example `plus_minus` rewards, where every unsolved rollout gets -1),
the difference is the systematic term

    bias = b (q - p)        (in logit space, per position).

It is not zero-mean noise. Its sign is fixed by b: with b < 0 the update lowers the logits of every
token the sampler over-produces (q > p) and raises the rest, so it pushes the trainer away from the sampler
at every step. Because the sampler is refreshed from the trainer, the offset is re-applied at every step and the
distortion accumulates. The mini-benchmark shows this directly. With 200 extra vocabulary tokens
(`MINI_EXTRA_TOKENS=200`, sampler tail mass about 0.175 at step 0), the current loss goes from reward 0.30 to 0
within about 20 steps, and the tail mass seen by the sampler falls to about 0.002 while the reward stays at 0.

Clipping does not remove this bias. Replacing w by min(w, c) for w = p/q changes the expectation over the
tokens where w > c. It is a bias of its own, and it grows as c shrinks, which matches the observation that
stronger token clipping can make things worse.

## 2. The fix: exact per-token importance weights and a free baseline

Two facts remove the bias.

**(a) Per-token IS is exact for the score term.** Let w_t = p(y_t | prefix) / q(y_t | prefix). Since y_t ~ q
given the prefix,

    E_q[ w_t grad log p(y_t) | prefix ] = sum_y q(y) (p(y)/q(y)) grad log p(y)
                                        = sum_y grad p(y) = 0.

For a constant b, the IS-weighted term has zero mean, so E_q[ w_t b grad log p(y_t) ] = 0, while the
unweighted term has mean b E_q[grad log p] = b(q-p) != 0. Multiplying the score by w_t turns the sampler
expectation into the trainer expectation at each position. For the advantage, the remaining error is the
dependence of A on the later tokens' ratios, which is the standard token-level approximation of the
sequence-level on-policy gradient and is second order in the mismatch.

**(b) A constant baseline is free under IS, and it is not free without IS.** From (a),
E_q[w_t b grad log p] = 0 for any constant b, so subtracting a batch-level constant from the advantages
leaves the expectation unchanged. The variance of the gradient is reduced: with plus_minus rewards most
rollouts carry a -1 that multiplies a noisy score, and removing the batch mean removes that constant
contribution to the variance. Without IS, subtracting b would change the bias (it adds b(q-p) to the mismatch term),
so the baseline is only valid together with the IS weight.

The loss implemented in `losses.py` when `correction=None` is therefore

    L = -(1/N) sum_{i,t} sg(w_{i,t}) (A_i - mean_j A_j) log p(y_{i,t}),
    w_{i,t} = exp(clip(log p - log q, -20, 20)),

with no clipping of w. The clip at +-20 only prevents overflow. This is the naive token-level IS of the README
(`rl.is.level=token`, no band) plus a batch-mean baseline. No flag is needed, and `rl.is.*` is not needed.
When `correction` is given, the original grid is used unchanged.

## 3. Why this is stable where the others are not

- The bias term b(q-p) of the current loss is replaced by b(p-p) = 0 in expectation.
- The variance from the constant offset is removed by the baseline, which costs no bias.
- Clipping (TIS, MIS, PPO-style bands) keeps a bias that grows with the mismatch, so it slows the collapse
  without removing it.
- Sequence-level IS (`seq_sum`, `seq_mean`), which is exact in expectation, has a product of T ratios whose
  variance is far larger. In the mini-benchmark it collapsed on every seed tried.
- Self-normalized IS (dividing the weighted sum by the batch sum of weights) was also tried. It removes the
  constant-offset bias only asymptotically and was less stable than the baseline version.

## 4. Empirical checks (mini benchmark, 600 steps, `reward_mode=plus_minus`, `noise=0.3` unless stated)

| loss | extra tokens | seed | final eval reward |
| --- | --- | --- | --- |
| original (no IS) | 200 | 0 | 0.00 (collapsed at ~step 20) |
| naive token IS, no baseline | 200 | 0 / 1 / 2 | 0.72 / 0.75 / 0.21 |
| TIS (`high=2 clamp`) | 200 | 0 | 0.81 |
| self-normalized IS | 200 | 0 / 1 / 2 | 0.11 / 0.63 / 0.01 |
| **this loss (IS + batch-mean baseline)** | 200 | 0 / 1 / 2 / 3 | 0.53 / 0.65 / 0.54 / 0.52 |
| **this loss** | 0 | 0 | 0.78 |
| original, noise 0.6 | 200 | 0 | 0.00 |
| TIS, noise 0.6 | 200 | 0 | 0.00 |
| **this loss**, noise 0.6 | 200 | 0 | 0.44 |

With the `group_centered` reward (the trainer default), naive IS and self-normalized IS did not collapse on the
seeds tried (final eval 0.53 to 0.78). This is consistent with the baseline argument, since the offset is already
zero there. The instability on `plus_minus` is the constant-offset variance that the baseline removes.

These are one-to-four-seed runs on a CPU proxy and are indicative only.

## Configuration

No configuration changes are needed. The new default is active with the trainer's `rl.is.level=null`.
