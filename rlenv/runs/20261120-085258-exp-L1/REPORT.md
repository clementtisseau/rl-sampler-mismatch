# Sampler-centered REINFORCE: removing the mismatch bias from the baseline

## Summary

The REINFORCE surrogate `A * log p(y)` uses the trainer's score function
`grad log p(y)`. That score has zero mean under the trainer `p`. It does **not**
have zero mean under the sampler `q` that actually drew `y`. So every part of
the advantage that does not depend on `y` (the group mean, the value of the
current prefix, a whole group of failures with `A = -1`) does not cancel. It
leaves a deterministic update along `q - p`, the mismatch direction. This
term is a bias, not noise, and the next sampler is rebuilt from the updated
trainer, so it builds up over steps.

The fix is to center the score under the distribution the tokens were
sampled from:

    loss = - sum_t A * w_t * ( log p(y_t) - sum_v m_t(v) log p_t(v) ) / n_tokens,
    m_t(v) = sg( q_t(v) w(p_t(v)/q_t(v)) / sum_u q_t(u) w(p_t(u)/q_t(u)) )

With the default `correction=None`, `w = 1` and `m_t = q_t`, the sampler's
next-token distribution. `q_t` is known exactly on the sampler's top-k tokens
(`sampler_topk_ids/logprobs`, which `train_rl.py` provides). The leftover mass
`1 - Q_top` is spread over the other tokens in proportion to `p`. The
centering term has no effect on any on-policy quantity. It only removes, exactly,
the drift that the mismatch adds through the baseline.

## 1. What is wrong with the current loss

Take a single position (state `s`) with trainer logits `z`, `p = softmax(z)`,
and sampler distribution `q`. The sampler may be noisy, quantized or stale. Write
`e_y` for the one-hot vector of token `y`. Then

    d log p(y) / dz = e_y - p.

Let `A(s, y)` be the advantage the token receives. In the sequence-level
setup this is the sequence advantage, and its conditional mean given `(s, y)`
is `Q(s, y) - b(s)`. Here `b(s)` collects everything that does not depend on
`y`: the group mean, and the effect of the tokens before `s`. The current
loss (PG, no correction) gives the expected update in logit space

    g_PG = E_{y~q}[ A (e_y - p) ].

Split `e_y - p = (e_y - q) + (q - p)`:

    g_PG = E_{y~q}[ A (e_y - q) ]  +  E_{y~q}[A] * (q - p)          (1)
         = q ⊙ (Q - Q̄_q)          +  (Q̄_q - b) * (q - p)

where `Q̄_q = E_q[Q(s, ·)]` and `⊙` is the elementwise product.

* The first term does not depend on the baseline `b`. It equals
  `∇_{z_q} E_{y~q}[Q(s,y)]`, the exact policy gradient of the sampler's own
  logits.
* The second term is the bias. Its size is the mean advantage of the state,
  `Q̄_q - b`, times the mismatch vector `q - p`. Note that
  `q - p = ∇_z E_q[log p] = -∇_z KL(q || p)`. So the bias is a KL term between
  sampler and trainer whose sign is set by the sign of the state's mean
  advantage:
  - at states with positive mean advantage, the trainer is pulled toward the
    sampler. It absorbs the sampler's error, the sampler is then rebuilt from it
    with the same perturbation on top, and the pull starts again;
  - at states with negative mean advantage (any failing group, all of
    `plus_minus` early on, the later tokens of failed rollouts), the trainer is
    pushed **away** from the sampler, which increases the mismatch.

This explains the observed behaviour:

1. **It is a bias, not noise.** Its direction `q - p` is set by the fixed
   mismatch, not by the sample. Averaging over many rollouts does not remove it.
   It is zero only if `q = p`, which the mismatch rules out.
2. **It compounds.** For a fixed logit offset, `q - p ≈ F ε` with
   `F = diag(p) - p pᵀ` and `ε = z_q - z_p`. Each step therefore moves `z` by
   `η (Q̄_q - b) F ε` in an almost constant direction. Since `q` is recomputed
   from the moved trainer, the update never closes the gap, and the
   displacement grows linearly with the number of steps.
3. **Group centering does not remove it.** `sum_i A_i = 0` holds only over the
   whole group, at the prompt. At a later position `t` the relevant quantity
   is `E[A | prefix]`. This is nonzero for nearly every state after the first,
   because the prefix already says a lot about the outcome. When every rollout
   in a group gets the same reward, `A` is constant. The useful term in (1)
   then vanishes and only the bias remains (for example `plus_minus` with all
   rollouts failing gives `-(q - p)` at every token).
4. **Why IS corrections and clipping only help partly, and stronger clipping
   can hurt.** With a detached weight `w(y) = w(p(y)/q(y))`:

       E_q[ w A (e_y - p) ] = E_q[ w A (e_y - m) ] + (E_q[w A]) (m - p),
       m := q w / E_q[w].                                                 (2)

   The bias is now proportional to `m - p`. Exact IS (`w = p/q`) gives
   `m = p`, so there is no bias, but the variance is unbounded. Truncation,
   clamping, dropping or PPO-style clipping all move `w` away from `p/q`, and
   so move `m` back toward `q`. The tighter the band, the larger `m - p`, and
   the closer the bias gets to the uncorrected one. Token-level clipping also
   changes `E_q[w]` with the sign of `A` (`sign=ppo`, `neg`), which adds its
   own systematic drift.

## 2. The modification

Use a score function centered under the distribution that actually produced
the token, so that its expectation under the sampler is exactly zero:

    s_t(y) = log p_t(y) - sum_v m_t(v) log p_t(v),    m_t = sg(q_t w_t / E_{q_t}[w_t]),
    loss   = - (1/N) sum_{i,t} A_i w_t(y_t) s_t(y_t).

Gradient with respect to the logits, using `sum_v m(v) = 1`:

    d s(y) / dz = (e_y - p) - sum_v m(v)(e_v - p) = e_y - m.

So the expected update becomes

    g_new = E_{y~q}[ w A (e_y - m) ].

**Baseline invariance (the key property).** For any `c` that does not
depend on `y`:

    E_{y~q}[ w(y) c (e_y - m) ] = c ( q ⊙ w - E_q[w] m ) = 0,

by the definition of `m`. Every `y`-independent part of the advantage drops out
exactly. This covers the group mean, the prefix's value, the all-fail group, and
the `plus_minus` offset. The bias term `E_q[wA](m - p)` in (2) is exactly what
gets subtracted. Without a correction (`w = 1`, `m = q`):

    g_new = E_q[ A (e_y - q) ] = q ⊙ (Q - Q̄_q) = ∇_{z_q} E_{y~q}[Q(s, y)].

This is the first term of (1) with the bias removed.

**Why this gives stability.**

* There is no drift. The update is the policy gradient of the policy that
  generated the data, evaluated in its own logits. Every term in it is
  multiplied by `Q(s,y) - Q̄_q(s)`. So it vanishes when the sampler is optimal
  on its support, and it vanishes for groups with no reward signal, both
  per sample in expectation and for any constant `A`. In the old loss those
  cases still produced `(Q̄ - b)(q - p) ≠ 0` at every step.
* The mismatch now only changes *where* the gradient is evaluated, not *what*
  it optimizes. Since `θ` moves the trainer's logits and the sampler is rebuilt
  as the trainer plus a perturbation, an update `Δz = η q ⊙ (Q - Q̄_q)`
  moves the sampler's logits by the same `Δz` to first order. The error from
  the different Jacobians `∂z_q/∂θ` and `∂z_p/∂θ` (or from a stale snapshot) is
  a relative perturbation of a true ascent direction of the sampler's return. It is
  not a constant offset, so it does not accumulate when the signal is zero.
  The sampler's return, which is the training reward, improves monotonically
  for small steps, just as in on-policy REINFORCE.
* It combines with corrections. With `w = p/q` it becomes standard
  importance-sampled REINFORCE with the usual `p`-centered score (`m = p`). With any
  truncated or clipped `w`, centering at `m ∝ q w` keeps the baseline
  invariance that clipping alone breaks. `log_mass` already computes
  `log(q w(p/q))`, so `m` is that function applied to every vocabulary
  entry and then normalized.
* It adds no variance and needs no tuning. It is a control variate with zero
  expectation under `q`, the same per-token "baseline" that on-policy
  REINFORCE gets for free from `E_p[∇ log p] = 0`. Nothing is clipped, so no
  signal is thrown away.

**Sequence-level view.** The loss is `-sum_i A_i sum_t w_t s_t(y_t)`. For each
`t`, conditioning on the prefix `s_t` gives
`E_{y_t~q_t}[w_t s_t'(y_t) | s_t] = 0`. Any part of `A_i` that is fixed given
`(s_t)` therefore contributes nothing at token `t`. With sequence-level
corrections (`seq_mean`, `seq_sum`) the weight is constant within the
sequence. In that case `m_t = q_t`, and the sequence weight multiplies the
centered score.

## 3. Implementation (`losses.py`)

* `centering_logmass` returns `C_t = sum_v m_t(v) log p_t(v)` with detached
  `m_t`, and the score becomes `w * (log p(y) - C_t)`.
* **Tail estimate.** The sampler reports its top-k (k = 20) logprobs. Let
  `Q_top` and `P_top` be the masses that `q` and `p` give to those k tokens.
  `q̃` takes the sampler's values on the top-k. On the rest of the vocabulary it
  takes `p_v (1 - Q_top) / (1 - P_top)`: it keeps the sampler's total tail mass
  and uses the trainer's shape inside the tail. Under a correction, the tail
  tokens share the single ratio `(1 - P_top)/(1 - Q_top)`. The tail sum is
  computed as `(sum_all p log p - sum_top p log p) / P_tail`. Its gradient is
  exact: the top-k contributions cancel elementwise. `P_tail` is
  floored at `1e-6` for safety.
* If the batch has no top-k logprobs, the sampled token is used as the known
  set (`q̃(y) = q(y)`, the rest in proportion to `p`). This removes part of the
  bias.
* Checked numerically: with an identity head, the gradient with respect to
  the logits equals `-(A/N)(e_y - q̃)` to within 1e-8.
* The signature and metrics are unchanged. The modification is always on,
  works with `correction=None`, and works with every `rl.is.*` correction.

## 4. Sanity check on the CPU proxy (`dev/mini.py`, 600 steps)

Final eval reward of the clean trainer, followed by the training (sampler)
reward over the last 50 steps:

| setting | original loss | centered loss |
| --- | --- | --- |
| default dev (noise 0.3, plus_minus) | 0.66 (train 0.58, plateau) | 0.80 (train 0.84) |
| group_centered, noise 0.5 | 0.57, falling (train 0.41) | 0.70, rising (train 0.51) |
| 40 extra tail tokens, noise 0.3 | **collapse to 0.00** | 0.75–0.87 (train 0.80) |
| 40 extra tail tokens, noise 0.5 | **collapse to 0.00** | 0.74 (train 0.51) |
| extra tokens, PPO clip (0.8, 1.2) | **collapse to 0.02** | 0.69, rising |
| extra tokens, TIS (high=2, clamp) | 0.82 | 0.90 |

At very large mismatch (noise 0.5), the log-ratio between sampler and trainer
still grows slowly. The weight perturbation is relative, so its effect on the
logits grows as the policy sharpens. The reward does not collapse.

## Configuration

No configuration changes are needed. The modification is active with the
default `rl.is.level=null`.
