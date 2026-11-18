# Drift-free REINFORCE under a mismatched sampler

## Summary

Under a sampler `q != p`, the REINFORCE score `grad log p(y|h)` no longer has
zero mean at a prefix, so the expected update contains a term
`E_q[A|h] * s_bar(h)`. That term does not come from the reward. It is a
systematic force that pushes `p` toward or away from `q`, and it starts over
at every step because the sampler is refreshed from the trainer. The fix
replaces the score with the **sampler-centered score**

    s_y - s_bar(h),        s_bar(h) = sum_v q(v|h) grad log p(v|h),

which in the loss is

    loss = - sum_{b,t} A_b * [ log p(y_t|h_t) - sum_v sg(q(v|h_t)) log p(v|h_t) ] / n_tokens

(`sg` = stop-gradient). With respect to the logits `z(h)` of the trainer,
the per-token gradient is `A (e_y - q)` instead of `A (e_y - p)`. Its
conditional expectation under `q` is exactly the covariance term, for any
advantage and any baseline. On-policy the extra term has gradient exactly 0
(sample by sample, not only in expectation), so nothing changes when
`q = p`. The method is on by default in `loss_fn_rl`. It needs the sampler's
next-token distribution (`sampler.vocab_logprobs`, see Configuration). When
that is not recorded, it falls back to a weaker approximation that uses only
`q(y)`.

## 1. What is wrong with the current loss

Notation: at a prefix `h` the trainer has logits `z(h)`, `p = softmax(z)`, and
the sampler draws `y ~ q(.|h)`. `J_h = dz(h)/dtheta`. In logit coordinates the score
of token `v` is

    s_v = grad_theta log p(v|h) = J_h^T (e_v - p(.|h)).

The current loss is `-A * log p(y|h)` summed over completion tokens. Each
token contributes `A s_y` to the ascent direction. Condition on `h` and write
`Q(h,v) = E[R | h, v]` for the expected return when the rest of the completion
also comes from `q`, `V(h) = sum_v q(v|h) Q(h,v)`, and `b` for the baseline
(the group mean in `group_centered`, 0 in `plus_minus`). Then `A = R - b` and

    E_q[A s_y | h] = sum_v q_v (Q(h,v) - b) s_v
                   = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),           (1)

    s_bar(h) = sum_v q_v s_v = J_h^T (q - p).                           (2)

**On-policy** (`q = p`), `s_bar = J^T(p - p) = 0`, equivalently
`sum_v p_v grad log p_v = grad sum_v p_v = 0`. Only the covariance remains, and
the baseline has no effect on the mean.

**Under mismatch** the first term is the drift

    D = sum_h d_q(h) (V(h) - b) J_h^T (q(.|h) - p(.|h)),                 (3)

where `d_q(h)` is how often prefix `h` is visited. Three properties make it harmful:

* **It is not reward signal.** It moves the logits along `q - p`, which is
  set by the sampler's error (noise, quantization, staleness) and not by which
  tokens were good. The baseline does not cancel it. `b` is constant within a
  group but `V(h)` changes along the completion, so `V(h) - b != 0` at almost
  every prefix past the first token. `group_centered` only makes the drift
  vanish at the root.
* **It has a consistent sign; it is not zero-mean noise.** `q - p` is a
  deterministic function of the current weights (a fixed noise draw, a fixed
  quantizer). The sign of `V(h) - b` correlates with it: a prefix that
  contains a token the sampler over-weights (often a tail token that noise or
  quantization inflates) has lower value. So D points the same way at every
  step and does not average out.
* **It compounds.** The sampler is refreshed from the trainer at every step,
  so `p` never catches up with `q`: after each update a new `q - p` of the same
  kind appears. For `V(h) < b`, the logits move along `-(q - p) = p - q`.
  When the sampler is a blurred copy of `p` (weight noise and quantization add
  entropy and tail mass), `p - q` is positive on `p`'s mode and negative on the
  tail, so `p` sharpens at every step. This continues until entropy collapses
  and `p` becomes deterministic on a bad answer. For `V(h) > b`, `p` is pulled
  toward `q`, so it absorbs the sampler's error, and the refreshed sampler then
  adds the error once more. That is a walk along the perturbation direction.
  Either way the displacement grows linearly in the number of steps
  (`O(lr * steps * |D|)`), not like `sqrt(steps)` as zero-mean noise would.

**Why IS corrections and clipping only slow this down.** A token-level
weight `w(r_y)`, `r = p/q`, turns the conditional mean score into

    s_bar_w(h) = sum_v q_v w(r_v) s_v = sum_v m_v s_v,   m_v = q_v w(p_v/q_v).

Untruncated IS gives `m = p` and `s_bar_w = 0`, but its variance is
unbounded. Any truncated or band-limited weight gives `m != p` and therefore a
drift `E[A|h] J^T(m/|m| - p)` scaled by `|m|`. Examples:
`m_v = min(p_v, c q_v)` (TIS), `m_v = p_v 1[r_v in band]` (MIS/PPO-style drop),
for which `sum_v m_v s_v = -sum_{out of band} p_v s_v`. The tighter the clip,
the more mass falls outside the band and the larger this residual. This is
why stronger token-level clipping can make things worse. Sequence-level
weights multiply `A` by a scalar and leave (1) unchanged, so the
`E[A|h] s_bar` term remains. Methods that use only the sampled token's ratio
cannot remove the drift at all, because `s_bar(h)` is an expectation over
tokens that were not sampled.

## 2. The modification

Subtract the conditional mean of the score under `q`, which is a control
variate:

    g = A (s_y - s_bar(h)).                                             (4)

Since `E_q[s_y - s_bar(h) | h] = 0`,

    E_q[g | h] = sum_v q_v (Q(h,v) - b)(s_v - s_bar) = Cov_q(A, s_y | h).  (5)

The drift is removed exactly, for any advantage definition and any
baseline `b`. Only the part of the update that correlates with the reward
remains.

As a loss, `s_bar(h)` is the gradient of `sum_v sg(q_v) log p(v|h)`. Hence

    L = -(1/N) sum_{b,t} A_b [ log p(y_t|h_t) - sum_v sg(q(v|h_t)) log p(v|h_t) ],

and the gradient with respect to the trainer logits at each token is

    dL/dz = -(A/N) [(e_y - p) - (q - p)] = -(A/N) (e_y - q).            (6)

`p` cancels. The trainer pushes up the sampled token relative to the
distribution the token was actually drawn from, not relative to its own.

**The update is the gradient of a real objective.** Model the sampler as the
trainer's logits plus a perturbation, `q(.|h) = softmax(z(h) + eps(h))`, with
`eps` treated as fixed during the step. The policy-gradient theorem applied to
the sampler's own return `J(q) = E_q[R]` gives

    grad_theta J(q) = sum_h d_q(h) J_h^T sum_v q_v Q(h,v) (e_v - q),

because `dz_q/dz = I`. By (5)-(6) this is exactly the expected update of the
new loss (`b` drops out since `sum_v q_v (e_v - q) = 0`). The update of plain
REINFORCE is

    E[update_REINFORCE] = grad_theta J(q) + sum_h d_q(h) (V(h) - b) J_h^T (q - p),

which is `grad J(q)` plus the drift (3). The drift is generally not a
gradient field. That is why it can circulate and accumulate indefinitely
instead of settling at a stationary point. With the centered score, training
is stochastic gradient ascent on `J(q_theta)`, the return of the rollout
policy itself, so there is no force beyond the objective. When `eps` is small,
`J(q)` and `J(p)` differ by `O(|eps|)` and have nearby maximizers. The
remaining bias is a slightly perturbed objective, not a drift that
accumulates across steps. (For weight noise `eps` also depends on `theta`;
neglecting `d eps/d theta` costs a second-order term `O(|eps| * |grad|)`, not
a first-order force along `q - p`.)

**On-policy it is identical to REINFORCE.** When `q = p`, the gradient of the
extra term is `sum_v p_v grad log p_v = 0` exactly, for every sample. The
modification therefore only acts on the mismatch.

**Variance.** `|e_y - q|` and `|e_y - p|` have the same order. The change
removes a bias without adding importance weights, so it adds no variance.

## 3. Implementation (`losses.py`)

`loss_fn_rl` computes, per chunk of positions, `baseline = sum_n coef_n * logp_n`
with detached coefficients and uses `w * (logp_y - baseline)` as the per-token
score (`w = 1` without a correction). `_centering` builds `coef` from whatever
the sampler recorded (`sampler_keys` now forwards all of these keys):

* `sampling_vocab_logprobs` [B,T,V] (`sampler.vocab_logprobs=full`): exact,
  `coef = q`.
* `sampler_topk_ids` / `sampler_topk_logprobs` [B,T,k] (`sampler.vocab_logprobs=k`):
  exact on the head `K`. The tail mass `Q_T = 1 - sum_K q_v` is assigned to the
  score of the tail event, `grad log P_T` with `P_T = 1 - sum_K p_v`. Since
  `grad log P_T = sum_{v not in K} (p_v/P_T) s_v`, this is exact whenever the
  sampler's tail is proportional to the trainer's. Otherwise the error is
  bounded by `Q_T` times the tail mismatch. If `P_T < 1e-4`, the tail term is
  dropped and the head is renormalized, to avoid cancellation in float32.
* Only `sampling_token_logprobs` (default): the same construction with
  `K = {y}`. The control variate is `q_y s_y + (1 - q_y) grad log(1 - p_y)`,
  which is exact on-policy and removes the part of the drift along the sampled
  token. It cannot see how `q` and `p` differ among the unsampled tokens, so
  it is only a partial fix. Use Configuration below.

**With an `ISCorrection`.** For token-level corrections the token term is
`w_y A (s_y - s_bar_m)`, with `s_bar_m = sum_v m_v s_v / sum_v m_v` and
`m_v = q_v w(p_v/q_v)`. This is the same `log_mass` the correction uses, now
evaluated for every recorded token. Then
`E_q[w_y A (s_y - s_bar_m)|h] = sum_v m_v (Q(h,v)-b)(s_v - s_bar_m)`, which is
again a covariance with no drift, for any clip, band or sign rule. On the
top-k path the tail keeps weight 1. Sequence-level corrections fold a scalar
into `A`, and (5) holds unchanged with `m = q`. With `correction=None`,
`m = q`.

The gradient was checked numerically. With full vocab or `k = V`,
`dL/dz = -(A/N)(e_y - q)` to 1e-8.

## 4. Evidence on the CPU dev setup (`dev/mini.py`, noise 0.3, 600 steps)

| run | last-10 train reward | final eval |
| --- | --- | --- |
| original loss, seed 0 | 0.58 (plateau, eval declining from 0.72) | 0.66 |
| new loss, no vocab info (fallback) | 0.63 | 0.76 |
| new loss, `vocab_logprobs=4` | 0.87 | 0.83 |
| new loss, `vocab_logprobs=full` | 0.86 | 0.85 |
| original loss, seed 2 | 0.55 | 0.70 |
| new loss, `vocab_logprobs=64`, seed 2 | 0.82 | 0.84 |
| new loss + TIS (`high=2 outside=clamp`), seed 2 | 0.60 (still rising) | 0.83 |
| 200 extra unlikely tokens, original loss, seed 1 | 0.09 (collapsed, entropy -> 0) | |
| 200 extra unlikely tokens, fallback | 0.17 | |
| 200 extra unlikely tokens, `vocab_logprobs=8` | 0.82 | |
| 200 extra unlikely tokens, `vocab_logprobs=full` | 0.84 | |

Combined with clipping (TIS, PPO) the new loss is stable and keeps
improving, but more slowly. The clipping only shrinks the step, so
`correction=None` is the recommended setting.

In the collapsing run of the original loss, the KL between sampler and trainer
falls to 0 within about 60 steps: the trainer sharpened until both policies
became deterministic, as predicted for `V(h) < b` in Section 1.

## Configuration

sampler.vocab_logprobs=64
