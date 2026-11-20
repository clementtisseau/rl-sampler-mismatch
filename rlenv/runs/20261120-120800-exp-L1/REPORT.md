# Sampler-centered policy gradient: removing the mismatch drift

## Summary

The REINFORCE loss uses the score `∇log p(y)` of the trainer `p` on tokens drawn
from a different sampler `q`. That score has zero mean under `p`, but not under
`q`. As a result, the part of the advantage that is constant within a state (the
"offset" that a baseline is supposed to make irrelevant) no longer has zero
expected effect. It produces an update with expectation `Ā(s)·(q − p)` in logit
space, which is `−Ā(s)·∇_θ KL(q‖p_θ)`. This term has a fixed sign and a fixed
direction. It does not average out across steps, and because the sampler is
rebuilt from the trainer every step, it keeps pushing in the same direction.

**Fix:** center the score under the distribution the token was actually drawn
from. Every per-token update has its expectation under `q` subtracted:

    loss = −(1/N) Σ_i A_i Σ_t [ w(y_t) log p(y_t|s_t) − Σ_v sg(q(v|s_t) w(v)) log p(v|s_t) ]

Here `q(v|s_t)` comes from the sampler's top-k logprobs, which are already in
the batch (`sampler_topk_ids`, `sampler_topk_logprobs`), and `w ≡ 1` when no
correction is used. Without a correction, the logit-space update becomes
`A·(e_y − q)` instead of `A·(e_y − p)`. A constant advantage then has exactly
zero expected effect under the sampling distribution. This is the default
behaviour of `loss_fn_rl`, and no flag is needed.

## 1. Setup and notation

Take one generated position with prefix (state) `s`. The trainer has logits `z`,
with `p = softmax(z)`. The sampler draws `y ~ q(·|s)`, where `q ≠ p` because of
weight noise, quantization, a quantized KV cache or staleness. Write `Q(s,y)`
for the expected return after emitting `y` and `b` for the baseline (the group
mean, or 0 for `plus_minus`). The advantage is `A = Q − b`. In logit space,
`∇_z log p(y) = e_y − p`. The parameter gradient is `J^T(·)` with
`J = ∂z/∂θ`, so it is enough to analyse logit-space updates.

The current loss is `−A·log p(y)`, with an optional detached weight `w`. Its
per-token update is `g = A·w(y)·(e_y − p)`.

## 2. What is wrong with the current loss

Take the expectation over the token actually sampled, `y ~ q`, with `w = 1`.
Let `Ā_q(s) = E_q[Q(s,·)] − b` be how much better state `s` is than the
baseline:

    E_{y~q}[ (Q(y) − b)(e_y − p) ]
      = Σ_y q(y)(Q(y) − E_qQ)(e_y − p)  +  Ā_q(s) Σ_y q(y)(e_y − p)
      = Cov_q(Q, e_y)                   +  Ā_q(s) · (q − p).            (1)

(In the first sum the `−p` part vanishes because `Σ q (Q − E_qQ) = 0`.)

* **On-policy (`q = p`)**, the second term is zero. The first term is the exact
  logit-space policy gradient `Σ p (Q − E_pQ) e_y`. This is why a baseline is
  "free" in ordinary REINFORCE: `E_p[∇log p] = 0`.
* **Off-policy**, the second term remains. Since `∇_z KL(q‖softmax z) = p − q`,

      Ā_q(s)·(q − p) = −Ā_q(s) · ∇_z KL(q ‖ p),                         (2)

  so the loss secretly contains `+Ā_q(s)·KL(q‖p_θ)` at every state. In states
  with `Ā_q < 0` it does gradient **ascent** on the trainer/sampler divergence:
  the trainer is pushed away from its own sampler. In states with `Ā_q > 0`
  it pulls the trainer toward the sampler's errors.

This is a bias, not noise, and it compounds:

* **The sign of `Ā_q` is systematic.** With `plus_minus` rewards and success
  below 50%, `Ā_q < 0` almost everywhere, so the whole batch ascends
  `KL(q‖p)`. With group centering, `Ā_q` averages to zero at the prompt, but
  not at later states. States reached through sampler errors are exactly the
  failing ones (`Ā_q < 0`), and at those states `q − p` is large. So the bias
  is concentrated where the mismatch is.
* **The direction is fixed.** For a weight perturbation `δ`, `q − p ≈ F_z·J δ`
  to first order, where `F_z = diag(p) − pp^T`. The bias in parameter space is
  then `≈ Ā·J^T F_z J δ`, a Fisher-weighted version of the same `δ` at every
  step. The sampler is refreshed as `θ + δ` (or `quant(θ)`, or `θ_{t−k}`), so
  the mismatch never disappears, and the drift adds up linearly over training
  while the useful signal saturates. The same holds for quantization and KV
  error, which are deterministic functions of `θ`.
* **The bias depends on an arbitrary quantity.** The bias scales with the
  advantage offset `b`, which on-policy theory says can be anything. A
  correct estimator must not depend on `b`. The current one does whenever
  `q ≠ p`.

### Why IS corrections and clipping only slow this down

With a token weight `w(y)`, the offset term in (1) becomes
`Ā·Σ_y q(y)w(y)(e_y − p) = Ā·(m − (Σ_v m_v)·p)` with `m = q⊙w`.

* For naive IS (`w = p/q`), `m = p` and the term is exactly zero. But the
  variance of `p/q` is unbounded. Also, `Q` is still estimated with `q`'s
  continuations, and the prefixes are still `q`'s.
* For TIS/MIS/PPO-type truncation or dropping, `m = min(p, Cq)`, or `p` on a
  band and 0 outside it. Then `m − (Σm)p ≠ 0`, so the offset bias returns in
  proportion to how much mass was clipped. The bias comes from exactly the
  tokens where `p/q` is far from 1, which is where the mismatch is. Stronger
  clipping means `m` is closer to `q` (or has more dropped mass), so more of
  the original bias `Ā(q − p)` comes back. A one-sided clip (one advantage sign
  only) also breaks the symmetry between positive and negative advantages and
  can make the drift worse.

The underlying problem is that the score is centered under the wrong
distribution.

## 3. The fix: center the score under the sampler

Any quantity whose expectation given the state is zero can be subtracted from
a per-token gradient without changing what it estimates. Subtract the
conditional expectation of the update under the distribution the token was
sampled from:

    g'(y) = A · [ w(y)∇log p(y) − Σ_v q(v) w(v) ∇log p(v) ].            (3)

**Property 1: offsets have no effect, for any `q` and any `w`.** For a
constant advantage `c`, `E_{y~q}[g'] = c·(Σ q w ∇log p − Σ q w ∇log p) = 0`
exactly. More generally `E_q[g'] = Cov_q(Q, w∇log p)`, which does not depend
on `b`. The drift term in (1)–(2) is gone, and so is the hidden
`Ā·KL(q‖p)` objective.

**Property 2: what is left is the right signal.** With `w = 1`, in logit space

    g' = A·(e_y − q),   E_q[g'] = Cov_q(Q, e_y) = Σ_y q(y)(Q(y) − E_qQ) e_y,

which is exactly `∂/∂z' E_{y~softmax(z')}[Q]` at `z' = log q`. That is the true
policy gradient of the policy that actually generated the data. The trainer
and the sampler share parameters (the sampler is a deterministic perturbation
of the trainer, refreshed every step), so this is the ascent direction for
the deployed policy. It is pulled back through the trainer's Jacobian instead
of the sampler's. That residual is a second-order discrepancy
(`J(θ) − J(θ+δ)`). Unlike (2), its sign does not track `Ā` and its direction
is not fixed. The fixed points also become the right ones. The old estimator
is stationary when `Cov_q(Q,e) + Ā(q − p) = 0`, which can happen at a
non-optimal policy, or never, which gives a perpetual drift. The new one is
stationary only when `Cov_q(Q, e_y) = 0`, i.e. when the sampler puts mass only
on maximal-`Q` tokens, the same fixed points as on-policy REINFORCE.

**Property 3: it reduces to known estimators where they are already correct.**

* On-policy (`q = p`, `w = 1`): the subtracted term is
  `Σ_v p ∇log p = ∇Σ p = 0` exactly, so this is plain REINFORCE.
* Untruncated naive IS (`w = p/q`): the subtracted term is again
  `Σ_v p ∇log p = 0`, so this is plain IS.
* Truncated or dropped IS (TIS, MIS, PPO, DPPO with `sign`): the subtracted
  term `Σ_v q w ∇log p` cancels exactly the leftover offset term
  `Ā(m − (Σm)p)` from Section 2. `w(v)` for the non-sampled top-k tokens is
  computed with the same `ISCorrection` rule as for the sampled token, using
  the same advantage sign. Sequence-level weights scale `A_i` and are combined
  with the `w = 1` centering.

**As a loss.** Equation (3) is the gradient of

    L = −(1/N) Σ_i A_i Σ_t [ w(y_t) log p(y_t|s_t) − Σ_v sg(q(v|s_t) w(v)) log p(v|s_t) ].

For `w = 1` this equals `−(1/N) Σ A [log p(y) + CE(q, p)]`. So the fix adds
`−A·CE(q,p)` per token, which exactly cancels the hidden `+Ā·KL(q‖p)` term
from (2) in expectation. No weight is ever larger than `q(v)·w(v)`, so no new
variance comes from ratios, and no hyperparameter is introduced.

### Only top-k of `q` is known

The sampler returns its top-k logprobs (k = 20, as inference engines do).
`loss_fn_rl` uses `Σ_{v∈K} q(v) w(v) log p(v)` without renormalising. The
missing tail mass `1 − m`, with `m = Σ_K q`, is then effectively spread
according to `p`, and that is free because `Σ_v p ∇log p = 0`. With `w = 1`,
the logit update is `A·(e_y − q_K − (1 − m)p)`, and the leftover offset bias is

    c · (q_tail − (1 − m) p),   with ‖·‖_1 ≤ 2(1 − m),

instead of `c·(q − p)`. For an LLM, the top-20 tokens carry almost all of
`q`'s mass. In `dev/mini.py`, the top-20 covers the whole vocabulary unless
`MINI_EXTRA_TOKENS` is set, so there the centering is exact. Entries with
non-finite logprobs are given zero mass. If the masses sum to more than 1 due
to rounding, they are rescaled to at most unit mass. Positions outside the
completion mask, including stale top-k rows after early stopping, are masked
as before. If a batch has no top-k keys, the loss falls back to the original
score.

## 4. Implementation (`losses.py`)

Inside the existing chunked, rematerialised loop of `loss_fn_rl`:

* `logp_k = take_vocab(logp_vocab, sampler_topk_ids)` uses the same TP-safe
  gather as the sampled token. `q_k = exp(sampler_topk_logprobs)` is detached.
* `w_k = is_weight(logp_k − logq_k, positive, token_c, |q_k − p_k|)` when a
  token-level correction is given, and `1` otherwise.
* `score = w·log p(y) − Σ_K q_k w_k logp_k`, and the loss is
  `−Σ A_i Σ_t score / N` as before. The signature, return values, metrics and
  all `ISCorrection` options are unchanged, and the centering is active with
  the default arguments (`correction=None`).

## Settings

No configuration change is needed. The method works with the default
`rl.is.level=null` (no IS correction), and it also composes correctly with any
`rl.is.*` correction (Section 3, Property 3).
