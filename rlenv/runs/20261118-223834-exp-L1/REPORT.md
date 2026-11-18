# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## Summary

The REINFORCE loss uses the trainer's score `∇ log p(y)`. On-policy, that score has
zero mean, so the mean of the advantage drops out of the update. When the tokens come
from a different sampler `q`, the score's mean is `q − p` (at the logits), which is
not zero. Whatever conditional mean the advantage has then turns into a
deterministic drift `Ā·(q − p)`. The drift does not depend on which tokens were
good. It points along the sampler's own error, and it comes back at every step
because the sampler is rebuilt from the trainer each step. This is the systematic,
compounding bias.

The fix is to center the score under the distribution that generated the tokens:
use `e_y − q` instead of `e_y − p` at the logits. The new loss is

    loss = − Σ_tokens A · ( log p(y_t) − Σ_v sg(q̂_v) log p_v ) / n_tokens

where `q̂` is the sampler's next-token distribution, either exact or reconstructed
from its top-k logprobs. The update is then a pure covariance between advantage
and action under `q`. It has no drift term, and to first order in the mismatch it
equals the policy gradient of the sampler's own reward. `losses.py` always applies
this, including with `correction=None`. When an IS correction is passed, it is
folded into the centering measure, so the two can be combined.

## 1. Notation

At one position (state `s`), the trainer has logits `z`, `p = softmax(z)`. The
sampler has `q`. `y ~ q` is the sampled token and `e_y` is its one-hot vector. The
advantage is `A` (per sequence: group-centered or ±1). `J = ∂z/∂θ` is the trainer's
logit Jacobian. Note that `∇_z log p_y = e_y − p`, so in parameter space the
update is `J^T(e_y − p)`. The trainer's REINFORCE gradient per state is

    g_naive(s) = E_{y~q}[ A · (e_y − p) ].

## 2. What is wrong with the current loss

Write `Ā(s) = E_q[A | s]` and split `A = Ā + (A − Ā)`:

    g_naive = E_q[(A − Ā) e_y]  +  Ā · E_q[e_y − p]
            = Cov_q(A, e_y)      +  Ā(s) · (q − p).                        (1)

- **On-policy (`q = p`)** the second term is zero: `E_p[e_y − p] = 0`. This is the
  only reason a REINFORCE update can ignore the mean of the advantage, i.e. why
  baselines and reward shifts do not change it in expectation.
- **Off-policy** the term `Ā(s)(q − p)` remains. It is a bias, not noise: its
  expectation is non-zero and it carries no information about which actions were
  good.

**Group centering does not remove it.** Group centering makes `A` average to zero
over a group at the prompt. At a later token state `s_t`, `Ā(s_t) = E_q[R | s_t] −
b(prompt)`, which is the sampler's value minus a prompt baseline. That is positive
on prefixes that are going well and negative on prefixes that are failing, and it
is generally non-zero. With `plus_minus` rewards `Ā` is large everywhere.

**The drift is a hidden cross-entropy objective.** `q − p = ∇_z Σ_v q_v log p_v`,
so per state

    g_naive = Cov_q(A, e_y) + Ā(s) · ∇_z [ −CE(q, p) ].                    (2)

The loss is therefore the intended policy gradient plus an extra objective weighted
by `Ā`:
- Where `Ā > 0`, the trainer does maximum likelihood towards the sampler's
  distribution. It takes on the sampler's errors, such as mass on tail tokens that
  noise or quantization inflated.
- Where `Ā < 0`, it does maximum *un*likelihood of the sampler's distribution and
  pushes `p` away from `q`. This term has no bounded optimum: `−Σ q log p` can be
  made arbitrarily large.

**Why it compounds.** The sampler is `q = f(θ + δ)` (noise), `Q(θ)` (int8/fp8
weights, quantized KV cache), or `f(θ_{t−k})` (staleness), and it is recomputed
from θ every step. So the gap `q − p` is a function of θ that does not shrink as θ
moves.
- Under the drift in (2), `p` is moved towards `q` (or away from it). The new
  sampler applies the same perturbation again on top, and the next step adds the
  same drift again.
- The bias therefore accumulates linearly over steps rather than averaging out
  like `√steps` noise. It can feed back on itself, for example:
  tail mass up → `q`'s inflated tail up → larger `q − p` on the tail → more drift.
- There is no stationary point: the expected update is zero only if
  `Cov_q(A, e_y) = −Ā(q − p)`, which is not an optimum of anything.

On the CPU model with 20 extra junk tokens, plain PG collapses to reward 0 within
about 50 steps, and the sampler's tail mass is what drives it.

**Why IS and clipping only slow it down.** A detached per-token weight `w(r)`,
`r = p/q`, gives

    E_q[w (e_y − p)] = m − (Σ_v m_v) p,      m_v = q_v w(p_v/q_v).         (3)

- **Exact IS** (`w = r`) gives `m = p`, so the term vanishes. But `r` is unbounded,
  and the variance is large exactly where the mismatch is large.
- **Clipping or dropping** (TIS, MIS, PPO, DPPO, ...) sets `m_v ≠ p_v` on every
  token where the bound is active. So the drift `Ā(m/Σm − p)Σm` comes back exactly
  on the most mismatched tokens, still with the sampler's sign.
- **Tighter bands** make `m` look more like `q` (clamp) or zero out signal while
  leaving the centering wrong (drop). This is why stronger token-level clipping can
  make things worse.
- **Sequence-level weights** rescale `A` but leave `e_y − p` unchanged, so (1)
  still applies with a reweighted `Ā`.

## 3. The modification: center the score under the sampler

Replace the logit-space score `e_y − p` by `e_y − q`:

    g_new(s) = E_{y~q}[ A · (e_y − q) ] = Cov_q(A, e_y).                   (4)

`E_q[e_y − q] = 0` holds exactly, so the update no longer depends on `Ā(s)` or on
any per-state baseline. The drift term in (1) and (2) is gone, whatever the
advantage's conditional mean is.

**Loss that produces it.** Because `∇_z Σ_v q_v log p_v = q − p`,

    ∇_z [ log p_y − Σ_v sg(q_v) log p_v ] = (e_y − p) − (q − p) = e_y − q,

so the per-token loss is `−A · (log p_y − Σ_v sg(q_v) log p_v)`. The coefficients
`q_v ∈ [0, 1]` are bounded, so no weight can blow up: `|e_y − q|_1 ≤ 2`, the same
as plain REINFORCE.

**What objective it follows.** `E_q[A (e_y − q)]` is exactly the policy gradient of
the sampler's expected return with respect to the **sampler's** logits,
`∇_{z^q} E_q[A]`. The loss maps this gradient back through the trainer's Jacobian.
The exact gradient of the sampler's reward with respect to θ is `J_q^T Cov_q(A, e_y)`.
Since `z^q = z(θ + δ)` (or its quantized version), `J_q = J + O(δ)`, and

    θ-update error of new loss   = (J − J_q)^T Cov_q(A, e_y)      = O(δ) · signal,
    θ-update error of naive loss = J^T Ā (q − p) + (J − J_q)^T Cov  = O(δ) · Ā   (first order, signal-free).

The remaining error is second order: mismatch times the reward signal. It is zero
when the reward says nothing about the action. It cannot create drift by itself,
and the update is zero exactly when the sampler is locally optimal
(`Cov_q(A, e_y) = 0`). Climbing the sampler's reward is the right goal here. The
sampler is the trainer's weights plus a fixed map, so the two policies improve
together. The rollouts that define the data and the reward come from the sampler.

### 3.1 Only partial knowledge of q (what `losses.py` does)

**No exact fix from the sampled-token logprob alone.** Suppose only `p_y` and `q_y`
are known and the score is `∇ f(p_y, q_y)`. For it to be centered for every `p`,
`Σ_y q_y ∂_p f(p_y, q_y) ∇p_y = 0` must hold for all tangent directions
(`Σ ∇p_y = 0`). That forces `q_y ∂_p f = const`, i.e. `f ∝ p_y/q_y`, which is plain
IS with its variance. So the fix needs more of `q`. The sampler can already return
its top-k logprobs per position (`sampler.vocab_logprobs=k`) or all of them
(`=full`). `loss_fn_rl` uses whichever is present in the batch.

**Binned reconstruction.** Let `K'` be the known tokens: the top-k ids plus the
sampled token. Let `Q_rest = 1 − Σ_{K'} q` and `P_rest = 1 − Σ_{K'} p`. Define `q̂`
as `q` on `K'`, with the remaining sampler mass spread over the other tokens in
proportion to `p`: `q̂_v = c p_v`, `c = Q_rest / P_rest`. The surrogate

    C(z) = Σ_{k∈K'} sg(q_k − c p_k) log p_k

has gradient

    Σ_k (q_k − c p_k)(e_k − p)
      = Σ_{K'} q_k e_k − c Σ_{K'} p_k e_k − (Q_{K'} − c P_{K'}) p
      = Σ_{K'} q_k e_k + c (p − Σ_{K'} p_k e_k) − p
      = q̂ − p,

using `Q_{K'} − c P_{K'} = 1 − Q_rest − c + c P_rest = 1 − c`. So
`∇_z(log p_y − C) = e_y − q̂`. The only bias left is `Ā (q − q̂)`. It lives on the
unseen tail, sums to zero there, and only moves mass within the tail. The total
tail mass, and every token in the sampler's head, are centered exactly.
- With the full vocabulary, `q̂ = q` exactly.
- With only the sampled token (`K' = {y}`), the surrogate reduces to a detached
  weight `(1 − q_y)/(1 − p_y)` on `log p_y`. This is exact for a 2-token vocabulary
  and partial otherwise. It is the fallback when no head is recorded.

**With an IS correction.** If a token-level correction weight `w(p/q)` is passed,
the expectation (3) is taken under the measure `m_v = q_v w(p_v/q_v)` (the existing
`log_mass`). The score is centered under its normalization:

    per-token term = w_y · (log p_y − Σ_v sg(m̂_v) log p_v),    m̂ = m / Σ m,

so `E_q[w (e_y − m̂)] = m − m̂ Σ m = 0`. With top-k, the same binning is applied to
`m`, with the rest bin weighted at ratio `P_rest/Q_rest`. Without a correction
`m = q`. Sequence-level corrections only rescale `A` and need nothing extra.

On the toy check (8 tokens, strong mismatch), `‖E_q[score]‖_1` is about 1e-8 with
the full vocabulary for PG, TIS and PPO. With top-3 the residual is a few
hundredths, and with the sampled token only it is about 0.1.

**Numerics.** All coefficients are detached and bounded, and they are zeroed off
the completion mask and where they are not finite, because the sampler's padding
rows can hold stale values. `P_rest` is floored at 1e-6, which bounds the tail
gradient `c Σ_rest p_v ≤ Q_rest`. Only the head `[B, T, k]` is gathered from the
chunked full-vocabulary log-softmax that `loss_fn_rl` already computes. The memory
is negligible next to a `[B, T, V]` `full` record.

## 4. Sanity check on the CPU model (`dev/mini.py`, noise 0.3, 600 steps; single runs, seeds 0-3)

| setting | loss | train reward, last sixth | eval reward, end |
| --- | --- | --- | --- |
| 0 extra tokens | original PG | 0.58 (stalls) | 0.66 |
| 0 extra tokens | centered, sampled token only | 0.64 | 0.75 |
| 0 extra tokens | centered, top-3 / top-20 / full | 0.85 / 0.84 / 0.86 | 0.85 / 0.84 / 0.83 |
| 20 extra tokens | original PG | 0.00 (collapse by step ~50) | 0.00 |
| 20 extra tokens | centered, sampled token only | 0.11 (slow decline) | 0.00 |
| 20 extra tokens | centered, top-1 / top-3 / top-20 / full | 0.64 / 0.84 / 0.83 / 0.83 | 0.79 / 0.89 / 0.79 / 0.84 |
| 200 extra tokens | centered, top-3 | 0.80 | 0.79 |
| 0 extra tokens, noise 0.5 | original PG / centered top-3 | 0.00 / 0.51 (rising) | 0.00 / 0.70 |

Centering with the sampler's head stops collapse and keeps improving in every
setting tried. The sampled-token-only fallback helps but does not fix the problem,
as Section 3.1 predicts. That is why the configuration below records the sampler's
top-k head. A top-20 head is cheap even with a 151k vocabulary
(`[B, T, 20]`, recorded exactly by `models/topk.py`).

## Configuration
sampler.vocab_logprobs=20
