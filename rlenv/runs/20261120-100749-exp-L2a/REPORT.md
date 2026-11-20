# Sampler-centered REINFORCE: removing the mismatch drift

## 1. Setup and notation

A rollout y = (y_1, ..., y_T) is drawn from the sampler q. The trainer is p = p_θ.
At position t the state is s_t, the trainer's logits are z_t, and
p_t = softmax(z_t). For a softmax, the score with respect to the logits is

    ∇_z log p(v | s) = e_v − p(·|s).                                          (1)

The current loss (with `correction=None`) is

    L(θ) = −(1/N) Σ_i A_i Σ_t log p_θ(y_{i,t} | s_{i,t}),   y_i ~ q,

where A_i is a sequence-level advantage (±1 for `plus_minus`, or the
group-centered reward).

## 2. What goes wrong: the advantage multiplies a score that is not zero-mean under q

Fix a state s, and let a be the part of the advantage that does not depend on
which token is chosen at s. That covers a reward offset, the group's mean
reward, and the effect of every other token. In a long sequence it is almost
all of A for almost every token. Its expected contribution to the update at s is

    a · E_{v~q}[∇_z log p(v)] = a · Σ_v q(v)(e_v − p) = a · (q − p).          (2)

**On policy (q = p)** this is exactly 0. This is the identity E_p[∇log p] = 0,
which is what makes baselines legitimate: only the covariance between A and the
chosen token moves the policy, and the expected update is
Σ_v p(v)(A(v) − Ā_p) e_v, the true policy gradient.

**Off policy (q ≠ p)** the term is a(q − p) ≠ 0. In parameter space (q fixed),
Σ_v q ∇_θ log p_θ(v) = −∇_θ CE(q, p_θ) = −∇_θ KL(q ‖ p_θ), so

    E_q[g] = (reward signal) − ā · ∇_θ KL(q ‖ p_θ)      (ascent direction),  (3)

with ā the token-weighted, reward-independent part of the advantage:

* ā > 0: the trainer is distilled toward the corrupted sampler.
* ā < 0 (failed rollouts, e.g. `plus_minus` at low reward): the trainer is
  anti-distilled away from q. Tokens that q over-samples relative to p are
  pushed down and tokens it under-samples are pushed up. The push applies to
  the whole distribution, including the long tail of useless tokens.

Group centering makes Σ_i A_i = 0 per group, but (2) is per token. The
coefficient Σ_i A_i Σ_t (q − p)_{s_{i,t}} does not cancel, because failed and
successful rollouts visit different states, have different lengths and
different mismatch, and token normalisation weights them unequally. Centering
shrinks the term without removing it.

**Why it is a bias and why it compounds.** q is a fixed function of p:
q = F(p), with the same relative weight perturbation, the same quantizer, or
the same lag. So q − p ≈ J_F δ points in a consistent direction at every step,
and (3) is a systematic drift, not zero-mean noise. The sampler is rebuilt
from the trainer every step, so there is no fixed point: after the update, the
new q = F(p_{k+1}) is again offset by the same perturbation, and the trainer
moves along it again. For relative weight noise θ_q = θ ⊙ (1 + σξ), the drift
to first order is

    θ_{k+1} ≈ θ_k ± η ā σ (ξ ⊙ θ_k),

which is multiplicative, i.e. exponential per coordinate. When reward drops,
ā becomes more negative and the anti-distillation speeds up. That positive
feedback is the collapse.

## 3. Why IS corrections and clipping only slow it down

A detached token weight w(r), r = p/q, turns (2) into

    a · Σ_v q(v) w(p(v)/q(v)) (e_v − p) = a · (m − (Σ_v m(v)) p),
    m(v) := q(v) w(p(v)/q(v))          (`log_mass` in losses.py).            (4)

This vanishes for every a only when m ∝ p, which requires w(r) = r (untruncated
token IS).

* TIS gives m = min(p, 2q).
* MIS, PPO, DAPO and DPPO give m = p·1[band] (or clamps).

The drop and clamp rules act exactly on the most mismatched tokens. So m − (Σm)p
is smaller but still systematic, and tighter bands move m further from p, which
can make the drift worse. Sequence-level weights rescale A but leave the
per-token score biased. Untruncated token IS cancels the drift only in
expectation, through heavy-tailed random weights (r up to e^20). That high
variance is exactly why people clip.

## 4. The fix: center every token's score under the sampler

Subtract from each token's score its own expectation under the sampling
distribution. This is a state-dependent control variate, computed analytically
rather than sampled:

    ŝ_t = w_t ∇log p(y_t|s_t) − Σ_v m_t(v) ∇log p(v|s_t),
    m_t(v) = q(v|s_t) w(p(v|s_t)/q(v|s_t))      (m = q with no correction).  (5)

The surrogate loss is

    L = −(1/N) Σ_i A_i Σ_t [ w_t log p(y_{i,t}) − Σ_v sg(m_t(v)) log p(v|s_{i,t}) ].

With no correction, the per-token term is A·[log p(y_t) + CE(sg(q_t), p_t)].

**P1 (no drift).** Conditional on any state, E_{y_t~q}[ŝ_t] = 0, by (4). So any
part of the advantage that does not depend on the token choice contributes
exactly zero, for any state distribution, any mismatch type and any
correction w. The term −ā∇KL(q‖p) in (3) is cancelled identically.

**P2 (what remains is the reward signal).** With no correction, in logit space,

    E_q[A ŝ_t | s_t] = Σ_v q(v)(e_v − p) A(v) − (Σ_v q(v)A(v))(q − p)
                     = Σ_v q(v)(A(v) − Ā_q) e_v,

compared with Σ_v p(v)(A(v) − Ā_p) e_v on policy. The corrected update is
nonzero only when reward depends on the choice. Its error relative to the
on-policy gradient is O(‖q − p‖·|A − Ā|): a first-order error in an update that
still points uphill. It is not a reward-independent drift that keeps running
when there is no signal. Uncorrected REINFORCE instead carries the extra term
ā(q − p), which depends on the level of the reward rather than on its variation.

**P3 (exact REINFORCE on policy).** If q = p and w ≡ 1, then
Σ_v p ∇log p = ∇Σ_v p = 0. The modification has zero gradient, and the loss is
exactly vanilla REINFORCE. It costs nothing without mismatch and grows with it.

**P4 (low variance).** The subtracted term is deterministic given the state.
It is an exact expectation, not a sample, so unlike untruncated IS it adds no
heavy-tailed weights.

## 5. Computing the expectation with what the sampler returns

The sampler returns its top-k logprobs per position (k = 20, as inference
engines do): `sampler_topk_ids`, `sampler_topk_logprobs`. On the top-k set K
the sum in (5) is exact. The leftover sampler mass q_rest = 1 − Σ_K q is spread
over the remaining tokens in proportion to p (exact if the two tails have the
same shape):

    Σ_{v∉K} q(v) ∇log p(v) ≈ q_rest ∇log p_rest,   p_rest = 1 − Σ_{v∈K} p(v).

The cost is a gather of k logits per position, inside the existing
rematerialized chunks. With a correction, m is evaluated with the same
`log_mass` the correction uses, on K and on the pooled tail. This keeps (5) the
exact centering for TIS, MIS, PPO, DPPO and the others.

Fallbacks:

* No top-k in the batch and no correction: the loss uses the other exact
  solution of m ∝ p, untruncated token IS (w = p/q, m = p, Σ_v p ∇log p = 0).
* No top-k but a correction is given: K is the sampled token, and the rest is
  pooled.

## 6. Implementation (losses.py)

* `sampler_centering` returns Σ_{v∈K} sg(m(v)) log p(v) + sg(m_rest) log p_rest.
* `loss_fn_rl` uses `score = w·log p(y_t) − cv` per token. It is always on with
  default arguments, and the signature and metrics are unchanged.
  Sequence-level corrections still rescale A, with the centering applied on top.

Sanity checks:

* On policy, the gradient equals the original loss's to 7e-9.
* CPU replica (noise 0.3, 600 steps, `plus_minus`, 100 extra tail tokens):

  | Loss | Reward |
  | --- | --- |
  | Original PG | collapses to reward 0 (2 of 2 seeds) |
  | New loss | eval reward ≈ 0.79–0.80 (3 seeds) |

* Default dev setting: last-10 train reward 0.85 vs 0.58 for the baseline.

## Configuration

No configuration changes are needed. The centering is active with default
arguments and `rl.is.level=null`.
