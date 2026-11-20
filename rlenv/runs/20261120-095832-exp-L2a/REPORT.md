# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## 1. Setting and notation

A rollout y = (y_1, ..., y_T) is sampled from the sampler q. The trainer is p = p_θ.
At position t, h_t is the prefix, p_t = p_θ(· | h_t) and q_t = q(· | h_t) are the
next-token distributions over the vocabulary, z_t are the trainer logits
(p_t = softmax z_t), and J_t = ∂z_t/∂θ. All sequences share one advantage A
(group-centered or ±1). The current loss is

    L(θ) = -(1/N) Σ_i A_i Σ_t log p_θ(y_{i,t} | h_{i,t}),

so its update direction is g = Σ_i A_i Σ_t ∇ log p(y_{i,t} | h_{i,t}). With respect to
the logits, the score of a token is

    ∇_z log p(y | h) = e_y − p          (e_y is the one-hot vector of y).

## 2. The expected update under the sampler

Fix one position with prefix h. Define Q(h, y) = E[A | h, y_t = y] (the advantage averaged
over everything sampled after y) and V(h) = E_{y~q}[Q(h, y)], the value of the prefix
relative to the baseline. Split Q = (Q − V) + V:

    E_{y~q}[ A ∇log p(y|h) ] = E_{y~q}[ (Q − V) ∇log p(y|h) ]  +  V(h) · E_{y~q}[ ∇log p(y|h) ].
                               \_________ signal _________/     \_________ drift __________/

The drift term involves the **expected score under q**:

    E_{y~q}[∇_θ log p_θ(y|h)] = Σ_y q(y) ∇_θ log p_θ(y)
                             = −∇_θ KL(q_h ‖ p_θ,h)        (q held fixed)
                             = J^T (q − p)                 (in logit space).

**On-policy (q = p)** it vanishes identically, Σ_y p ∇log p = ∇ Σ_y p = ∇1 = 0. This is the
fact that makes baselines and centered rewards unbiased in REINFORCE. Every token whose
choice does not affect the reward (Q(h, ·) ≡ V(h), which is most tokens of a long
completion) then contributes zero in expectation. Only the signal term remains, and in
logit space it equals p ⊙ (Q − V), the exact policy gradient.

**Mismatched (q ≠ p)** the term no longer vanishes, and the expected update of the
current loss is

    E[g] = Σ_t E_h[ J^T q_h ⊙ (Q − V) ]  −  ∇_θ Σ_t E_h[ V(h) · KL(q_h ‖ p_θ,h) ],

with V treated as a constant. So besides policy improvement, the loss silently minimizes
the **advantage-weighted KL** Σ_h V(h) KL(q_h ‖ p_h):

* in prefixes with V(h) < 0 (failing trajectories; with ±1 rewards this is most of them
  early in training, and with group-centering it is every token after the point where a
  failing completion diverges from its group, because then V(h_t) → A_i < 0), the update
  **ascends** KL(q ‖ p) and pushes p away from the sampler. This objective is unbounded:
  the logits are driven off in the direction p − q;
* in prefixes with V(h) > 0 it pulls p toward q, toward the sampler's *error*.

This is a bias, not noise. Its direction J^T(q − p) is set by the fixed perturbation
(weight noise, quantization grid, KV quantization, staleness), so it has the same sign
at every step and does not average out. It acts on **every** token of every sequence,
while the signal is concentrated on the few decisive tokens, so for long completions it
can dominate the gradient. It also compounds. The sampler is refreshed from the trainer
(q = f(θ) with a systematic error), so after p moves by −η∇(V·KL), the next sampler
reproduces its error around the new p and the same drift applies again. Nothing in the
objective pays for the accumulated displacement: the drift has no fixed point except
q = p, which the mismatch prevents. The signal term, by contrast, goes to zero as the
policy converges (Q − V → 0). Close to convergence the drift is all that remains, which
is why runs collapse after reaching their peak instead of only learning more slowly.

### Why IS corrections and clipping only slow it down

With a detached token weight w = w(p/q), the drift term becomes
V(h) Σ_y q w(p/q) ∇log p. Exact IS (w = p/q) gives V Σ_y p ∇log p = 0, but its variance
is unbounded. Any truncation leaves a residual drift:

* clamping, w = min(r, C): Σ_y min(p, Cq) ∇log p = −Σ_y (p − Cq)_+ ∇log p ≠ 0;
* dropping outside a band B (MIS, PPO/DAPO/DPPO trust regions): Σ_{y∈B} p ∇log p =
  −Σ_{y∉B} p ∇log p ≠ 0.

The residual drift sits exactly on the tokens where p and q differ most. Its size grows
with the mass outside the band, so a **tighter** token-level band produces a **larger**
systematic term, V(h)·(−Σ_{y∉B} p ∇log p). That explains why stronger token clipping can
make things worse. Sequence-level weights multiply A, i.e. they rescale V(h), but they do
not center the per-token score. They change the size of the drift, not its existence.

## 3. The fix: center the score under the sampler

The baseline must be subtracted from the **score**, under the distribution that actually
produced the token, not only from the reward. Replace each token's score by its
q-centered version:

    s̃_t = ∇log p(y_t | h_t) − E_{v~q_t}[∇log p(v | h_t)],

i.e. the per-token loss term becomes

    ℓ_t = −A · ( log p(y_t) − Σ_v sg(q_t(v)) log p(v) ).

Properties:

1. **Zero mean under q for any V.** E_{y~q}[s̃] = 0 by construction, so
   E_{y~q}[A s̃] = E_{y~q}[(Q − V) s̃] = Cov_q(Q, ∇log p): the drift term is removed
   exactly, whatever the sign or size of V(h) and whatever the mismatch.
2. **Unchanged on-policy.** If q = p, the subtracted term has gradient
   Σ_v p_v ∇log p_v = 0, so the update is exactly REINFORCE. (Checked numerically: with
   q = p the gradient matches the original loss to 3e-7 relative error.)
3. **Logit-space form.** ∇_z ℓ_t = −A (e_y − q) instead of −A (e_y − p). Its expectation is
   q ⊙ (Q − V), which is the exact policy-gradient step for the logits of the
   distribution that sampled the data. Since the sampler is a fixed perturbation of the
   trainer, a step on p's logits moves q's logits by the same amount to first order.
   The update therefore improves the sampler as an on-policy step would. Its stationary
   points are those of on-policy RL (Q = V on the support), with no leftover term that
   pushes p relative to q. Any remaining error is multiplicative (J at θ instead of at
   the sampler weights, the state distribution of q instead of p). It vanishes when the
   signal vanishes, so it cannot accumulate the way the drift did.
4. **Equivalent reading.** Σ_v q log p = −H(q) − KL(q‖p), so the new term adds
   +A · KL(q_t ‖ p_t) to the maximized objective (up to the constant H(q)). This is the
   advantage-weighted KL that cancels the implicit −V · KL of Section 2. It is not an
   extra regularizer with a tunable coefficient. Its coefficient is fixed to the
   sequence's own advantage, which is what makes the cancellation exact in expectation.
5. **Bounded, low variance.** The coefficients e_y − q are bounded. There are no ratios.

### With an IS correction (optional)

For a token-level weight w, the score w(y)∇log p(y) is centered the same way:
s̃ = w(y)∇log p(y) − Σ_v q_v w(p_v/q_v) ∇log p_v. This is again zero-mean under q, and it
removes the residual drift of truncated, dropped or clipped weights described above.
Sequence-level corrections keep rescaling A and use the centered token score.

## 4. Estimating Σ_v q_v log p_v from what the sampler returns

The sampler returns the logprob of the sampled token and its top-k (k = 20) logprobs per
position (`sampler_topk_ids`, `sampler_topk_logprobs`, as inference engines provide).
I build a detached q̃:

* q̃_v = q_v on the top-k ids K;
* the remaining mass m = 1 − Σ_{k∈K} q_k is spread over the other tokens in proportion to
  p: q̃_v = m · p_v / p_tail, where p_tail = 1 − Σ_{k∈K} p_k.

Then Σ_v q̃_v log p_v = Σ_{k∈K} q_k log p_k + (m/p_tail)(Σ_v sg(p_v) log p_v − Σ_{k∈K} sg(p_k) log p_k).
The full-vocabulary sum has zero gradient, so it is kept only as a stop-gradient value.
This needs one gather of k entries per position and no vocabulary scatter, so it works
with the TP-sharded vocabulary and the chunked log-softmax. Its properties:

* exact when q = p, so the on-policy gradient is unchanged;
* the residual drift is V · J^T (q − q̃), which is confined to the tail outside the
  sampler's top-20 and bounded by its mass m (typically < 1%);
* q̃ does not depend on the sampled token, so E_{y~q}[e_y − q̃] = q − q̃ exactly. Using
  q(y) of the sampled token inside q̃ would break this.

If the top-k logprobs are absent from the batch, a fallback uses the sampled token alone:
q̃ = q(y) on y and 1 − q(y) spread over the other tokens in proportion to p. This is still
exact on-policy, but because q̃ then depends on y it removes the drift only partially. Its
expectation is Σ_y q_y (1 − q_y)/(1 − p_y) (e_y − p) instead of q − p. `train_rl.py` always
provides the top-k logprobs, so the main path is the one that applies.

Implementation: `sampler_expected_logp` and `loss_fn_rl` in `losses.py`. The change is
active with default arguments (`correction=None`). The metric `q_known` reports the
sampler mass covered by the top-k.

## 5. Sanity check (CPU dev setup, `dev/mini.py`, noise 0.3, 600 steps, seed 0)

| loss | eval reward at step 0 → 600 | last-10 train reward |
| --- | --- | --- |
| original (PG) | 0.56 → plateaus at ≈ 0.70, falls to 0.66 | 0.58 |
| sampler-centered | 0.56 → 0.84 (rising monotonically) | 0.83 |

With 30 extra never-correct tokens (`MINI_EXTRA_TOKENS=30`, so the top-20 does not cover
the vocabulary), seed 1, 400 steps: the original loss reaches an eval reward of 0.47 → 0.69
(last-10 train 0.61). The sampler-centered loss reaches 0.48 → 0.82 (last-10 train 0.82).

## Configuration

No configuration change is needed. The fix is active with `correction=None`.
