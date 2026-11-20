# A sampler-centered score for REINFORCE under sampler/trainer mismatch

## Summary

The REINFORCE loss uses the score `grad log p(y)` of the *trainer* p on
tokens drawn from the *sampler* q. Under q this score does not have zero mean.
Its mean is `-grad KL(q_t || p_t)` at every context. The loss multiplies that
mean by the advantage, so the update gets a deterministic drift: it pulls p
toward the sampler on good trajectories and pushes it **away from the sampler on
bad ones**. On-policy the drift is exactly zero, so nothing in on-policy theory
accounts for it. It does not average out over batches. It grows with the
mismatch, and because the sampler is rebuilt from the trainer every step, it
feeds on itself.

The fix centers each token's score by its exact conditional expectation under
the sampler:

    loss = -(1/N) sum_i A_i sum_t [ w_t log p(y_t) - sum_v m_{t,v} log p_v ],
    m_{t,v} = sg( q_t(v) * w(p_t(v)/q_t(v)) )

(w = 1, m = q when there is no correction.) The new term is a per-token
control variate: on-policy its gradient is identically zero, and off-policy it
cancels the drift exactly. Implemented in `losses.py` (`score_baseline`). It is
active by default, needs no configuration and also works with any `rl.is.*`
correction.

## 1. The expected update of the current loss

Notation: context (prefix) c_t, trainer logits z_t, p_t = softmax(z_t), and
sampler distribution q_t. Gradients are taken w.r.t. the logits. The
parameter gradient is J_t^T times this, where J_t = dz_t/dtheta. The score of the
sampled token is

    grad_z log p(y) = e_y - p_t .

The current loss is `-(1/N) sum_i A_i sum_t w_{i,t} log p(y_{i,t})`, where
w = 1 when there is no correction. The update direction (minus the gradient)
has expectation over rollouts y ~ q

    g = E_q[ sum_t A * w_t * (e_{y_t} - p_t) ] .

**Decomposition (no correction, w = 1).** Write e_y - p = (e_y - q) + (q - p):

    g = E_q[ sum_t A (e_{y_t} - q_t) ]            (I)  sampler's own score
      + E_q[ sum_t A (q_t - p_t) ]                (II) drift

and note

    q_t - p_t = grad_z sum_v q_t(v) log p_t(v) = - grad_z KL(q_t || p_t).

* (I) is the REINFORCE gradient of the sampler's expected reward. If the
  sampler's logits are z + delta with delta fixed (weight noise,
  quantization error), then e_y - q = grad_z log q(y). It satisfies
  E_q[e_{y_t} - q_t | c_t] = 0, so the part of A that does not depend on y_t (any
  baseline, the group mean, a reward offset) adds nothing to it in
  expectation.
* (II) = -E_q[ sum_t A * grad KL(q_t || p_t) ]. Gradient descent on the loss
  therefore takes a step along -A * grad KL(q_t||p_t) at every token. The
  step lowers the KL to the sampler where A > 0 and **raises it where A < 0**.

**On-policy (q = p).** q_t - p_t = 0 for every context, so (II) vanishes
identically, and g is the policy gradient. This is the only reason
REINFORCE does not care about baselines, reward offsets, or token-count
weighting: E_p[grad log p | c] = sum_v grad p_v = 0.

**Mismatched (q != p).** (II) is not zero-mean noise. It is a fixed function
of (theta, q), so it acts as a bias that is present at every step:

* With uncentered rewards (`plus_minus`, `zero_minus`) E[A] != 0. When E[A] < 0
  (most rollouts fail) the drift is -E[A] * grad KL summed over all tokens:
  plain gradient *ascent* on KL(q || p).
* With group centering, sum_i A_i = 0, but per group
  E[sum_i A_i sum_t (q-p)_{i,t}] = (G-1) Cov_q(R, sum_t (q_t - p_t)). This is
  nonzero whenever the contexts or the number of tokens correlate with reward.
  For example, failed rollouts are longer, and the loss sums over tokens with
  a global 1/N, so they contribute more tokens of KL ascent than successful
  ones contribute of KL descent. Centering removes the sequence-level mean of
  the score, not the per-token, context-dependent mean.

**Why it compounds.** Ascent on KL(q||p) w.r.t. p lowers p on tokens that
the sampler favors over the trainer (where q > p). That raises the mismatch, which
makes q - p, and so (II), larger at the next step. The sampler is rebuilt as
f(theta) with the same perturbation (fixed noise delta, the same quantizer, the same
staleness), so successive steps of (II) point the same way and add up
instead of averaging out. KL(q||p) is unbounded (p(y) -> 0 on tokens that q
samples), so the loop ends in exploding -log p, huge gradients, entropy
collapse and reward collapse. Rare-token mass in large vocabularies makes
this worse: q puts small mass on many tokens where p is even smaller, and on each
of them the negative-advantage term keeps pushing p down.

## 2. Why IS corrections and clipping only slow it down

With a detached weight w(r), r = p/q, the conditional mean of the weighted
score is

    E_{y~q}[ w(y) (e_y - p) | c ] = sum_v m_v e_v - (sum_v m_v) p,   m_v = q_v w(p_v/q_v).

This is zero if and only if m = p, i.e. w(r) = r on **every** token, which is
naive IS (huge variance). Any truncation, clamp or drop makes m_v != p_v on the
out-of-band tokens. The residual sum_v (m_v - p_v)(e_v - p) is again a
systematic drift that the advantage multiplies, so the same mechanism is
present at a smaller scale. A tighter token-level band moves more tokens out of
band, so m differs from p more and the drift gets *larger*. That is why
stronger token-level clipping can make things worse. Sequence-level weights
have the same issue, with variance that grows with length.

## 3. The fix: subtract the score's expectation under the sampler

For each token, define the control variate

    b_t = sum_v m_{t,v} log p_t(v),   m_{t,v} = sg(q_t(v) w(p_t(v)/q_t(v)))     (w = 1: m = q)

and use `w_t log p(y_t) - b_t` in place of `w_t log p(y_t)`. Its logit gradient is

    w(y)(e_y - p) - sum_v m_v (e_v - p).

**Property 1 (exact centering).** E_{y~q}[ w(y)(e_y - p) - sum_v m_v (e_v - p) | c ]
= sum_v q_v w_v (e_v - p) - sum_v m_v (e_v - p) = 0, for any weighting
function w, including none. The per-token score is a martingale difference
under the distribution that actually generated the data. As in the on-policy
case, offsets and baselines in A, and the token-count imbalance between good
and bad rollouts, cannot produce a systematic update.

**Property 2 (no change on-policy).** If q = p and w = 1, then
grad b_t = sum_v p_v grad log p_v = grad sum_v p_v = 0. The gradient equals the
original loss's gradient exactly, not just in expectation. For naive IS
(w = p/q) we get m = p and the same zero, so naive IS is unchanged too. The
modification only acts where the original estimator is biased.

**Property 3 (what the update becomes).** Without correction, the new expected
update is

    g_new = g - (II) = E_q[ sum_t A (e_{y_t} - q_t) ] .

This is term (I) alone: REINFORCE on the sampler's own score, applied through
the trainer's logit Jacobian. With q = softmax(z(theta) + delta) and delta
fixed, it is the gradient of the sampler's expected reward E_q[R], up to
the difference J_p vs J_q between the two Jacobians. That difference is O(delta),
has no sign, and does not grow. The only term that systematically moved p
relative to q, -A grad KL(q||p), is gone, so the feedback loop of Section 1 is
broken. Improving the sampler's reward is the right objective here: the sampler
is a fixed map of theta, and it is what generates the data.

**Equivalent reading.** Since sum_v q_v log p_v = -H(q) - KL(q||p), with no
correction the per-token loss is, up to terms with zero gradient,

    -A [ log p(y_t) + KL(q_t || p_t) ]   ... with grad  -A[(e_y - p) - (q - p)] = -A (e_y - q).

The modification adds an advantage-weighted KL(q_t||p_t) term with exactly the
coefficient that cancels the drift. It is not a tuned regularizer. Its
coefficient is the advantage, so it is zero in expectation when q = p and
needs no hyperparameter.

## 4. Implementation (`losses.py`)

`score_baseline` computes b_t inside the existing chunked, rematerialized loop
(no extra [B, T, V] memory). `loss_fn_rl` uses `score = w*logp - b` with no flag.

* **q over the vocabulary.** The sampler returns its top-k (k = 20) logprobs
  per position (`sampler_topk_ids`, `sampler_topk_logprobs`). This is the same
  data inference engines return. On the head K, q is exact. The tail outside the
  head has mass q_tail = 1 - sum_K q and is given the trainer's shape:
  q_hat(v) = q_tail p(v)/p_tail. Then
  b = sum_K m_v log p_v + sg(m_tail / p_tail) * sum_{v not in K} sg(p_v) log p_v,
  where the ratio p_tail/q_tail is shared by the whole tail, so the tail weight
  is m_tail = q_tail w(p_tail/q_tail). The tail sum is computed as
  (full-vocab sum minus head sum), using only the existing gather.
  The leftover bias is q_tail * (q_tail-conditional - p_tail-conditional),
  which is second order. With k >= vocabulary size the centering is exact. The
  tail term is skipped when p_tail or q_tail < 1e-4, to avoid 0/0 when the head
  covers everything.
* **Fallback.** Without top-k data the head is the sampled token alone
  (its sampler logprob is always available), with the tail as above.
* **Corrections.** m uses the existing `log_mass(log p, log q, ...)`, which is
  exactly log(q w(p/q)), with the same band, outside rule, sign and TV rule
  as the token weight. The control variate therefore centers TIS, MIS, PPO,
  DPPO and others. With sequence-level corrections the per-token mass is q, and
  the sequence weight multiplies A as before.
* Checked numerically: with q = p the gradient matches the original loss to
  1e-6 (also with corrections). With the full vocabulary as the head, the
  q-expected logit gradient is 0 to 3e-8, against 5e-2 for the original loss.
* Metrics and the signature are unchanged.

No configuration change is required: the modification is active with
`correction=None`.

## Sanity check (CPU dev setting, 600 steps, seed 0)

On the visible setting, the original loss reached eval reward 0.72 and then
fell back to 0.66. The new loss went from 0.56 to 0.83, rising steadily. With
30 extra unlikely vocabulary tokens (`MINI_EXTRA_TOKENS=30`), the original loss
collapsed (eval 0.49 -> 0.25 by step 25 and stayed there). The new loss went
from 0.49 to 0.79 with no correction, and to 0.88 with TIS.
