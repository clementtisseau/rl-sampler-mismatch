# REINFORCE under sampler/trainer mismatch: a sampler-expectation baseline

## 1. What is wrong with the current loss

Notation. At each completion position t, with context c_t (prefix), the trainer
defines p_t(v) = p_θ(v | c_t) and the sampler defines q_t(v) (the distribution the
rollouts were actually drawn from). A completion y = (y_1..y_T) is drawn from q,
and receives a scalar advantage A(y) that depends on the whole completion. The
current loss with `correction=None` is

  L(θ) = −(1/N) Σ_i A_i Σ_t log p_θ(y_it | c_it),        y_i ∼ q,

so its gradient estimate is ĝ = −(1/N) Σ_i A_i Σ_t ∇log p_θ(y_it | c_it).

The on-policy REINFORCE gradient is g* = E_{y∼p}[A(y) Σ_t ∇log p(y_t | c_t)].
The loss therefore estimates E_q[·] where it should estimate E_p[·].

Two facts about the score function drive the failure.

(a) On-policy, the score has zero mean at every position:
    E_{v∼p_t}[∇log p_t(v)] = Σ_v ∇p_t(v) = 0.
    This is what makes a constant advantage harmless: adding a constant to A
    leaves g* unchanged.

(b) Under the sampler it does not:
    E_{v∼q_t}[∇log p_t(v)] = Σ_v q_t(v) ∇log p_t(v) = −∇_θ H(q_t, p_t) ≠ 0,
    where H(q, p) = −Σ_v q(v) log p(v) is the cross-entropy.

Consequence. Take a position whose advantage is (approximately) constant given the
context, A ≡ a. The expected update from the current loss is

  E_q[ĝ] ⊃ −a · E_q[∇ log p_t] = a · ∇_θ H(q_t, p_t).

Minimising the loss with a > 0 (correct completions) pulls p_t toward q_t. Minimising it
with a < 0 (wrong completions, the common case early on, and always the case in
`plus_minus` mode, where advantages are ±1 and not centred) pushes p_t away from q_t,
toward the tokens the sampler under-produces. This is a systematic drift, not
zero-mean noise. Its direction depends only on the mismatch, and it does not vanish
when the sample size grows. Because the sampler is refreshed from the trainer every
step, each round of drift changes q, and the drift compounds.

Group centring does not remove it. For a group of G samples with rewards r_i and
A_i = r_i − r̄ (r̄ the sample mean), a short calculation gives
  E_q[A_i ∇log p_i] = (1 − 1/G) · Cov_q(r, ∇log p),
and Cov_q(r, s) = E_q[r s] − E_q[r] E_q[s]. The term E_q[r] E_q[s] is the same
q-expectation of the score, now multiplied by the mean reward. Centring the rewards
cancels constant shifts within the sample, but not the drift in expectation.

Why importance sampling helps, and why clipping brings the bias back. The token-level
weight w = p/q makes the constant part exact:

  E_q[ (p/q) ∇log p ] = Σ_v p(v) ∇log p(v) = 0.

With a clip at c, the weight is min(p/q, c), and

  E_q[ min(p/q, c) ∇log p ] = Σ_v min(p(v), c·q(v)) ∇log p(v)
                           = −Σ_{v: p>cq} (p(v) − c q(v)) ∇log p(v).

This is nonzero, and it grows as c shrinks: the tokens the trainer likes much more
than the sampler (p > cq) are under-pushed. So stronger token-level clipping
makes the drift larger, as the task notes. Clipping and IS slow the drift but do not
remove it. Also, the unclipped weight p/q has large variance under a large mismatch,
which is why naive IS is not stable either.

## 2. The modification: a sampler-expectation baseline

For each position, define the sampler-expected log-probability

  β_t(θ) = Σ_v q_t(v) log p_θ(v | c_t),        with q_t held fixed (stop-gradient).

Its gradient is ∇β_t = Σ_v q_t(v) ∇log p_t(v). The loss uses the centred score
∇log p_θ(y_t | c_t) − ∇β_t:

  L_new(θ) = −(1/N) Σ_i A_i Σ_t [ log p_θ(y_it | c_it) − β_t(θ) ].

Its expectation over y_t ∼ q_t given the context is exactly zero, because
E_{y∼q_t}[∇log p_t(y)] − ∇β_t = Σ_v q_t(v)∇log p_t(v) − Σ_v q_t(v)∇log p_t(v) = 0.
So the constant part of the advantage, which is the source of the drift in
Section 1, is removed exactly, and the on-policy identity E[∇log p] = 0 is restored
under the sampler's own distribution. Since β_t does not depend on the sampled token,
this is a control variate: it leaves the expectation of the constant part unchanged
(zero) and adds no weights, so no clipping, no variance blow-up from p/q, and no
extra sampling.

Implementation. The rollout batch already carries the sampler's top-K logprobs per
position (`sampler_topk_ids`, `sampler_topk_logprobs`; K = 20 in both trainers), which
the current loss ignores. We approximate q_t by its top-K renormalised distribution
q̂_t(k) = softmax(sampler_topk_logprobs)_k and set β_t = Σ_k q̂_t(k) log p_θ(id_k | c_t).
The change is confined to `chunk_sums` in `losses.py`: the score becomes
logp − β_t instead of logp, and the baseline is used only when `correction is None`.
With an ISCorrection the existing weighted estimator is unchanged, since there the
constant part is already exact at the token level and the baseline would add a
−∇β bias.

## 3. Remaining bias and limits

- Top-K truncation. q̂ drops the sampler's tail mass q_tail(t) outside the top K and
  renormalises the rest. The residual bias per position is the tail contribution
  Σ_{v∉topK} q(v)∇log p(v) minus the renormalisation error. It is of the size of the tail
  mass q_tail(t), and vanishes when K = |V| (as in dev/mini.py, where |V| ≤ 32).
- Advantage–token covariance. The baseline removes only the part of the drift that is
  constant given the context. The remaining mismatch is
  Σ_y (q − p)(y) (A(y) − Ā) ∇log p(y), which is a covariance between the reward and
  the sampler's deviation from the trainer. This is not removed by a per-position
  baseline. It is the part that would require sequence-level weights, and those have
  the variance problem discussed above. It is expected to grow with the mismatch, which
  matches the degradation seen at the largest noise.

## 4. Empirical check (dev/mini.py, not required, for context)

Same setup as the mini runs, `plus_minus` advantages, lr 0.2, 300 steps, 3–4 seeds per
setting. Final eval reward (clean sampling) is shown; "collapse" means eval 0 from about step 75.

- noise 0.3: baseline (current loss) 0.68–0.73, new loss 0.75–0.82.
- noise 0.6: baseline collapses to 0 in 4 of 4 seeds; new loss stays at 0.54–0.59
  (seeds 0, 1, 2, 3).
- noise 0.6 with 20 extra never-correct tokens (MINI_EXTRA_TOKENS=20): baseline
  collapses; new loss 0.45.
- noise 0.3 with extra tokens: baseline collapses; new loss 0.81.
- noise 1.0: both degrade; the new loss ends at 0.14–0.24 against 0 for the baseline.
- group-centred rewards at noise 0.6: new loss 0.63, baseline 0.47 (one seed).
- Existing corrections at noise 0.6: token-level TIS, MIS, PPO and DPPO end at eval
  0.0–0.05 (collapsed), and unclipped token-level IS collapses. The sequence-level
  TOPR-style variant (seq_sum, sign=neg, high=1) stays stable, with eval 0.61 at seed 0,
  but its training reward is about 0.18 against 0.37 for the new loss.

These numbers come from a tiny CPU model and are not evidence for Qwen3-0.6B.

## 5. Configuration

No configuration changes are needed. The loss is active under the default arguments
(`correction=None`) and takes effect whenever the batch carries the sampler's top-K
logprobs. Without them it falls back to plain REINFORCE.
