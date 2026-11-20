# Sampler-mismatch-robust REINFORCE: a sampler-baseline control variate

## Summary

The current loss weights every sampled token's score function log p(y_t | c_t) by
the advantage A and nothing else. Under a sampler q != p, the expectation of that
score is not zero, so any nonzero advantage level leaks a systematic term into the
update: an advantage-weighted pull of p toward q. Because the sampler is reset to p
every step, this pull is applied again and again. The fix subtracts, from each
token's score, the sampler's own expected log-probability over the next token,
log p(y_t | c_t) - E_{v~q(.|c_t)} log p(v | c_t). This score has zero mean under q
at every prefix, exactly as log p has zero mean under p. It needs no weights, no
clipping and no extra flag, and it is on by default in `loss_fn_rl`. It uses the
sampler's top-k logprobs, which the trainer already receives (`sampler_topk_ids`,
`sampler_topk_logprobs`).

## Notation

- c_t = (prompt, y_1..y_{t-1}) is the prefix; p_c(v) = p(v | c), q_c(v) = q(v | c).
- Score of token t: s_t = ∇_θ log p_c(y_t). Sequence score: s(y) = Σ_t s_t.
- Advantage A(y) per sequence (group-centered, so it depends on the whole group).
- Loss: L(θ) = -(1/N) Σ_i A_i Σ_t w_it log p_θ(y_it | c_it), with w = 1 when no correction is used.
- Update direction under the sampler: g_q = E_{y~q}[ A(y) Σ_t s_t ] (the loss gradient, up to the 1/N normalization).
- Desired direction: g_p = E_{y~p}[ A(y) s(y) ] = ∇_θ E_p[R] (the policy gradient, for R = reward).

## 1. What the current loss computes

Per prefix c, the expected score under a distribution d is

  m_d(c) := E_{v~d_c}[∇ log p_c(v)] = Σ_v d_c(v) ∇ log p_c(v).

Two identities matter:

1. On-policy, m_p(c) = Σ_v p_c(v) ∇ p_c(v) / p_c(v) = Σ_v ∇ p_c(v) = ∇ Σ_v p_c(v) = ∇1 = 0.
   So under q = p, the score has zero mean at every prefix. Any constant added to the
   advantage, or any baseline that does not depend on the sampled token, has zero
   expected effect, and the loss is an unbiased policy gradient.

2. Under q ≠ p, m_q(c) = m_q(c) - m_p(c) = Σ_v (q_c(v) - p_c(v)) ∇ log p_c(v) ≠ 0 in general. It is first order in the mismatch δ = q - p.
   Also m_q(c) = Σ_v q_c(v) ∇ log p_c(v) = -∇_θ H(q_c, p_c), where H(q, p) = -Σ_v q(v) log p(v)
   is the cross-entropy. So m_q is the gradient of the cross-entropy of p against the sampler.

Now split the advantage at each token using a prefix-measurable baseline β_t(c_t) := E_q[A | c_t]:

  E_q[A Σ_t s_t] = Σ_t E_q[(A - β_t) s_t] + Σ_t E_q[β_t(c_t) m_q(c_t)].

The first sum is the centered REINFORCE term. The second is a bias: it equals
-∇ Σ_t E_q[β_t H(q_{c_t}, p_{c_t})], a gradient step on the cross-entropy between
p and the sampler, weighted by the conditional mean advantage β_t. Under q = p, m_q = 0 and this bias vanishes for any β.

Consequences for the group-centered advantages used by `compute_advantages`:

- The group mean does not remove the bias. With G samples per group, iid draws give
  E[A_i s_i] = (1 - 1/G)(E_q[a s] - E_q[a] μ_q), where a is the un-centered ±1
  advantage and μ_q = E_q[s] = Σ_t E_q[m_q(c_t)]. The term E_q[a] μ_q is the bias. It is
  nonzero whenever the mean reward is nonzero and q ≠ p. Its sign follows the mean advantage (positive mean reward: pull toward the sampler's samples; negative: push away).
- Clipping makes it worse. Naive IS with weights w = p/q removes the constant term exactly, since E_q[(p/q) s] = Σ_v p(v) ∇ log p(v) = 0. Clipping or dropping w replaces p/q by a function f(p/q) with E_q[f s] ≠ 0 in general. The leftover bias is then E_q[(f(r) - 1) s]·β, which grows with the clip strength and can exceed the unclipped error. This explains why stronger token-level clipping can make things worse. IS and clipping only shrink this bias. They do not remove it.
- Why it compounds. Each update moves p toward q (or away from it) in the direction m_q. The next step's sampler is a fresh copy of p, so the mismatch δ is re-created every step, and the bias accumulates over steps instead of averaging out as zero-mean noise would.

## 2. The fix: a sampler-expectation control variate

Replace the token score s_t = ∇ log p_c(y_t) by

  ℓ_t := log p_c(y_t) - Σ_v π_c(v) log p_c(v),     π_c = q_c (the sampler distribution),

so that ∇ℓ_t = ∇ log p_c(y_t) - m_π(c). Then

  E_{y_t~q}[∇ℓ_t | c] = m_q(c) - Σ_v π_c(v) ∇ log p_c(v) = m_q(c) - m_π(c) = 0 when π = q.

The score now has zero conditional mean under the sampler at every prefix. Consequences:

1. The baseline term is gone: E_q[β_t ∇ℓ_t] = E[β_t] · 0 = 0 for every prefix baseline β_t. The estimator is E_q[A Σ_t ∇ℓ_t] = Σ_t E_q[(A - β_t) ∇ℓ_t] = Σ_t E_q[(A - β_t) ∇ log p(y_t)] - Σ_t E_q[(A - β_t) m_q(c_t)] = Σ_t E_q[(A - β_t) ∇ log p(y_t)] with the second part zero because E[A - β_t | c_t] = 0. So the update is the centered REINFORCE with the correct conditional baseline, and the cross-entropy pull is removed.

2. On-policy nothing changes. With π = p, m_p(c) = 0, so ∇ℓ_t = ∇ log p(y_t) in expectation and the loss is the same unbiased policy gradient.

3. No weights are needed. The control variate uses only the sampler's distribution, which is already returned as top-k logprobs. Its variance is that of a few extra score terms per token, and it does not blow up the way p/q weights do. It also does not clip anything, so it has no bias from clipping.

Notes on the implementation. The code sets π = softmax of the sampler's top-k logprobs (renormalized over the top k) and computes
`baseline = Σ_v π_v log p_c(v)` with `take_vocab` over the top-k ids. The tail mass outside the top k (sum of q over the non-top-k tokens) is left out, so the residual bias is bounded by that tail mass times the spread of ∇ log p over the tail. For the 12-token dev model the top-k is the full vocabulary, so π = q exactly. When the batch has no top-k keys, the loss falls back to the plain score log p(y_t) (the old behavior).

The IS weight for `rl.is.*` corrections is unchanged and multiplies this score. With the default settings there is no correction, so w = 1 and the modified score is the only change.

## 3. Why this addresses the stated failure modes

- Systematic bias, not zero-mean noise: the bias term E_q[β m_q] is removed in expectation at every prefix, so there is no systematic drift toward the sampler.
- Compounding over steps: since the per-step expected bias is zero, the sampler refresh no longer feeds a bias back into the next update.
- Clipping: nothing is clipped, so no clipping-induced bias.

## Configuration

No configuration changes are needed. The modification is active under default arguments (`correction=None`) and uses the top-k fields that the sampler already returns. This section is intentionally left without `key=value` lines.

## Empirical check (dev/mini.py, not required)

On the CPU mini setup (plus_minus rewards, sampler noise 0.3 and 0.5, 4 seeds, 600 steps, correction=None), the original loss reached a reward of about 0.55 to 0.59 at noise 0.3, and it collapsed to 0 for every seed at noise 0.5. With the modified loss the reward reached about 0.82 to 0.85 at noise 0.3, and it stayed at 0.49 to 0.56 at noise 0.5 without collapse. At noise 0.7 it was reduced to about 0.28 (peak 0.39) and did not collapse to 0. Adding naive token IS on top of the new loss did not help in that setting (collapse at noise 0.5), which is consistent with the claim that IS weights don't address the bias on their own. These numbers are not required, and they come from the 12-token dev model, not Qwen3.
