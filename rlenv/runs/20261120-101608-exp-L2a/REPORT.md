# Sampler mismatch in REINFORCE: the score has a nonzero mean under q

## 1. What the current loss computes

Let p be the trainer, q the sampler, A the (group-centered) advantage of a
sequence, and h_t the prefix before token t. With no correction the loss is

    L = -(1/N) sum_b sum_t A_b * log p(y_bt | h_bt),          (w = 1)

and the tokens are drawn from q, not p. Its expected gradient is

    g = -(1/N) sum_t E_q[ A ∇log p(y_t | h_t) ].

The policy gradient we want is the same expression with E_p in place of E_q.

## 2. On-policy versus mismatched: the score has zero mean only under p

For a fixed prefix h, write s_v = ∇log p(v | h) and define the sampler-score mean

    mu_q(h) = E_{v~q(.|h)}[ s_v ] = sum_v q(v|h) ∇log p(v|h).

Then

- On-policy (q = p): mu_p = sum_v p_v ∇log p_v = sum_v ∇p_v = ∇(1) = 0.
  The score has zero mean, so any prefix-only baseline b(h) is harmless:
  E[(A - b) s] = E[A s].
- Mismatched (q != p): mu_q = sum_v (q_v - p_v) s_v, which is generally nonzero.

The baseline argument therefore fails off-policy. Split the update:

    E_q[A s_t] = E_q[ A (s_t - mu_q(h_t)) ]  +  E_q[ A mu_q(h_t) ].

The second term is zero on-policy, because mu_p = 0. Off-policy it is not.
For a constant advantage it is

    E_h[ mu_q(h) ] = ∇ sum_v q_v log p_v = -∇ H(q, p),

the gradient of the cross-entropy from q to p. So the uncorrected loss contains
a term that performs maximum-likelihood on the sampler's own tokens, that is,
it pulls the trainer toward q. For an advantage that depends on the prefix,
the same term is weighted by V(h) = E[A | h].

This is a systematic bias, not zero-mean noise, and it has a specific
direction: mass that q puts on tokens that p does not (for example the
sampler's tail or its noise-induced junk) is pushed up in p. The sampler is
refreshed from the trainer every step, so each step's drift toward q becomes
the next step's starting point, and the bias compounds. That is the collapse
mechanism. The tail mass that q puts on tokens outside the answer vocabulary
is exactly what grows in this loop.

## 3. The fix: subtract the sampler-score mean from each token's score

Replace the score of each token by its centered version

    ŝ_t = ∇log p(y_t | h_t) - mu_q(h_t),

with mu_q(h_t) computed from the sampler's distribution. The loss is

    L = -(1/N) sum_b sum_t A_b * [ log p(y_bt | h_bt) - sum_v q(v | h_bt) log p(v | h_bt) ],

with the sum in the bracket treated as a constant (stop-gradient on q). Its
gradient is exactly A_b * ŝ_t, so the expected update is

    E_q[ A ŝ_t ] = E_q[A s_t] - E_q[A mu_q(h_t)].

Because E_q[ŝ_t | h_t] = mu_q - mu_q = 0 exactly, for any q, the term
E_q[A mu_q] that caused the bias is removed exactly, not in expectation over
a Monte Carlo estimate.

Properties:
- On-policy (q = p), mu_p = 0, so the expected update equals the standard
  REINFORCE update. The fix is a no-op in expectation.
- It needs no importance weights, so there is no variance blow-up and no
  clipping bias.
- It is active by default: it applies whenever `correction` is None (and
  for sequence-level corrections, where the per-token score is unweighted).
  It does not change the loss signature.

## 4. Implementation

`losses.py` uses the sampler's top-K log-probabilities, which the sampler
already returns (`sampler_topk_ids`, `sampler_topk_logprobs`). The sampler
distribution is approximated by its top-K renormalized probabilities
omega_v = softmax(sampler_topk_logprobs)_v, and

    mu_q(h_t) ≈ sum_{v in topK} omega_v log p(v | h_t),

computed with the trainer's log-softmax (taken through the head). The
baseline is detached in omega and differentiated through log p. The correction
path (`level=token` etc.) keeps its existing per-token weight. If the top-K
arrays are absent, the loss falls back to the plain score.

The approximation error is the sampler's probability mass outside its top K,
times the difference between the average of ∇log p over that tail and over
the top K. The mass outside the top 20 is small for a sampler close to the
trainer, which is the regime this fix targets.

## 5. Why importance sampling and clipping only slow the drift

The same bias can be removed by weighting each token by w = p/q, because
E_q[(p/q) s] = sum_v p_v ∇log p_v = 0. Two things go wrong in practice:

- Clipping (TIS: w = min(p/q, c); MIS: a band) changes the weighted mean.
  E_q[min(p/q, c) s] = sum_v q_v min(p_v/q_v, c) s_v ≠ 0, so a clipped
  correction moves the bias rather than removing it. Stronger clipping moves
  more of the mass out of the centered term, which is why it can make things
  worse.
- Unclipped weights p/q have heavy tails where q is small and p is not.
  Those are the tokens the sampler under-produces, so the variance is largest
  exactly where the mismatch lives.

The centering above gets the cancellation without weights. It cancels the
sampler's own score mean rather than reweighting samples.

## 6. Limits and what is not fixed

- The residual term E_q[A s_t] - E_p[A s_t] (the shift of the reward-weighted
  score itself) is not removed by centering. That part is the one an
  importance weight addresses, and it remains first order in (q - p).
- The approximation uses the top-K sampler probabilities. Mass outside the top
  K is not accounted for.
- Mini-environment evidence (`dev/mini.py`, 600 steps, sampler noise 0.3,
  40 extra vocabulary tokens, 4 seeds): the uncorrected loss collapses to
  reward 0 by about step 75-150 in every seed. The centered loss reaches a
  last-20-step reward of about 0.76-0.85 and stays there. Naive token-level IS
  reaches about 0.55, and TIS (`high=2`, clamp) about 0.57. With no sampler
  noise (clean sampler), the centered loss matches the uncorrected loss
  (last-20 reward about 0.93-0.94 in both). At noise 0.5 the centered loss
  degrades (last-20 about 0.47-0.49, where the uncorrected loss collapses to
  0.00), so it is more stable but not immune to a stronger mismatch.
