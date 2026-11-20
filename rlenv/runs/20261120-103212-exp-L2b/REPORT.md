# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The REINFORCE score `grad log p(y)` has mean zero only when tokens are drawn from
p. When they are drawn from a different sampler q, its mean at a fixed prefix is
`q - p` (in logit space). Every advantage carries a component that is constant
given the prefix, so the loss adds a systematic term `V_q * (q - p)` to each
update. With V_q < 0 that term does gradient *ascent* on KL(q || p). With V_q > 0
it distils p toward the sampler's error. The sampler is rebuilt from the trainer
at every step, so the same bias returns every step and adds up.

**Fix (in `losses.py`, on by default):** subtract, at every token, the exact
conditional expectation under q of the per-token score:

    s(y) = w(y) log p(y) - sum_v sg(q(v) w(v)) log p(v)          (w = 1 without a correction)

The expectation over v ~ q is computed from the sampler's top-k logprobs
(`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20, which `train_rl.py` and
`dev/mini.py` already put in the batch). The q mass outside the top-k is spread
over the rest of the vocabulary in proportion to p. With w = 1 the per-token
logit gradient becomes `A (e_y - q)` instead of `A (e_y - p)`. That is exactly
on-policy REINFORCE for the distribution that generated the token, so the
expected update no longer has a term in (q - p). On-policy (q = p) the subtracted
gradient is identically zero (not just zero in expectation), so the loss is
unchanged there.

## 1. Setup at one token position

Fix a prefix h. The trainer has logits z, p = softmax(z), and the sampler draws
y ~ q. For a softmax,

    grad_z log p(y) = e_y - p.

The current loss contributes `-A log p(y)` per token, where A is the sequence's
advantage. Let Q(v) = E[A | h, y = v]. This is the expected advantage when token
v is taken at h and the rest of the sequence is sampled by q. Let
V_q = E_{v~q}[Q(v)] and V_p = E_{v~p}[Q(v)]. The update direction (minus the
loss gradient) has conditional expectation

    g_q = E_{y~q}[A (e_y - p)] = sum_v q(v) Q(v) (e_v - p).

**On-policy (q = p).** Because sum_v p(v)(e_v - p) = 0,

    g_p = sum_v p(v) (Q(v) - V_p) (e_v - p) = Cov_p(Q(y), e_y).

The update is invariant to any baseline b(h). Adding a constant to all
advantages at h changes nothing in expectation. This property is why REINFORCE
can use group-centered or ±1 rewards at all. Group centering only makes the mean
advantage zero per prompt, not per prefix, so V_p(h) != 0 at almost every
position.

**Mismatched (q != p).** Split Q = V_q + (Q - V_q):

    g_q = sum_v q(v)(Q(v) - V_q)(e_v - p) + V_q sum_v q(v)(e_v - p)
        = Cov_q(Q(y), e_y)            +  V_q (q - p).                       (1)

(The first term loses its p because sum_v q(v)(Q(v) - V_q) = 0.) The first term
is the real learning signal: the on-policy gradient of the sampler policy. The
second term is a **drift that has nothing to do with the reward ranking**. It is
proportional to the mean advantage at the prefix and to the mismatch. Since
grad_z KL(q||p) = p - q,

    V_q (q - p) = -V_q grad_z KL(q || p).

- V_q < 0 (prefixes in failing rollouts, all of the "unsolved" mass under ±1
  rewards): the update **ascends** KL(q||p). It raises logits where p > q and
  lowers them where q > p. This is unbounded: it pushes p away from q, which
  increases ||q - p||, which increases the drift. That is positive feedback.
- V_q > 0: the update pulls p toward q, i.e. it trains the policy on the
  sampler's error. The next sampler is f(p'), which applies the same systematic
  perturbation again, so p walks along the direction of the sampler bias step
  after step.

For a fixed mismatch mechanism f (fixed weight noise, quantization, stale
weights), q - p = f(p) - p is a deterministic, slowly varying function of p. The
drift therefore has the same sign every step and compounds; it is not zero-mean
noise. It also does not shrink as learning progresses. As the policy sharpens,
the signal Cov_q(Q, e_y) goes to zero, but V_q (q - p) stays as long as q != p.
This explains the late collapse. Large vocabularies make it worse. Noise or
quantization gives the many unlikely tokens q(v) > p(v), and the ascent on
KL(q||p) keeps pushing them around.

The prefix (state) distribution also differs between q and p. That only
reweights which prefixes receive updates. By itself it does not create a biased
direction at a prefix. The bias is per position, as (1) shows.

## 2. Why IS weights and clipping do not remove it

With a detached per-token weight w(y) = w(r(y)), r = p/q:

    E_q[w A (e_y - p)] = sum_v q w (Q - V_q)(e_v - p) + V_q sum_v q(v) w(v) (e_v - p).

The drift term is V_q (m - |m| p) with m = q∘w and |m| = sum_v m(v). It vanishes
for every V_q **only if q w ∝ p**, i.e. w ∝ p/q (naive, unclipped IS). Naive IS
does give g = sum_v p(v)(Q(v) - V_p)(e_v - p), which is unbiased. But its
variance is E_q[r^2] = 1 + chi^2(p||q), and the weights must be clipped or
truncated in practice. As soon as w != r on some tokens, the drift returns:

- Truncation (TIS, w = min(r, C)): drift = V_q (min(p, Cq) - |m| p),
  which is nonzero on the tokens with p > Cq.
- Dropping outside a band [1-eps, 1+eps] (PPO/MIS style, w = r 1[in band]): then
  q w = p 1[in], and the drift is V_q (p_in - P_in p). For V_q < 0 it moves
  probability **from in-band tokens to out-of-band tokens**, i.e. toward the
  tokens where p and q already disagree most. A tighter band puts more tokens
  outside it, which makes the drift larger. This is why stronger token-level
  clipping can make things worse.

So IS reweighting only shrinks the coefficient of the drift. It does not make
the score mean-zero under q.

## 3. The fix: center the score under the sampler

The on-policy property we lost is E_{y~q}[score] = 0. We restore it directly by
subtracting the exact conditional mean of the (weighted) score. Weights are
detached:

    s(y) = w(y) log p(y) - C,     C = sum_v sg(q(v) w(v)) log p(v),
    grad_z s = w(y)(e_y - p) - sum_v q(v) w(v)(e_v - p).

C depends only on the prefix, not on y, so it acts as a control variate. For an
advantage that is constant at h:

    E_{y~q}[V grad_z s] = V [ sum_v q w (e_v - p) - sum_v q w (e_v - p) ] = 0

for **any** weight function w, clipped or not. The full expected update is

    E_q[A grad_z s] = sum_v q(v) w(v) (Q(v) - V_q) (e_v - p).               (2)

Special cases:

- **No correction (w = 1, the default).** The subtracted term is the
  deterministic vector q - p, and per sample

      grad_z s = (e_y - p) - (q - p) = e_y - q,      E_q[A (e_y - q)] = Cov_q(Q(y), e_y).

  This is exactly on-policy REINFORCE for the distribution that produced the
  token, applied to the trainer's logits. The V_q (q - p) term of (1) is gone.
  Nothing in the update depends on p - q, so the mismatch cannot feed back into
  itself.
- **Naive IS (w = r).** q w = p, so C has zero gradient (sum_v p(e_v - p) = 0) and
  (2) is the exact on-policy gradient sum_v p (Q - V_p)(e_v - p). Centering and
  unbiased IS agree.
- **Clipped or banded IS (TIS, MIS, PPO, DPPO, ...).** (2) is a reweighted
  covariance with nonnegative weights q w. The clipping changes how much each
  token counts, but it no longer creates a drift. The bias of clipping is reduced
  from O(V_q ||q - p||) to a weighting of the true signal.
- **On-policy (q = p, w = 1).** grad_z C = sum_v p(e_v - p) = 0 exactly, so the
  loss gives the same gradient as before.

**Improvement property (w = 1).** To first order, the change in the trainer's
conditional value is sum_v p(v)(Q(v) - V_p) Delta z_v with
Delta z_v = q(v)(Q(v) - V_q):

    sum_v p q (Q - V_q)^2 + (V_q - V_p) sum_v p q (Q - V_q).

This is positive whenever p ≈ q: the second term is second order in the mismatch.
Suppose the sampler acts on the trainer's logits as a perturbation, which to
first order shares the same Jacobian. Then the sampler's own value changes by
sum_v q^2 (Q - V_q)^2 >= 0. The policy that generates the data improves
monotonically, and the fixed point is Cov_q = 0 (the sampler is greedy-optimal
at h). There is no drift along f(p) - p.

**Variance.** The added term is A times a vector that is fixed given the prefix,
with ||q - p||_1 <= 2. It contains no ratio, so unlike IS it cannot blow up.

## 4. Computing E_{v~q} with top-k logprobs

Inference engines return only the top-k sampler logprobs. Let K be the top-k
set, q_T = 1 - sum_{v in K} q(v), and P_T = 1 - sum_{v in K} p(v). The tail
q-mass is spread like p, q̂(v) = q_T p(v) / P_T for v not in K, and the tail is
treated as a single outcome:

    C = sum_{v in K} sg(q(v) w(v)) log p(v) + sg(q_T w_T) log P_T,      w_T = w(P_T / q_T).

Since grad_z log P_T = p_tail / P_T - p, this equals sum_v q̂ w (e_v - p) with q̂ in
place of q on the tail. With w = 1, the leftover drift is
V_q (q_tail - q_T p_tail / P_T). It lives only on tail tokens, sums to zero, and
has l1 norm at most 2 q_T (q_T is tiny for k = 20). If the top-k covers the whole
vocabulary, the formula is exact. Numerics: P_T is floored at 1e-6 and the tail
term is dropped below the floor. The gradient of q_T log P_T is bounded by 2 q_T
regardless of P_T, so this floor is harmless.

If a batch has no top-k logprobs, the code falls back to K = {sampled token} (an
approximation that is still exact on-policy). Sequence-level corrections
(`seq_mean`, `seq_sum`) still multiply the advantage as before. The centering is
token-level and is exact for token-level weights.

## 5. Implementation (`losses.py`)

- `centering_term(...)` computes C per position. It gathers log p on the
  sampler's top-k ids, builds the detached masses q∘w on the top-k and the tail
  with the existing `log_mass` (so every `rl.is.*` correction is supported), and
  returns `sum_K m log p + m_T log P_T`.
- `loss_fn_rl` uses `score = w * logp - center` per token, inside the same
  chunked and rematerialized loop. The signature and defaults are unchanged, and
  the centering is active with `correction=None`.
- I checked the gradients against the closed forms on random logits: with full
  top-k they match `A (e_y - q)` (w = 1) and
  `A [w(y)(e_y - p) - sum_v q w (e_v - p)]` (naive IS and TIS) to 1e-7.

No configuration changes are needed: the modification is active with the default (no IS correction).
