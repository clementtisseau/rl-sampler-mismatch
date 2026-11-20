# Sampler-centered REINFORCE: removing the mismatch drift of the score function

## Setup and notation

Look at one completion position t with a fixed prefix h = (prompt, y_<t). The
trainer's next-token distribution is p = softmax(z), where z in R^V are the
trainer's logits at that position. The sampler's distribution is q (noisy, quantized
or stale weights), and the token is drawn as y ~ q(.|h). The sequence advantage
is A (group-centered reward). Write e_y for the one-hot vector of y.

The current loss is plain REINFORCE, L = -(1/N) sum_{i,t} A_i log p(y_{i,t}).
The gradient of one term with respect to the logits at its position is

    d/dz [A log p(y)] = A (e_y - p).                                     (1)

Every parameter gradient is J^T times this vector, where J = dz/dtheta. So it is
enough to study the logit-space vector (1).

## What goes wrong: the score is not zero-mean under q

Split the advantage into the part that depends on the sampled token and the part
that does not: A(y) = b(h) + (A(y) - b(h)), with b(h) = E_{y~q}[A | h]. Here b
contains every component of the advantage that this token does not explain: the
group-mean offset, the value of the prefix, and the credit for later tokens.

The expected update at this position, over y ~ q, is

    E_q[A (e_y - p)] = E_q[(A - b)(e_y - p)] + b * E_q[e_y - p]
                     = E_q[(A - b) e_y] + b (q - p).                    (2)

* **On-policy (q = p).** E_p[e_y - p] = p - p = 0. The score function has
  zero mean, so b drops out of the expectation. Only the covariance between
  advantage and token remains, sum_y p_y (A_y - b) e_y, which is the policy
  gradient. That is why REINFORCE works with any baseline, and why centered and
  uncentered rewards have the same expectation.
* **Mismatch (q != p).** E_q[e_y - p] = q - p != 0. A term b(h) (q - p)
  appears that does not depend on how good y is. It is a deterministic drift of
  the logits along q - p, with a size set by the prefix's advantage offset b(h):
    - b > 0 (prefix inside a successful rollout): p is pulled toward q;
    - b < 0 (prefix inside a failed rollout): p is pushed **away** from q,
      along p - q. The gap between trainer and sampler grows.

This is a bias, not noise. Averaging more samples does not remove it. Group
centering does not remove it either: centering makes sum_i A_i = 0 only over
whole sequences that share a prompt, and only the first completion token
shares a prefix. Deeper in the sequence the conditional mean b(h) is nonzero
almost everywhere (it is the advantage of the prefix). With
plus/minus rewards b is nonzero even at the first token. The sampler is rebuilt
from the trainer at every step (q_k = Q(p_k), where Q is the fixed perturbation,
quantizer or lag), so the drift has the same sign step after step, and it acts
where the mismatch is largest. Its size grows with ||q - p||, so the drift
feeds itself: the trainer moves along p - q, which increases the mismatch
the next sampler shows. The usual result is that probability mass leaks to
tokens the sampler under-weights (for example the long tail of unlikely
tokens), followed by collapse. In the CPU dev setup with extra tail tokens, plain REINFORCE goes
to reward 0.

### Why IS weights and clipping do not fix it

With a detached token weight w(y) = w(p_y/q_y), the expected update for a
constant advantage b is

    b * E_q[w (e_y - p)] = b (m - |m| p),     m_y := q_y w_y.            (3)

This vanishes iff m is proportional to p, i.e. w = c * p/q exactly. Any
truncation or dropping makes m != p on the tokens where the ratio leaves
the band. Under clamping at C, m_y = min(p_y, C q_y), and the drift is
b(|d| p - d) with d = p - m >= 0. For b < 0 this raises exactly the
logits of the tokens where p >> q, so it again pushes p away from q. So
TIS/MIS/PPO-style clipping only shrinks the drift, and tighter token clipping
truncates more mass (larger d), which can make it worse. Exact IS removes the drift but has variance
sum p^2/q, which is large in the tail. Sequence-level ratios multiply these
factors over T tokens.

## The fix: center the score under the sampler, not under the trainer

The drift exists because the loss subtracts the trainer's expectation p,
while the token is drawn from q. The remedy is to subtract the sampler's
expectation. Per token, use

    l_t = A * ( log p(y_t) - sum_v sg(q_v) log p_v ),                   (4)

where sg is stop-gradient. Since d/dz sum_v q_v log p_v = q - p sum_v q_v = q - p,

    d l_t / dz = A (e_y - p) - A (q - p) = A (e_y - q).                  (5)

Its expectation under the sampler is

    E_q[A (e_y - q)] = E_q[(A - b)(e_y - q)] + b E_q[e_y - q]
                     = sum_y q_y (A_y - b) e_y + 0.                     (6)

Properties:

1. **No drift for any baseline.** The token-independent part b(h) of the
   advantage now contributes exactly 0 in expectation, as in the on-policy
   case. What is left is the covariance between the advantage and the token,
   i.e. the policy-gradient direction for the policy that actually generated
   the data. Logits move only toward tokens that did better than average
   under q.
2. **Exactly REINFORCE on-policy.** If q = p the extra term has gradient
   A(q - p) = 0, so the method changes nothing when there is no mismatch.
3. **No ratios, no clipping.** The coefficients q_v are probabilities in
   [0, 1], so the variance is the same as REINFORCE's (||e_y - q|| <= sqrt 2).
   No IS weight can explode or needs truncation.
4. **Ascent direction.** For a fixed prefix, (6) is the exact gradient of
   E_{y~softmax(z)}[A] evaluated at the sampler's logits. Since the
   sampler's logits are a small perturbation of the trainer's, it is the
   trainer's gradient up to O(||q - p||) curvature error. The error is
   proportional to the advantage signal, not to the baseline, and it does not
   have a fixed sign in the p - q direction, so it does not compound.

### What the sampler exposes: top-k plus tail

The sampler returns the logprob of the sampled token and its top-k (k = 20)
logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`), as inference engines
do. q is known exactly on the top-k set K. On the rest, only the total mass
Q_tail = 1 - sum_{v in K} q_v is known. That mass is spread in proportion to p:

    qhat_v = q_v                       for v in K,
    qhat_v = Q_tail * p_v / P_tail     for v not in K,   P_tail = 1 - sum_{v in K} p_v.

Then sum_v qhat_v = 1, and E_q[e_y] - qhat = 0 on K. Off K, the residual
q_v - Q_tail p_v / P_tail has total mass zero. It only redistributes mass
inside the tail and never moves mass between the tail and the head, so it
cannot drain the head or inflate the tail as a whole, which is how collapse happens. The
centering term is computed in O(V) per position from the trainer's
log-softmax, which is already materialized:

    sum_v qhat_v log p_v = sum_{k in K} q_k log p_k
        + (Q_tail / P_tail) * ( sum_v sg(p_v) log p_v - sum_{k in K} sg(p_k) log p_k ).

If the batch carries no top-k logprobs, K = {y} with q_y from
`sampling_token_logprobs` (top-1 fallback). This is still exact on-policy, and it
removes the drift in the direction of the sampled token. It is weaker than the
top-k version.

### Combination with IS corrections (optional)

With an `ISCorrection`, the sampled token keeps its usual weight w_y.
The centering term uses the same weight per vocabulary entry,
sum_v sg(qhat_v w_v) log p_v, with w_v = w(p_v/qhat_v) (w(P_tail/Q_tail) on
the tail). The gradient is then w_y (e_y - p) - (qhat∘w - p * sum qhat∘w).
Its mean under q is q∘w - qhat∘w ~ 0 for any truncation band. So with
centering, clipping no longer produces the drift in (3). Sequence-level weights
scale A as before. No correction is needed: the default
`loss_fn_rl(hidden, head, batch)` (correction=None) is the method.

## Implementation (losses.py)

* `sampler_centering(logp_vocab, b, positive, c)` returns
  sum_v sg(qhat_v w_v) log p_v per position: top-k from the sampler, tail mass
  spread in proportion to p, with a top-1 fallback.
* `loss_fn_rl` uses score = w * log p(y) - centering for every completion token,
  inside the existing rematerialized chunks. Everything else is unchanged:
  normalization by token count, sequence-level corrections, and the
  metrics. The signature and the `correction=None` default are unchanged, and the
  method is always active.
* Checks: when the top-k covers the vocabulary, the logit-gradient equals
  A(e_y - q)/N to 1e-8. With q = p, the gradient equals the original
  REINFORCE gradient to 1e-8.

On the CPU dev setup (noise 0.3, plus/minus rewards), plain REINFORCE and TIS
collapse to reward ~0 when the vocabulary has unlikely extra tokens
(MINI_EXTRA_TOKENS=200/1000, or 50 with 7 digits). The sampler-centered loss
trains stably to 0.75-0.93 in every case. Without extra tokens it also does
better (0.85 vs 0.6). This is a sanity check only. The argument above does not
depend on it.

## Configuration

No configuration changes are needed. The method is active with the default
arguments and with `correction=None`.
