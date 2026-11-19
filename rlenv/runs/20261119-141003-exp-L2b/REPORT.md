# Sampler mismatch in REINFORCE: a q-centred score

## Summary

With a sampler q different from the trainer p, the REINFORCE token gradient
is biased. At each position the expected update is `A (q - p)` in logit
space, a systematic push of the trainer toward (positive A) or away from
(negative A) the sampler's distribution. Because the sampler is refreshed
from the trainer every step, this push accumulates. `losses.py` removes the
part of the bias that comes from the advantage's prefix-level mean by
centring the sampled score: the score `grad log p(x)` is replaced by
`grad log p(x) - E_{v~q}[grad log p(v)]`. Under q, this centred score has
zero mean for any q, which is the property the on-policy score has under p.

## 1. What the current loss does

Fix a prefix s and let p = p_theta(.|s), q = q(.|s) be the next-token
distributions of trainer and sampler, with logits z. For a sampled token x
the default loss (no correction, w = 1) gives the token term

    l(x) = - A * log p(x),

where A is the advantage of the sequence containing x. Write Q(s,x) for the
expected advantage given the prefix and the token, Q(s,x) = E[A | s, x]. For
the per-position analysis, A is replaced by Q(s,x).

The gradient in logit space is grad_z log p(x) = e_x - p, so

    G(x) = grad_z l(x) = - Q(s,x) (e_x - p).

**On-policy (q = p).** The expected gradient is
E_p[G] = -(sum_x p(x) Q(x) e_x - (p . Q) p), the policy gradient in logit
space. It is correct and unbiased.

**Mismatched (q != p).** Tokens are drawn from q, so

    E_q[G] = -(q o Q - (q . Q) p),

where o is the elementwise product. Let u = q - p (the mismatch, sum u = 0).
The error relative to the on-policy target is

    E_q[G] - E_p[G] = -( u o Q - (u . Q) p ).

Split Q = V + Q~, where V = sum_x p(x) Q(x) is the prefix value under the
trainer and sum_x p(x) Q~(x) = 0. Then

    E_q[G] - E_p[G] = -( V u  +  u o Q~  -  (u . Q~) p ).          (1)

The first term, V u, is the systematic bias. It is first order in the
mismatch, and it is nonzero whenever the prefix has nonzero expected
advantage. Group-centring makes advantages average to zero across the
sequences of a group, not across the continuations of each prefix. So V(s)
is generally nonzero at intermediate positions: after a wrong digit V is
negative, and after a correct prefix it is positive. For constant advantage
(Q~ = 0), the expected update is exactly -V(q - p): the trainer moves
toward q where V > 0 and away from q where V < 0. Each step's sampler is
refreshed from the trainer, so these moves accumulate and the trainer drifts
along its own sampler-induced direction. This is the compounding bias
described in the task.

Clipping or a truncated ratio changes which samples contribute but not the
form of this bias, which is why it slows degradation without removing it.

## 2. The fix: a q-centred score

Replace the token score by its centred version, subtracting the sampler's
expected score as a baseline:

    b(s) = E_{v~q}[ log p(v|s) ] = sum_v q(v|s) log p(v|s),

    l_c(x) = - A * ( log p(x) - b ).

b depends on p (it is differentiated) and on q (a constant). Its gradient is
grad_z b = sum_v q(v)(e_v - p) = q - p, so the centred token gradient is

    G_c(x) = - A ( (e_x - p) - (q - p) ) = - A (e_x - q).             (2)

**Key identity.** Under the sampler, the centred score has zero mean for any
q, since E_q[e_x - q] = q - q = 0. So for constant A the expected update is
exactly zero, both on-policy and off-policy:

    E_q[G_c] = 0 = E_p[G_c]   (constant A).

The bias term V u in (1) is removed exactly. In the notation of (1), with
the new gradient,

    E_q[G_c] - E_p[G] = -( u o Q~  -  (u . Q~) q ).                    (3)

Comparing (1) and (3), the new error contains no V-term, and its remaining
part is first order in u times the token-specific advantage Q~. This is the
same kind of term as in (1), with q replacing p in the last part. The
prefix-level bias is gone. Only the part of the advantage that depends on
which token is chosen remains, and that is the part an exact importance
correction (p/q) would address.

The centring costs nothing on-policy: at q = p, the baseline gradient
q - p vanishes, so the on-policy update is unchanged. The score itself is
bounded (e_x - q has entries in [-1, 1]), so there is no importance-weight
variance blow-up of the kind that makes IS and clipping fragile under
large mismatch.

## 3. Implementation with the sampler's top-k

The trainer only sees the sampler's top-K logprobs per position, not the
full q. Let K_t be the top-K token ids at position t and q_v = exp(logq_v)
the sampler's probabilities on them. The baseline is estimated as

    b_hat_t = sum_{v in K_t} q_v log p_t(v),

with unnormalised q_v, so the expectation is taken under the sampler's own
values. If the sampler's support lies inside its top-K, b_hat is exact. In
general the error is the tail term: with R the complement of K and
m_R = sum_R q, the expected update is -A(p_R - m_R p) on-policy and
-A(q_R - m_R p) off-policy, both of size O(tail mass). This is much smaller
than the O(mismatch) bias in (1), which is not limited by the top-K cut.

`losses.py` computes `score = w * (log p(x) - b_hat)` in the same chunked
pass as the existing score, with `b_hat` differentiated through
`log p` at the top-K ids. Where the batch has no top-K logprobs, b_hat = 0
and the loss reduces to the uncentred REINFORCE. Both trainers supply them
(top-20 in `train_rl.py` and `dev/mini.py`).

## 4. Why not importance sampling alone

The exact sampler-to-trainer correction for the score, with weight p/q,
gives E_q[(p/q) A (e_x - p)] = A sum_x p(x)(e_x - p) = 0 for constant A. It
removes the same bias exactly, but its weights p(x)/q(x) grow without bound
for tokens the sampler under-produces, so variance explodes under large
mismatch. Clipping at c bounds the weight but biases every clipped token
back toward q, which is why stronger clipping can make things worse. The
centred score achieves the constant-advantage correction without any
weight, so its variance does not depend on the mismatch size.

## 5. Scope and limitations

- The analysis is per position, conditional on the prefix, with Q(s,x) the
  advantage expected given the prefix and token. The continuation after x is
  still drawn from q, so the Q~ term in (3) remains. It is the part an
  importance weight would address, and combining the centred score with an
  `rl.is.*` correction is supported (the weight multiplies the centred term).
- The top-K estimate has the tail error stated in section 3.
- Dev evidence (not required, for orientation only): on the CPU mini setting with the
  uncentred loss, the reward collapses to 0 at weight-noise 0.6 and at noise 0.3 with 300
  extra vocabulary tokens. With the centred score, the mean reward over the last 20 steps (2 seeds) is about
  0.35 at noise 0.6 and 0.72 at noise 0.3 with 300 extra tokens (uncentred: 0), and
  on-policy (noise 0) it matches the uncentred loss (about 0.94). Under the harshest setting the
  reward still drifts down, which is consistent with the uncorrected token-specific term in (3).
- No configuration changes are required: the modification is active with
  the default `correction=None`.
