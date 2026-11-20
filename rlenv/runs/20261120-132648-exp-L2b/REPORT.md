# Stable RL under sampler/trainer mismatch: a sampler-centered score

## Summary

The REINFORCE loss `-A * log p(y)` is only zero-mean "for free" when the token is
drawn from the policy being differentiated. When the token is drawn from a
different sampler q, the score `grad log p(y)` has a non-zero mean, `q - p` in
logit space, and every part of the advantage that does not depend on the sampled
token (the baseline part, which with outcome rewards is almost all of it)
multiplies that mean. That gives a systematic drift that is not reward-driven.
Depending on the sign of the advantage it pushes p toward q or away from q. Since
q is rebuilt from p at every step, the drift never runs out and it compounds.

The fix is to center the score under the sampler: from the per-token score, subtract
its analytic expectation under q at the same prefix. Then the token-independent
part of the advantage gives exactly zero expected update for any p and q. On-policy
(q = p) the subtracted term has zero gradient, so the loss is plain REINFORCE. The
sampler's top-k logprobs, which the batch already carries (`sampler_topk_ids`,
`sampler_topk_logprobs`, k = 20, the same thing inference engines return), give
q over the vocabulary. The change is always active and needs no configuration.

## 1. Setup at one token position

Fix a prefix s. Let z be the trainer's logits at s, p = softmax(z) and q the
sampler's next-token distribution. The trainer samples y ~ q. The advantage
A(y) is a scalar per sequence, so at this position it is a random variable that
depends on y only through the continuation it leads to. Write

    Abar_q(s) = E_{y~q}[A(y) | s],      A(y) = Abar_q(s) + (A(y) - Abar_q(s)).

The current loss contributes `-A(y) log p_y`, with gradient in logit space

    g(y) = A(y) * grad_z log p_y = A(y) (e_y - p).

## 2. On-policy versus mismatched expected update

Expected update (direction of ascent):

    U = E_{y~q}[A(y)(e_y - p)] = sum_y q_y A(y) e_y - Abar_q p.

**On-policy (q = p).**
`U = p ⊙ (A - Abar_p)`. The score has zero mean, `E_p[e_y - p] = 0`, so any
part of A that is constant over y, including the baseline Abar_p(s) itself,
contributes nothing. Only the relative advantage of the next token moves the
logits. This is why REINFORCE works with any baseline.

**Mismatched (q != p).** Rearranging,

    U = q ⊙ (A - Abar_q)  +  Abar_q(s) (q - p).
        \______________/     \____________/
        learning signal          drift

The first term is the normal policy-gradient signal, evaluated at q. The second
term does not depend on how good the next token is. It is the mean advantage at
this prefix times the mean of the score under q:

    E_{y~q}[grad_z log p_y] = q - p = -grad_z KL(q || p).

So for every prefix the current loss also does

    Abar_q(s) > 0 :  gradient descent on KL(q||p)   (pull p toward the sampler)
    Abar_q(s) < 0 :  gradient ascent  on KL(q||p)   (push p away from the sampler)

with step size |Abar_q(s)|. This term is not zero-mean noise. Its sign and
direction are fixed by the prefix and by the sampler's systematic error q - p.

**Why Abar_q(s) is large.** With outcome rewards every token of a sequence gets
the same A. Group centering makes the advantages sum to zero only at the prompt,
that is at the first completion token. Deeper in a completion,
`Abar_q(s) = V(s) - V(prompt)` is already close to the sequence's own A once the
outcome is mostly decided. So on most tokens most of A is baseline, and its
update is mostly drift. For failing rollouts (A < 0) the update maximizes
KL(q||p), and it does so on exactly the tokens the sampler over-weights
relative to the trainer.

**Why it compounds.** q = Q(p) is a fixed corruption of the current trainer
(noise, quantization, staleness), rebuilt every step. Moving p does not remove
q - p; the corruption is applied again to the new p. The drift is therefore a
persistent force along a direction set by the corruption, not something that
averages out. It accumulates across steps until the policy leaves the region
where it solves the task, for example by mass leaking into tokens the sampler
over-weights. Once rewards fall, more advantages are negative, Abar becomes
more negative and the push away from q gets stronger. That feedback is the
collapse.

## 3. Why IS weights and clipping do not remove the drift

A token-level correction multiplies the score by a detached weight w_y = w(p_y/q_y):

    E_{y~q}[w_y (e_y - p)] = m - |m| p,   where m_v = q_v w(p_v/q_v), |m| = sum_v m_v.

This is zero for every p, q only if m ∝ p, which means w_y ∝ p_y/q_y: the
unclipped ratio. Any bounded or clipped w (TIS, MIS, PPO, DAPO, DPPO bands)
leaves `Abar * (m - |m| p) != 0`. Clipping only replaces q with the
partially-corrected measure m, so the drift becomes smaller but stays.
Dropping tokens (w = 0) removes mass from m in a way that depends on the ratio.
This makes m less proportional to p, which explains why stronger token-level
clipping can make the drift worse. The unclipped ratio is unbiased but has
variance sum_v p_v^2/q_v, which blows up on tokens the sampler under-weights,
so in practice it gets clipped. Sequence-level weights rescale A by a per-sequence
factor. That changes Abar but not the fact that it multiplies a score whose
mean is not zero.

## 4. The fix: center the score under the sampler

Replace the score with the score minus its expectation under q at the same prefix:

    g(y) = A(y) [ w_y grad log p_y  -  E_{y'~q}[ w_{y'} grad log p_{y'} ] ]
         = A(y) [ w_y (e_y - p)  -  (m - |m| p) ].

With no correction (w = 1, m = q) this is just

    g(y) = A(y) (e_y - q).

**Zero drift.** For each prefix and any p, q, E_{y~q}[w_y(e_y-p) - (m-|m|p)] = 0.
So the constant part Abar_q(s) of the advantage contributes exactly nothing:

    U_new = E_{y~q}[(A(y) - Abar_q) (w_y (e_y - p) - (m - |m| p))]
          = q ⊙ (A - Abar_q)                (w = 1)

This is the pure learning-signal term from Section 2, with the KL(q||p)
term removed. Equivalently, U_new is the exact on-policy REINFORCE update for
the sampler's distribution q: the trainer's logits of tokens that do better
than average under q go up, the others go down. It is a valid ascent direction
for the reward wherever q is close to p, and it has no component that moves p
toward or away from q for any reason other than reward.

**No change on-policy.** If q = p (and so w = 1, m = p) the subtracted term is
grad sum_v sg(p_v) log p_v = p - p = 0, and the loss is plain REINFORCE.

**It is a control variate, so the variance does not go up.** The subtracted term
depends only on the prefix, not on y. So it is a baseline for the score vector,
not an importance weight. No ratio is inverted, nothing is clipped and nothing
is unbounded. Its magnitude is |A| * ||m - |m|p|| <= |A| * O(TV(q,p)).

**Loss form.** In logit space the subtracted vector is the gradient of a
cross-entropy with detached coefficients:

    m - |m| p = grad_z sum_v sg(m_v) log p_v,

so the per-token score becomes

    score_t = w_y log p_y  -  sum_v sg(m_v) log p_v,      loss = -(1/N) sum A * score_t.

For w = 1 this is `log p_y + CE(sg(q), p)`. Read this way, the added
`A * CE(q, p)` term cancels, in expectation, exactly the `-A * CE(q, p)` that
the plain loss optimizes implicitly when tokens come from q.

**Getting q over the vocabulary.** The batch carries the sampler's top-k
logprobs. On those ids q is exact. The leftover sampler mass
q_tail = 1 - sum_{k in K} q_k is spread over the other tokens in proportion to
p: q̂_v = q_tail * p_v / p_tail for v not in K, with p_tail = 1 - sum_{K} p_k.
This keeps sum_v q̂_v = 1. It is exact on the top-k (where nearly all the mass
and nearly all of the systematic error live), and it keeps the head/tail mass
split exact. The remaining error, the shape of q inside the tail, enters the
drift only as `Abar * (q - q̂)`, which is bounded by the tail mass and is zero
on the first-order head/tail split. With a correction, m on the tail uses the
pooled tail ratio p_tail / q_tail. If the top-k fields are missing
and no correction is given, only q(y) is known. Then the one zero-mean score
that can be built from it is used: the unclipped token ratio,
w_y = p_y/q_y, for which m = q ⊙ p/q = p and the control term vanishes
(E_q[(p_y/q_y)(e_y - p)] = 0). Using the sampled token alone as the "known" part
of q would not work, because q̂ would then depend on y and centering would be biased.

**Implementation (`losses.py`).** The control term needs only gathers of the
full-vocab log-softmax that is already computed per chunk:
`sum_v sg(m_v) log p_v = sum_{k in K} m_k log p_k + (m_tail/p_tail) * (sum_v sg(p_v) log p_v - sum_{k in K} sg(p_k) log p_k)`.
No [V x K] tensor is built, and nothing changes for a TP-sharded vocabulary. The
coefficients m are detached, so all the gradient goes through log p. Unit
check (exact enumeration over y with K = V): E_q of the per-token gradient for a
constant advantage is 0 (about 1e-8 in float32) with no correction, with TIS,
with PPO and with DPPO. For q = p, the gradient equals the original loss's
to 2e-8.

**Interaction with corrections.** The centering is applied to whatever
token-level weight is configured: w_y e_y minus its own q-expectation m. So any
`rl.is.*` point also becomes drift-free at the token level. Sequence-level
corrections rescale A and keep the centered score. No correction is needed,
and the default (`correction=None`) is the recommended setting.

## Sanity check on the CPU proxy (dev/mini.py, noise 0.3, 600 steps)

Final eval reward (last 10 train reward in parentheses):

| setting | original loss | centered score |
| --- | --- | --- |
| default (plus_minus), seed 0 | 0.67 (0.57) | 0.83 (0.86) |
| default (plus_minus), seed 1 | 0.68 (0.57) | 0.76 (0.85) |
| 40 extra vocab tokens (top-20 < V, tail estimate used) | 0.00, collapsed | 0.79 (0.84) |
| same, top-k fields removed (ratio fallback) | 0.00, collapsed | 0.72 (0.57) |
| TIS (high=2, clamp) | 0.82 (0.62) | 0.86 (0.65) |
| group_centered, 40 extra tokens, seed 2 | 0.74 (0.64) | 0.72 (0.62) |

The original loss collapses when the sampler can put mass on tokens outside the
answer alphabet, because the drift pushes p away from q there. The centered loss
does not collapse in any of these runs.

## Configuration

No configuration changes are needed. The modification is active with the
default `rl.is.level=null`.
