# Sampler-centered REINFORCE: the score must have zero mean under the sampler

## 1. Setup

At one completion position, fix the prefix s. The trainer's next-token
distribution is p = softmax(z) with logits z = z_theta(s). The sampler draws
the token y from q (another distribution: noisy, quantized or stale
weights), and records log q(y). The rollout's advantage is A (group-centered
reward, one number per sequence). The current loss is, per token,

    L = - A * w * log p(y),     w = 1 without a correction (or a detached w(p/q)),

so the per-token update direction in logit space is

    g(y) = A w (grad_z log p(y)) = A w (e_y - p),                         (1)

where e_y is the one-hot vector of y. The parameter update is
J^T g with J = dz/dtheta, so whatever holds for the expectation of g holds
for the parameter update at that prefix.

## 2. Expected update: on-policy versus mismatched

Write the advantage at this position as A = b(s) + (A - b(s)), where b(s) is
any part that does not depend on y (e.g. the mean of A over continuations of
s; group centering only removes such a part at the prompt, not at deeper
prefixes, and with `plus_minus` rewards the advantage is ±1 everywhere).
For the y-independent part b, with w = 1:

    E_{y~q}[ b (e_y - p) ] = b (q - p).                                   (2)

* **On-policy (q = p):** E_p[e_y - p] = p - p = 0. Any y-independent part of
  the advantage has zero expected effect; this is exactly why baselines
  (and group centering) are allowed in REINFORCE. Only the y-dependent part
  of A moves the logits, and it moves them along the true policy gradient
  sum_y p(y) (A(y) - b)(e_y - p).

* **Mismatched (q != p):** the score e_y - p has mean q - p != 0 under the
  distribution the tokens actually come from. So the constant part of every
  advantage produces a deterministic drift b (q - p). Since

        q - p = grad_z sum_v q_v log p_v = - grad_z KL(q || p),          (3)

  a token with b > 0 performs a step of *minimizing* KL(q||p) (pulling p
  toward the sampler), and a token with b < 0 performs a step of
  *maximizing* KL(q||p) (pushing p away from the sampler). This term
  carries no reward information at all; it is a bias of the objective, not
  zero-mean noise: averaging more samples makes it more precise, not smaller.

Why it compounds and collapses: the sampler is rebuilt from the trainer at
every step (q = q(theta + delta), same fixed perturbation, quantizer or lag),
so q - p does not shrink when p moves. KL maximization along negative
advantages is unbounded (it lowers log p on tokens that q favours, pushes
mass into tokens q rarely produces, sharpens or flattens p in the direction
opposite to the perturbation), and the sampler follows p, so the next step
sees the same kind of gap and pushes again. The drift adds up linearly in
the number of steps until p reaches regions where the trainer/sampler gap
blows up (tokens with p << q), which feed the drift more: the collapse.
Negative-advantage tokens sampled where p(y) << q(y) are the worst: their
push -|A|(e_y - p) lowers an already small p(y), increasing log q - log p.

Why IS / clipping only slows it down: with a detached weight w(p/q),

    E_q[ b w (e_y - p) ] = b (m - |m| p),   m_y = q_y w(p_y/q_y),  |m| = sum_y m_y.  (4)

This vanishes for all p iff m is proportional to p, i.e. only for the
unclipped ratio w = p/q. Any clamp or drop (TIS, MIS, PPO/DAPO bands, DPPO)
makes m a mixture of q and p on the clipped tokens, and the drift
b (m/|m| - p) remains, only smaller. Tighter token clipping puts *more*
tokens on q's side (m = q times a constant there), which is why it can make
things worse. Sequence-level weights multiply A by a constant per sequence,
so they rescale b but cannot remove (2). The unclipped ratio removes the
drift but its variance is 1 + chi^2(p||q) per token and explodes on exactly
the tokens where the mismatch is large.

## 3. The fix: center the score at the sampling measure

Replace log p(y) by a score that has zero mean under the distribution that
actually generated (and weighted) the sample. Let m be the effective
sampling measure, m_v = q_v w(p_v/q_v) (m = q with no correction). The
per-token loss becomes

    L = - A * w(y) * ( log p(y) - sum_v sg(m_v / |m|) log p_v ),           (5)

(sg = stop-gradient; only log p carries gradient). In logit space,

    grad_z [ log p(y) - sum_v (m_v/|m|) log p_v ] = (e_y - p) - (m/|m| - p) = e_y - m/|m|.

With no correction this is e_y - q: the score of the *sampler's* own
categorical distribution, applied to the trainer's logits.

**Drift removal (exact).** For the y-independent part b of the advantage,

    E_{y~q}[ b w(y) (e_y - m/|m|) ] = b ( m - |m| m/|m| ) = 0             (6)

for every p, q and every correction w. The baseline-invariance of
REINFORCE is restored under the sampler, so there is no KL(q||p) term left
in the objective, whatever the sign of the advantage.

**On-policy unchanged.** If q = p and w = 1, sum_v p_v grad log p_v = 0, so
(5) has exactly the REINFORCE gradient: the modification only removes the
term that is zero on-policy.

**What signal remains.** With w = 1, the expected logit update is

    E_q[ A(y) (e_y - q) ] = sum_y q_y (A(y) - Abar_q) e_y,  Abar_q = E_q[A],     (7)

which is the exact on-policy policy gradient (in logit space) of the
sampler's own policy: a token's logit goes up iff its advantage is above the
sampler's average at that prefix. The trainer and sampler logits differ by a
perturbation that is fixed or slowly varying in theta (noise delta,
quantization error, a lag), so moving z along (7) improves the sampler that
is rebuilt from it, which is the policy that produces the rewards. No term
of the update depends on the size or sign of q - p, so nothing accumulates
from the mismatch itself: the mismatch now only changes *which* policy's
improvement direction we follow (q's, which is p's up to the perturbation),
instead of adding a reward-independent force.

The per-token update magnitude is also safer: |e_y - q| <= sqrt(2), and a
negative-advantage token with p(y) << q(y) no longer pushes a tiny p(y) toward
zero beyond what the sampler's own gradient would.

## 4. What the sampler needs to record

The centering term needs q over the vocabulary at each position. The
framework can record it (`sampler.vocab_logprobs`): `full` ([B,T,V], fine
for small vocabularies), or the sampler's top-k head
(`sampler_topk_ids`, `sampler_topk_logprobs`). For a head K, `losses.py`
keeps the centering exact as follows: a sampled token inside K is treated as
above (measure q w), a sampled token in the tail (outside K) gets the exact
ratio w = p/q. Then the tail's measure is q (p/q) = p, which the trainer
knows, so

    m_v = q_v w(p_v/q_v) for v in K,   m_v = p_v for v not in K,

and (6) holds exactly again (checked numerically by enumerating y at a
fixed prefix: the expected gradient for a constant advantage is 0 to
float precision, for no correction, TIS and PPO bands, with full or top-k
recording). With k large enough that q's tail mass is negligible (k = 64
for an LLM vocabulary; in `dev/mini.py` k = 64 exceeds the vocabulary and is
the full distribution) the tail ratios are almost never used.

Fallback when nothing is recorded (default batch, `correction=None`): the
centering of (5) is not computable from log q(y) alone; the only
per-token weighting that satisfies the zero-drift condition (4) is m = p,
i.e. the unclipped token ratio w = p/q, whose centering term
sum_v p_v grad log p_v is identically zero. `losses.py` uses it in that
case (this is the k = 0 limit of the top-k rule above), so the default call
is drift-free in expectation in every case; recording the head of q is what
makes it low-variance. If the caller passes a clipped correction without
recorded distributions, the call honours that correction (and its residual
drift (4)).

## 5. Implementation (`losses.py`)

* `sampler_measure(logp_vocab, b, positive, c)` returns
  sum_v (m_v/|m|) log p_v per position from `sampling_vocab_logprobs`
  (full) or from `sampler_topk_ids`/`sampler_topk_logprobs` (head + p on
  the tail), with m built by the same `log_mass` as the IS weights, so it
  is consistent with any `rl.is.*` correction.
* `loss_fn_rl` uses score = log p(y) - center in place of log p(y);
  sampled tail tokens get w = p/q; with no recorded distributions and no
  correction it uses w = p/q. Signature, chunking, sequence-level
  corrections and metrics are unchanged (plus `is_weight` and
  `sampler_centered` metrics).
* On-policy (q = p) the gradient equals the original REINFORCE gradient
  (checked numerically to 1e-8).

## 6. Sanity check (dev/mini.py, CPU, noise 0.3, 600 steps; not required)

| setting | original loss | centered, k=64 | fallback (no recording, w = p/q) |
| --- | --- | --- | --- |
| 12-token vocab, seed 0: last-10 train reward | 0.58 | 0.85 | 0.64 |
| 12-token vocab, seed 1 | - | 0.85 (full) | 0.64 |
| +200 unlikely tokens (`MINI_EXTRA_TOKENS=200`) | collapses to 0.00 by step 25 | 0.79 | 0.62 |

## Configuration

sampler.vocab_logprobs=64
