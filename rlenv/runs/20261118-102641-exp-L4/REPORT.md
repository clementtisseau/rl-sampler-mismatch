# Sampler-mismatch stability: a drift-free score in REINFORCE

## What is wrong with the current loss

`loss_fn_rl` (with `correction=None`) is plain REINFORCE on the sampled tokens:
`loss = -sum_t A * log p(y_t|h_t) / N`. Its per-token gradient is `-A * s_y`,
with `s_y = grad_theta log p(y|h)`.

Fix a prefix h. The next token y is drawn from the sampler q(.|h), and A is the
(sequence-level) advantage. The expected update at this position is

    E_q[A s_y | h] = E_q[A | h] * s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h) = sum_v q(v|h) s_v.

On-policy (q = p) the first term vanishes identically, because
`s_bar_p(h) = sum_v p_v grad log p_v = grad sum_v p_v = 0`. Only the covariance,
which is the true policy-gradient signal, is left.

Under a sampler mismatch, `s_bar_q(h) != 0` in general. Using
`sum_v p_v s_v = 0`,

    s_bar_q(h) = sum_v (q_v - p_v) s_v = grad_theta [ sum_v q(v|h) log p(v|h) ],

which is the gradient of the cross-entropy of p(.|h) towards q(.|h). So the
current loss adds, at every prefix, the term

    drift(h) = E_q[A | h] * grad_theta sum_v q(v|h) log p(v|h),

i.e. it moves p(.|h) towards q(.|h) where the expected advantage is positive
and away from q(.|h) where it is negative. This has nothing to do with the
reward: it appears even when the advantages carry no information about the
tokens. It is not zero-mean noise: for a fixed prefix it has a fixed direction
set by the sampler's systematic offset from p. Because the sampler is refreshed
from the trainer at every step, q tracks p, the drift keeps pushing p toward a
sampler that moves with it, and the bias compounds over training.

Importance sampling does not remove this term in practice. Per-token weights
w(p/q) are exact only when unclipped, and clipping or dropping (TIS, MIS, PPO,
...) truncates the weights in a way that depends on the token, which leaves a
residual bias of the same kind, proportional to the clipped mass. Stronger
token-level clipping truncates more, so the residual grows.

## The fix: subtract the sampler-expected score

Replace each token's score `s_y` by `s_y - s_bar_q(h)`, keeping the same
advantage A. Its expectation under q is exact:

    E_q[A (s_y - s_bar_q) | h] = E_q[A s_y | h] - E_q[A | h] s_bar_q(h)
                               = Cov_q(A, s_y | h).

The drift is gone for every prefix, with no importance ratio. On-policy,
`s_bar_p(h) = 0` exactly, so the modified term is identical to the original and
the change is zero (not only in expectation: the subtracted quantity is
identically zero when q = p).

Implementation with one backward pass. Use the surrogate, with sg = stop-gradient,

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),

whose gradient is `-A (s_y - sum_v q_v s_v) = -A (s_y - s_bar_q)`. The
sampled token keeps its own `log p(y_t|h)` term. In `losses.py` the per-token
term is `w * (logp - sampler_baseline)` with `w` the (detached) IS weight from
`rl.is.*` if a correction is given, and `w = 1` otherwise.

### Top-k head only

If the sampler only returns its top-k head H (ids and logprobs), the tail of
q is modelled as proportional to p: `q_tail(v) = rho p_v` for `v not in H`,
with `rho = q_tail / p_tail`, where `q_tail = 1 - sum_H q_v` and
`p_tail = 1 - sum_H p_v`. Then

    sum_v q'_v s_v = sum_H q_v s_v + rho sum_{tail} p_v s_v
                   = sum_H (q_v - rho p_v) s_v,

using `sum_v p_v s_v = 0`. The tail contributes no gradient, so the surrogate is

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ),

with `rho` and `p_v` in the coefficient stop-gradient. The sampled token keeps
its own log p even when it lies outside H. The result is exact if the sampler's
tail really is proportional to p, and otherwise leaves a residual drift
proportional to the error in the tail shape, multiplied by the advantage
(see the limitation below).

## Why this is the right object

- It is a control variate. For any q, `E_q[s_y - c(h)] = s_bar_q - c(h)`, so
  subtracting the baseline `c(h) = s_bar_q(h)` (a function of h only, not of y)
  leaves the gradient unbiased for the policy-gradient *covariance* and removes
  only the term that is proportional to E[A|h].
- The advantage is sequence-level, but the drift argument is at a single
  position: the expectation is over the next token given the prefix, and A is
  the advantage of the completion, so the identity above holds with
  A = A(completion). Future tokens enter only through E[A | h].

## Limitations

- For q != p, `Cov_q(A, s_y|h)` is not the same vector as the policy gradient
  `E_p[A s_y|h]`. The remaining difference is `sum_v (q_v - p_v) A_v s_v`, where
  `A_v = E[A | h, y = v]`. It is of the size of the mismatch and does not carry
  the factor `E[A|h] s_bar`, so it does not grow with the advantage offset or
  compound through the drift. It is not removed by this change.
- With the top-k head, the tail-shape error reintroduces a small drift, scaled
  by the advantage. In the CPU mini setup (12-token vocabulary, k=4), this
  was enough to collapse 2 of 3 seeds late in training, whereas the full
  vocabulary was stable on all seeds. The full vocabulary is the recommended
  setting; use top-k only with a large head.
- Without the sampler's next-token distribution (`sampler.vocab_logprobs=null`)
  the loss has no q(.|h) and falls back to plain REINFORCE. The
  modification is active by default in `loss_fn_rl`, but it needs the sampler
  to record that distribution.

## Configuration

sampler.vocab_logprobs=full

## Checks run

- Dev mini setup (`dev/mini.py`, sampler noise scale 0.3, 600 steps, 3 seeds),
  eval reward at the end: plain REINFORCE 0.66, 0.66, 0.68; with the
  sampler-centred score (full vocabulary) 0.83, 0.83, 0.86, with no collapse.
- Top-k head with k=4 on the same setup: final eval 0.48, 0.51, 0.79
  (two collapses), so the top-k variant is only a fallback.
