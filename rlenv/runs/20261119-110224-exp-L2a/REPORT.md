# Sampler-centered REINFORCE under sampler/trainer mismatch

## 1. What is wrong with the current loss

Notation. Trainer policy p = p_theta, sampler q (the rollouts come from q).
At a prefix s, write g(y) = grad_theta log p(y|s) for a next token y. The loss
in `loss_fn_rl` with w = 1 gives, per token,

    grad(-L) = (1/N) sum_i A_i sum_t g(x_t^i),   x_t^i ~ q(.|s_t^i).

Taking the expectation over the sampler, for a token at prefix s:

    E_q[ g(x) | s ] = sum_y q(y|s) grad log p(y|s) =: b_q(s).

On-policy, q = p, the score identity gives b_p(s) = grad sum_y p(y|s) = 0.
So under q = p any part of the advantage that does not depend on the sampled
next token contributes nothing in expectation. That is why REINFORCE with
baselines and group centering is unbiased on-policy.

Under q != p, b_q(s) != 0. In logit space, for a softmax head, grad_z log p(y) = e_y - p, so
b_q = q - p. A constant advantage c gives an expected update c (q - p): a
cross-entropy step toward the sampler's distribution, i.e. -c grad CE(q, p).
Whenever the advantage of a token has a nonzero mean given its prefix (it
does, since the value V(s) is not zero and group centering removes it only
over whole sequences and weighted by length), the update contains the
term c_s (q(.|s) - p(.|s)) at every position.

This term is systematic and not zero-mean noise. Because the sampler is
refreshed from the trainer every step, q - p is re-created from the new theta
each step, so the drift accumulates. Its sign depends on the advantage: a
positive offset sharpens p toward q, and a negative offset pushes mass away
from the sampler's modes, which tends toward a high-entropy or off-task
distribution. This matches the observed collapse of plain REINFORCE.

Clipped importance sampling does not remove the term. With the ratio
w(y) = min(p_y/q_y, h), a constant advantage gives

    E_q[ w(x) g(x) | s ] = sum_y q_y min(p_y/q_y, h) g(y)
                         = b_p(s) - sum_{y: p_y > h q_y} (p_y - h q_y) g(y)
                         = - sum_{y: r_y > h} (p_y - h q_y) g(y),

which is nonzero whenever clipping is active. It is proportional to the clipped
mass, so a stronger clip (smaller h) means a larger residual drift. This is
consistent with the known fact that stronger token-level clipping can make
things worse. Unclipped per-token IS, w = p/q, is exact for this term, because
sum_y q_y (p_y/q_y) g(y) = b_p = 0. Its variance is the problem.

## 2. The modification: subtract the sampler expectation of the score

Use the per-token estimator

    g_t = w(x_t) g(x_t) - sum_y q(y|s_t) w(y|s_t) g(y),

where w is the existing detached IS weight (w = 1 with no correction). The
second term is a control variate that depends only on the prefix s_t and the
sampler distribution. It is what makes the constant-advantage expectation
vanish exactly for any weight.

Check: for any advantage A that is independent of x_t given s_t,

    E_q[ A (w(x_t) g(x_t) - sum_y q_y w_y g_y) ]
      = E_q[A] ( sum_y q_y w_y g_y - sum_y q_y w_y g_y ) = 0.

Properties:
- On-policy (q = p) with w = 1: sum_y q_y g_y = b_p = 0, so the baseline
  vanishes and the loss is the original REINFORCE. Nothing changes.
- With unclipped w = p/q: sum_y q_y w_y g_y = sum_y p_y g_y = 0, so the
  baseline is zero and the estimator reduces to naive token IS.
- With clipped w (TIS, DPPO-style bands, etc.): the baseline cancels the
  residual drift from section 1 exactly, without removing the clip.

The loss is in `losses.py`. The baseline is computed on the sampler's top-k
tokens (`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20 in `train_rl.py`,
which both trainers already pass). This gives sum over top-k instead of the
full vocabulary. Excluded tail mass is small for an LLM, and in `dev/mini.py`
the top-k covers the full vocabulary, so there it is exact. The baseline
gradient flows through log p (not detached), and the IS weight of the baseline
is detached, as the derivation requires.

Scope and limits. The per-token centering removes the drift created by the
state-dependence of b_q. It cannot remove the trajectory-level part of the
bias, because the advantage of a token depends on future tokens drawn from q,
not p. That part (the signal distortion term, E_q[(A - V) g] versus
E_p[(A - V) g]) needs sequence-level weights, and the IS band in section 3
handles it. Centering without the band is still stable for a while but
eventually declines, as shown below.

## 3. Derivation summary

1. On-policy, E_p[g] = 0, so constant advantage offsets are harmless.
2. Under q, E_q[g] = b_q = q - p (logit space), a drift toward the sampler.
3. Clipped IS leaves the residual -sum_{r>h}(p - hq) g, which is nonzero.
4. Subtracting the q-weighted expectation of w g removes the constant-offset
   bias exactly for any w, and is a no-op on-policy with w = 1.

## Configuration

rl.is.level=token
rl.is.high=2.0

The default (no correction) already contains the centering. The two keys add a
per-token IS band `[0, 2]` with `outside=drop`, which bounds the trajectory-level
residual. Both keys are needed to match the best setting below.

## Evidence (dev/mini.py, noise 1.0 on the sampler, eval reward at step 800)

These runs are the basis for the choices above. They are not required.
- Plain REINFORCE (seed 0): 0.554 at step 0, 0.125 at step 800 (KL to sampler 86).
- Centered, no correction (seed 0): 0.287 at 800, 0.321 at 800 with seed 1.
  Plain REINFORCE with seed 1 reached 0.319 at step 400.
- TIS high=2 clamp (seed 0): 0.48 at 800. Centered TIS: 0.488 (seed 0), 0.53 (seed 1).
- Centered, token IS high=2 drop (seed 0): 0.52 at 800. Centered + high=1.2
  clamp 0.508; high=5 clamp 0.491. Centered + seq_mean band [0.5, 2] 0.453.
- Naive token IS (no centering) and centered token IS with high=inf give
  identical trajectories, which checks the claim that the baseline vanishes
  for w = p/q.
