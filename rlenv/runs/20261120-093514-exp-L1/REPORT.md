# Sampler-centered score for REINFORCE under sampler/trainer mismatch

## What is wrong with the current loss
Rollouts y ~ q (sampler), the loss is -A(y) * sum_t log p(y_t | y_<t) (p = trainer). Consider one position and
write the gradient w.r.t. the trainer logits z: grad_z log p(y) = e_y - p. REINFORCE is unbiased only because
E_{y~p}[e_y - p] = 0, which is what lets a baseline / group centering remove the "common" part of the reward.
When y ~ q:

  E_{y~q}[e_y - p] = q - p != 0.

So every sampled token carries a spurious term (q - p) that does not cancel with A-centering: its weight is the mean advantage
(and, for the correlated part, tokens the sampler over-produces). Equivalently E_q[grad log p] = -grad_theta CE(q,p): the loss
contains a systematic term that pulls p toward q (and away from q for negative A) in addition to the reward signal. This is a bias, not
zero-mean noise. Because the sampler is rebuilt from the trainer each step, p moves, q moves with it, and the drift accumulates
(e.g. with a noisy sampler that over-samples tail tokens, failing samples push the observed tokens down, shifting mass in a consistent direction),
so reward degrades over training. Importance weights p/q on the token only rescale the sampled token's score and
do not fix the (q-p) term in the other vocabulary entries; clipping/dropping tokens additionally removes the (q - p) term
selectively by ratio, making the bias state-dependent (hence stronger clipping can hurt).

## Fix: sampler-centered score
Use, at each position, the score of the distribution that actually generated the data:

  g(y) = e_y - m,   m = q (the sampler's next-token distribution),

implemented as the surrogate  s_t = log p(y_t) - sum_v m_v log p(v)  (m detached), so that grad_z s = e_y - m (log-softmax is shift-invariant, and m sums to 1).
Now E_{y~q}[e_y - q] = 0 exactly, for any p and any q. The centering term vanishes in expectation, so baselines/group-centering work as
in the on-policy case and the mismatch-induced drift (q-p) is gone. Moreover E_q[A (e_y - q)] is precisely the policy-gradient of the sampling
policy in logit space, J(q) = E_q[A]: the update direction is the correct policy-gradient direction for the policy that produced the data,
pushed through the trainer's Jacobian dz/dtheta. When q = p, s has gradient e_y - p and the loss equals REINFORCE exactly, so nothing is lost
without mismatch. It is bounded (|e_y - m| <= sqrt2), so no variance blowup like sequence-level IS ratios.

Practicalities: the sampler returns top-20 logprobs (batch keys sampler_topk_ids / sampler_topk_logprobs). m is the top-k
q exactly; the residual tail mass q_tail = 1 - sum_topk q is distributed over non-top-k tokens in proportion to p
(m_v = q_tail p_v / p_tail), so the sum is cheap: sum_v m_v log p_v = sum_topk q_k log p_k + (q_tail/p_tail) * (sum_v p_v log p_v - sum_topk p_k log p_k).
If top-k keys are absent the loss falls back to the plain score. The optional IS correction (`correction`) still multiplies the score as before;
default is `correction=None`, and the modification is active by default.

## Configuration
rl.is.level=null
