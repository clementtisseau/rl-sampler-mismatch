# Sampler-mismatch-robust REINFORCE

## What is wrong with the current loss

Notation: at a state s the trainer has p(.|s) = softmax(z), the sampler has q(.|s) != p. Rollouts come from q, but
the loss is `-A * sum_t log p(x_t|s_t)`. Its logit-gradient at one position is A (e_x - p).

For a token drawn from q, with advantage A independent of x (a constant baseline, a nonzero group or batch mean,
or credit that is spread over a whole sequence):

    E_{x~q}[A (e_x - p)] = A (q - p) != 0.

Under p this is the usual score-function identity E_p[e_x - p] = 0, which is why REINFORCE needs no per-token
baseline. Under q the identity breaks, leaving a systematic term A (q - p) at every position. It is not zero-mean noise.
- Its size is proportional to the mismatch, so it does not shrink with more samples.
- It is a cross-entropy gradient: the term equals A * grad_z E_q[log p]. For A > 0 the trainer is pulled toward the
  sampler. For A < 0 (failed rollouts, plus_minus rewards, any sequence-level credit) it is pushed away from q:
  p is driven to where q is small, and the mismatch is made larger.
- The sampler is refreshed from the trainer, so the next q inherits the distortion and the bias compounds.
  The result is a feedback loop: bigger mismatch, bigger bias, bigger mismatch. This shows up as entropy or tail-mass
  drift and collapse. In the dev run with many tail tokens, plain PG goes to reward 0.
- Importance weights w = p/q (token level) only partly fix this. They are detached and clipped, so they do not remove
  the term when the weights are truncated. They also blow up the variance on the tail tokens that matter, and
  tighter token clipping zeroes exactly the tokens that carry the correction, which makes things worse.

## Fix: sampler-centred score (a control variate that is exact under q)

Replace log p(x) by

    log p(x) - b(s),   b(s) = E_{v~q}[log p(v)]   (q detached, p differentiable).

Its logit-gradient is (e_x - p) - (q - p) = e_x - q. So E_{x~q}[A (e_x - q)] = 0 for any constant A. The state-dependent
bias is removed exactly, and what is left is the covariance between A and the token, which is the real learning signal:
E_q[A (e_x - q)] = Cov_q(A, e_x). This is the usual baseline argument, applied to the right distribution: the
baseline for samples from q must be centred under q. The score function e_x - p is centred only under p.
The control variate costs nothing in signal and removes the term that drives the feedback loop.
When q = p it is identical to the original loss, since b has zero gradient contribution beyond the original score.

Computation. The sampler's top-k logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20, which inference
engines return) give

    b = sum_{v in top} q_v log p_v  +  q_tail * (sum_{v not in top} p_v log p_v) / p_tail,

with q_tail = 1 - sum_top q, p_tail = 1 - sum_top p. The tail is approximated by p restricted to the tail. This is
exact when q and p agree on the tail shape, and the error is second order in the tail mismatch. It is computed
inside the existing rematerialised vocabulary chunks. If top-k fields are absent the loss falls back to the plain score.
The path is active by default with `correction=None`. An optional IS correction multiplies the centred score as before.

## Configuration

rl.is.level=null
