# Sampler-mismatch-robust REINFORCE

## What is wrong with the current loss
Rollouts y ~ q (sampler), but the loss is -A(y) log p(y) (trainer p), whose gradient is A(y) grad log p(y).
The true policy gradient is grad J = E_{y~p}[A grad log p]. Under q:

  E_q[A grad log p] = E_q[(A - c) grad log p] + c * E_q[grad log p],   with   E_q[grad log p] = -grad_theta KL(q || p) != 0.

When p = q the second term vanishes (E_p[score] = 0), which is why REINFORCE needs no baseline in expectation.
When p != q, any component of the advantage that is constant (given the prefix) - a non-zero batch/group mean, the
length-weighting of the token-sum normalisation, the correlation of A with the future - multiplies a score-function
mean that is not zero. This is a systematic term, not noise: for c>0 it pulls p toward q, for c<0 (typical when most
rollouts fail) it pushes p *away* from q, i.e. in the logit space it adds c(q - p) per token. The sampler is rebuilt from
the trainer each step (q = perturb(p)), so the push is applied again every step and compounds: tails/entropy drift, the
sampler's tail mass grows, and training collapses. Weight noise, int8/fp8 and KV-quantisation all create such a q - p gap,
typically with more mass than p on unlikely tokens.

Importance weights w = p/q make the expectation correct only at sequence level and with huge variance; the token-level
clamp/drop versions (TIS, MIS, PPO...) change the weights by band membership, which is itself a state-dependent,
biased re-weighting: E_q[w_clipped grad log p] != 0 again (tokens where p/q is large - exactly those the trainer wants to
increase - are removed or clamped). Tighter clipping increases that bias, which is why it can be worse. They slow the
drift but do not remove the c*E_q[score] term's source.

## The modification: a q-centred score (control variate)
Per position, the score of the sampled token in logit space is e_y - p. Its q-mean is q - p != 0. Replace it by

  e_y - q  (zero mean under y~q, for every prefix),

by subtracting a detached-q baseline in the loss: 

  L = - sum_t A * ( log p(y_t) - b_t ),    b_t = sum_{v in topk} q_v log p_v + q_other log(1 - sum_{v in topk} p_v),

with q_v the sampler's top-20 probabilities (batch["sampler_topk_*"], stop-gradient) and q_other = 1 - sum topk q
(lumped tail). grad_z b_t = q - p on the coarse outcomes (top-k + "other"), so grad_z (log p_y - b_t) = (e_y - p) - (q - p) = e_y - q:
the per-token gradient is the exact score of the sampler's policy, E_{y~q}[...] = 0 up to the within-tail term
(tail mass x within-tail score, small). Then E_q[A grad(log p_y - b)] = E_q[(A - c)·(e_y - q)] for any prefix-measurable
c: the drift c*(q-p) is removed exactly, for any advantage offset, with no weights, no clipping and no extra variance
from ratios (b is a function of the prefix only, so it is a valid baseline in the usual sense: unbiased relative to the q-score).
The remaining gradient is the policy gradient of q expressed in logits shared with p (logits of q = logits of p + small
perturbation, so grad wrt p's logits ~ grad wrt q's logits), which is the signal that is actually correct to follow; the
bias-ful part -c grad KL(q||p) is gone, so errors no longer compound through the refreshed sampler.

Properties: active by default (uses top-k in the batch automatically), works with correction=None. If an ISCorrection
with token level is passed, the weight w=p/q already centres the score under p (E_q[w score]=E_p[score]=0), so the
control variate is switched off in that case (adding it with clipped weights re-introduced collapse in my CPU tests).
If the batch has no sampler top-k, the loss falls back to plain REINFORCE.

CPU mini task (extra-token vocabulary, noise 0.3): plain PG collapses to reward ~0.09 (entropy collapse, grad norm ~0);
the centred loss reaches 0.72-0.82 over seeds; TIS clamp ~0.72.

## Configuration
rl.is.level=null
