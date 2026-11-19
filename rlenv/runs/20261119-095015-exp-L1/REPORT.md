# Sampler/trainer mismatch in REINFORCE

## What is wrong with the current loss
Rollouts y ~ q (sampler), but the loss is -(1/N) sum_i A_i sum_t log p(y_it) with p the trainer. Its expected gradient at one position (logits z, state s) is
  E_{y~q}[A(y) grad_z log p(y)] = sum_v q_v A_v (e_v - p),
whereas the policy gradient we want is sum_v p_v A_v (e_v - p).
The difference is not zero-mean noise. Even for A = const, E_q[grad log p] = q - p != 0 (whereas E_p[grad log p] = 0), so every
update contains a systematic drift of p towards q (and away from p-mass the sampler under-visits) that group-centering cannot cancel
(it is zero-sum across a group only, not per token or per state, and sequences differ in length and tokens). Because the sampler is
rebuilt from the trainer each step, the bias enters every step and compounds. Tail tokens that the sampler inflates (q >> p)
are sampled often, receive gradient as if they were on-policy, and the loss drifts toward collapse.
Clipping/dropping tokens with a band on p/q changes the weighting by a state/token-dependent factor, i.e. adds a second bias
(it removes exactly the tokens carrying the correction), which is why stronger clipping can worsen it.

## Fix
Use the importance weight w = p/q (detached) per token, active by default:
  E_{y~q}[ (p_y/q_y) A(y) grad log p(y) ] = sum_v p_v A_v (e_v - p),
which removes the drift exactly (in particular E_q[w grad log p] = sum_v p_v grad log p_v = 0 for constant A). Equivalent to
grad of p/q, so a sampled token with q>>p contributes ~0 instead of a full-size push.
Two stabilisers, kept as weak as possible so as not to reintroduce bias:
1. Self-normalisation: divide by sum_t w_t rather than the token count N. The weighted estimator is then a ratio estimator
   (consistent), with no global scale drift when the weights have mean != 1 (mean of p/q over q-samples is 1 only in expectation).
2. A loose cap w <= 10 (and the existing exp(+-20) clip on the log ratio) only to bound variance from rare p >> q tokens;
   it affects only tokens with extreme ratio, unlike tight bands.
The per-token (not per-sequence) weight is used because the product of per-token ratios (sequence level) has exponentially
large variance in length and in my tests collapsed (seq_sum, seq_mean), while the token weight corrects the per-state action
distribution, the part that creates the drift (the state-distribution mismatch remaining is second order).

With `correction=None` this is the default; an explicit ISCorrection still behaves as before.
In the CPU dev setup with many tail tokens (MINI_EXTRA_TOKENS=200) the plain loss collapses to 0 reward, while this loss reaches ~0.67 and keeps improving.

## Configuration
rl.is.level=token
rl.is.high=10.0
rl.is.outside=clamp
