# Sampler/trainer mismatch in REINFORCE

## What is wrong
Rollouts x ~ q (sampler) but the loss uses the score of the trainer p:
g = E_q[ A * sum_t grad log p(x_t|x_<t) ].
Under p, E_p[grad log p(x_t)] = sum_v grad p(v) = 0, which is why REINFORCE needs no
baseline for unbiasedness and why group centering is "free". Under q != p:

  E_{x~q}[grad_theta log p(x)] = sum_v q(v) grad log p(v) = -grad_theta CE(q, p) != 0.

So every sampled token carries a nonzero mean gradient term -grad CE(q,p), which
multiplies whatever advantage mass is not exactly cancelled (A is a sequence quantity,
tokens/contexts differ across a group, plus_minus rewards have nonzero mean, and
length differs between solved/unsolved rollouts). With mean advantage Abar this adds
-Abar * grad CE(q,p): for Abar<0 the trainer is pushed *away* from q (increasing
CE(q,p)), for Abar>0 pulled toward q, i.e. a systematic bias, not zero-mean noise.
It is a deterministic drift in logit space of size (q-p), and because the sampler is
rebuilt from the trainer each step, q-p keeps being regenerated and the drift
compounds (entropy/tail-mass blow-up, then collapse). Token-level IS weights and
clipping reweight the samples but leave this term (and clipping adds its own bias,
removing exactly the high-|q-p| tokens that carry the signal).

## Fix
Use a control variate that is exactly the mean of the score under the sampling law,
with q detached:

  score_t = log p(x_t) - b_t,   b_t = sum_v sg(q_t(v)) log p_t(v).

Then E_{x_t~q}[grad score_t] = sum_v q grad log p - sum_v q grad log p = 0 identically,
for any q, so the mismatch-induced term -grad CE(q,p) cancels per position and per
context, independent of the advantage. The remaining gradient is
E_q[A * (grad log p(x) - E_q grad log p)] = Cov_q(A, grad log p), which is the
advantage-weighted covariance (the REINFORCE signal) and equals the true policy gradient
when q = p (b has zero gradient then). It costs no extra variance: b is a baseline that
does not depend on x_t, and it is computed in the same rematerialized vocab chunk
(sum over V of q*logp, using the sampler's recorded full-vocab logprobs; with top-k
logprobs the sum is over the k ids, renormalized by their mass). In logit space the
gradient per token becomes (e_x - q) instead of (e_x - p): the zero-mean "e_x - q"
is the centered score for the distribution actually sampled.

The modification is on by default (active whenever the sampler's vocab logprobs are in
the batch, with correction=None or any correction, which still multiplies the score).
In dev/mini.py with noise 0.5, plus_minus: baseline loss collapses to reward 0 while
this loss reaches ~0.71; at noise 0.8 baseline collapses, this stays at ~0.4.
Without vocab logprobs in the batch only the sampled-token logprob is known and the
loss falls back to the plain score, so the configuration below is needed.

## Configuration
sampler.vocab_logprobs=full
