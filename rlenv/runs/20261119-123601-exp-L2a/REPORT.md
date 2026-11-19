# Sampler mismatch and the REINFORCE loss

## What is wrong
The loss is L = -(1/N) sum_i A_i sum_t log p(y_it), with y ~ q (sampler), p the trainer. Its expected gradient is
E_q[A grad log p(y)].
On-policy (q=p) the score has zero mean: E_p[grad log p] = 0, so any baseline b (group centering) is harmless:
E_p[(R-b) grad log p] = grad J.
When q != p, write A = (R - b) and per token
E_q[grad log p(y_t)|prefix] = grad sum_v q_v log p_v = -grad_p CE(q,p) = -grad KL(q||p) + const,
which is nonzero: it is a maximum-likelihood (SFT) pull of p toward the sampler q. So the expected update is
E_q[A grad log p] = Cov_q(R, grad log p) + E_q[A] * E_q[grad log p].
The first term is the (biased but useful) policy gradient; the second is a baseline-times-pull term that no longer
vanishes. Since q is a noised/quantized/stale copy of p refreshed every step, p is repeatedly pulled toward its own
degraded version (typically lower-entropy / distorted), a systematic bias that compounds -> drift and collapse.
Group centering makes sum_i A_i = 0 per sequence, but the loss sums over tokens: the pull is proportional to
sum_i A_i L_i (L_i = length), nonzero whenever advantage correlates with length (e.g. wrong answers are longer or
truncated), so a net push proportional to Cov(A, L) * grad KL remains. Token-level IS clipping/dropping makes
the zero-mean property of w*(e_y-p) (E_q[(p/q)(e_y-p)]=0) fail further, which explains why stronger clipping hurts.

## Fix
Make the baseline term vanish exactly in the batch: replace A_i by
A_i' = A_i - (sum_j L_j A_j)/(sum_j L_j)   (detached),
so that sum_i L_i A_i' = 0, i.e. the token-weighted mean advantage is zero. Then the E_q[A]*E_q[grad log p]
term (per-token average advantage times the pull toward q) is cancelled, and the remaining update is
the covariance term Cov_q(A, grad log p), which depends only on reward differences and is the on-policy
gradient up to the (small) q-vs-p reweighting. This is active by default and works with correction=None
(with a correction, weights are applied afterwards as before).
