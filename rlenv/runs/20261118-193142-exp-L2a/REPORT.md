# Why the REINFORCE loss breaks under sampler/trainer mismatch

## Setup
Loss: L = -(1/N) sum_i A_i sum_t log p(y_it | y_i,<t), with y_i ~ q (sampler), p = trainer.
Gradient: g = -(1/N) sum_i A_i sum_t grad log p(y_it | .).

## On-policy (q = p)
Score identity: E_{y~p}[grad log p(y)] = sum_y grad p(y) = grad 1 = 0 at every context. Hence for any
baseline b independent of y_t, E[b * grad log p] = 0: centering the reward is free and only reduces
variance, and the expected gradient is exactly grad J(p) = E[(R - b) grad log p].

## Mismatched (q != p)
E_{y~q}[grad log p(y)] = sum_y q(y) grad log p(y) = -grad_theta KL(q || p) =: -k(q,p) != 0 (sum_y q grad log p = grad of -CE(q,p)).
Write A_i = (A_i - Abar) + Abar. The expected update is
  E[g] ∝ Cov_q(A, grad log p) + Abar_tok * E_q[grad log p],
and the second term is a bias: Abar_tok * grad KL(q||p) is a gradient of a forward KL toward q, scaled by the
(token-weighted) mean advantage. Its sign is set by Abar_tok and it is NOT noise: it does not average out
with more samples. Group-centering makes the per-sequence mean zero, but the loss sums over
tokens, so sequence i contributes A_i * L_i; sum_i A_i L_i != 0 whenever lengths correlate with reward
(typically failing rollouts are longer -> net negative weight). Negative mean weight means the
objective pushes p away from q (maximizes KL(q||p)) on every token: an unlikelihood term that
raises entropy / leaks mass to junk tokens, which degrades the policy; the sampler is then refreshed
from the trainer so q follows p and the error compounds. Since the bias is a mean-weight times a
q-vs-p gradient, it exists for any q != p (noise, quantization, KV quant, staleness), and
token-level ratio clipping/dropping only reweights tokens (the removed tokens are exactly where
q and p disagree, making the surviving E_q[grad log p] still nonzero and the weights asymmetric),
so it cannot remove it; stronger clipping can worsen the asymmetry.

## Fix
Use token-weighted centered advantages in the loss (default, no flag):
  A'_i = A_i - (sum_j A_j L_j) / (sum_j L_j),   so  sum_i A'_i L_i = 0.
Then the total weight on log-probabilities of sampled tokens is zero, the term Abar_tok * E_q[grad log p]
vanishes (exactly in the batch-aggregate sense), and the remaining expected update is the covariance
term Cov(A, grad log p) -- the true policy-gradient signal, which equals the on-policy gradient up to
the (small, signal-bearing) difference between q and p. The shift is
a constant per batch, so on-policy (E_p grad log p = 0) the expected gradient is unchanged:
the modification is a pure variance-neutral baseline there and strictly removes the bias off-policy.
Any IS correction passed in still works, applied on top of the centered advantages.
Caveat: it cancels the common-mode bias, not the context-dependent residual correlated with A.

## Configuration
