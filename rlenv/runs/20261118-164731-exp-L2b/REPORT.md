# Sampler/trainer mismatch in REINFORCE

## What is wrong
Fix a prefix and a next-token distribution. Trainer p, sampler q, token x ~ q, advantage A(x).
The plain loss -A * log p(x) has expected gradient
  g_PG = sum_x q(x) A(x) grad log p(x).
Write A(x) = b + (A(x) - b), b = E_q[A].
On-policy (q = p): sum_x p grad log p = sum_x grad p = grad 1 = 0, so any baseline/constant shift
(group centering, which is only exact per sequence, not per prefix) contributes nothing; g_PG is the policy gradient.
Mismatched (q != p): sum_x q grad log p = sum_x (q/p) grad p != 0. In logit space grad_z log p(x) = e_x - p, so
E_q[grad_z log p] = q - p. Hence
  g_PG = (policy-gradient term) + b * (q - p)   (logit space),
i.e. the baseline/mean-advantage no longer cancels, and each step pushes p toward (b>0) or away from (b<0) the sampler q, weighted by
the prefix-level mean advantage. This is a systematic bias, not zero-mean noise: it is a fixed-direction drift (like gradient of
b*cross-entropy(q,p)), and since the sampler is refreshed from the trainer each step, q tracks p and the error compounds
(negative-b prefixes in particular push mass away from q's modes, raising entropy/collapse).
Token-level clipping/dropping w = clip(p/q) makes it worse: sum_x q w(p/q) grad log p = sum_x w grad p, which is no longer zero
for the constant part of A, so clipping reintroduces the bias, asymmetrically (it removes exactly the tokens where p>>q).

## Fix
Use the detached token importance ratio w = p/q with no clipping or dropping, as the default:
  loss = -A * sg(p/q) * log p  =>  grad = -A grad p / q.
Then for any A, E_{x~q}[w A grad log p] = sum_x A(x) grad p(x) = sum_x p(x) A(x) grad log p(x) = the exact on-policy gradient,
and for constant A=b it equals b * grad sum_x p = 0. So baselines/group centering cancel again and the expected update is
independent of q (provided supp p within supp q). If q = p, w = 1 and the loss reduces to the original one.
Only the log-ratio clip at +-20 (numerical safety) remains. The price is variance when p/q is large; no bias-variance trade through
clipping is made since clipping reintroduces the q-dependent bias above.

## Configuration
rl.is.level=token
rl.is.low=0.0
rl.is.high=inf
rl.is.outside=drop
rl.is.sign=both
