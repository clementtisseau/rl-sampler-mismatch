# Report

## Problem
Fix a prefix and look at one position. Let z be the logits, p = softmax(z) the trainer, q the sampler, x ~ q, and A(x) the advantage assigned to the token. The loss gradient is -A ∇_z log p(x) = -A (e_x - p).

- On-policy (q = p): E_p[A(e_x - p)] = ∇_z E_p[A]. This is the policy gradient. Any constant baseline b adds b·E_p[e_x - p] = 0, so baselines and group-centering are free.
- Mismatched (q != p): E_q[A(e_x - p)] = Σ_x q(x)A(x)(e_x - p). Write A = b + (A - b), with b = E_q[A]. Then E_q[A(e_x - p)] = E_q[(A-b)(e_x - p)] + b(q - p). The extra term b(q - p) = b ∇_z E_q[log p] = -b ∇_z CE(q, p) does not vanish. It is a systematic bias, not zero-mean noise.
  - If b > 0 it pulls p toward q (the sampler's errors get imitated).
  - If b < 0 it pushes p away from q, which raises entropy and destroys the distribution.
- b is not zero in practice. The loss sums over tokens divided by the total token count, so each sequence is weighted by its length. Failed rollouts are longer and have negative group-centered advantage, so token-weighted advantages have net negative mean. Clipping or dropping tokens by ratio changes which tokens contribute but does not centre b. Dropping or clamping with weight w(p/q) gives E_q[w A (e_x - p)], which is biased unless w = p/q exactly. Token-level clipping also removes the tokens that carry the correction.
- The sampler is refreshed from the trainer at every step, so the bias is applied again at every step and compounds.

## Fix (active by default in losses.py)
1. Exact token-level importance weight w = p/q (detached) on the score. Then E_q[(p/q) A (e_x - p)] = E_p[A(e_x - p)], which is the on-policy gradient at that position, so the b(q - p) term is cancelled. The clamp is only a loose variance guard at r <= 10, so it almost never triggers. It is not a trust-region clip.
2. Each sequence is weighted 1/len instead of its token count. Group-centered advantages then sum to zero over sequences, so the net mean advantage seen by the loss is zero. This removes the sign-definite source of b (long failures) that drives entropy up. Together with the exact weight, the leftover bias is second order.

## Caveat
The IS weight only fixes the conditional distribution at each position. The prefix distribution is still q, not p, so a residual error remains. It is small when q is close to p.

## Configuration
rl.is.level=token
rl.is.high=10.0
rl.is.outside=clamp
