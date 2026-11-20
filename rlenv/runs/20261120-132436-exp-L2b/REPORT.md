# Report

## What is wrong
At one position with a fixed prefix, let the next token a ~ q (sampler), p the trained policy, z the logits, and A the advantage.
The current loss is -A * logp(a), with optional clipped or dropped weights w(r), r = p/q. Its expected logit-gradient is

E_q[A w grad_z logp(a)] = sum_a q(a) A(a) w(a) (e_a - p).

Split A(a) = Abar + (A(a) - Abar), where Abar is the mean advantage at this prefix. With no weight (w=1):

Abar * sum_a q(a)(e_a - p) = Abar (q - p).

On-policy (q = p) this term is exactly 0. This is why a baseline or group-centering is valid there:
E_p[grad logp] = sum_a grad p(a) = grad 1 = 0.
For q != p it is nonzero and is not zero-mean noise. It is a systematic drift Abar*(q-p) = -Abar * grad CE(q, p).
- If Abar > 0, p is pulled toward q.
- If Abar < 0, which is common after centering at hard prefixes, p is pushed away from q. The loss then maximizes cross-entropy to q, and this is unbounded.
The sampler is refreshed from the trainer every step, so the drift compounds. The trainer drifts away from q, which creates a new q != p, and so on. The result is entropy blow-up and collapse.
Group-centering only makes the advantage zero-mean across sequences. It does not make it zero-mean per token and prefix, so the leak remains.

Clipping or dropping tokens (w = min(r, c), or w = 0 outside a band) gives sum_a q w (e_a - p) = sum_a min(p, cq)(e_a - p). This is nonzero whenever any token is clipped, so the same drift remains. It is also state-dependent: it is biased toward the tokens where p and q disagree. A stronger clip therefore removes more of the cancellation and can make things worse.

## Fix
Use the exact token-level importance weight w = sg(p/q) with no clipping or dropping. This is the default in `loss_fn_rl` when `correction=None`.
The surrogate gradient is A * (p/q) * grad logp = A * grad p / q. Then:

E_q[A (p/q) grad_z logp] = sum_a A(a) p(a) (e_a - p) = E_p[A grad logp].

This is the on-policy gradient, whatever q is. For the Abar part it gives Abar * sum_a grad p = 0. So the leak is removed and the baseline property is restored exactly.
The estimator is unbiased for any q with support covering p. The variance cost is Var(r), which is small for small mismatch. The weight is computed in log space and clamped only at e^{±20} for numerical safety.
A user-supplied `correction` is still honored.

## Configuration
rl.is.level=token
rl.is.outside=clamp
