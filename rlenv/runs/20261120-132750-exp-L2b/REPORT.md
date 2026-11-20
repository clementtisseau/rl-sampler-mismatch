# Report

## What is wrong
The loss is L = -(1/N) sum_i A_i sum_t w_t log p(y_t|prefix), y_t ~ q, with w=1 by default (or a clipped/dropped ratio). Fix a prefix s and take the expectation over the next token y ~ q. The gradient is

  g(s) = A * E_q[ w * grad log p(y) ].

On-policy (q=p, w=1): E_p[grad log p] = sum_y grad p(y) = grad 1 = 0. So a prefix-level advantage that is not correlated with y contributes nothing. Only the correlated part (the real policy gradient) survives, and group-centering/baselines are harmless.

Mismatched (q != p, w=1):
  E_q[grad log p] = sum_y q(y) grad log p(y) = sum_y (q(y)-p(y)) grad log p(y) = -grad_theta KL(q || p),
which is nonzero. The update therefore contains the term A(s) * (-grad KL(q||p)): for A>0 it pulls p toward q, for A<0 it pushes p away from q (raising the KL and flattening/destabilising p). Group-centered advantages sum to zero over a group, but at a deep prefix there is a single sample and A(s) is the sequence advantage, so this term does not cancel; failed (A<0) completions are also usually longer, so the token-weighted mean advantage is negative. The result is a systematic drift, not zero-mean noise: p is pushed away from the sampler's distribution at every step, and since the sampler is refreshed from p (or is a perturbation of it) the bias compounds -> entropy blow-up/collapse. Clipping or dropping tokens with weight w=min(r,C) (or 1[r in band]) changes the bias to sum_y q(y)(w(y)) grad log p(y) which is still nonzero; tighter clipping moves w further from p/q and enlarges the bias (explaining why stronger token-level clipping can worsen things).

## Fix
Use the token-level importance weight r = p/q (detached) as a default:
  E_q[ r grad log p(y) ] = sum_y p(y) grad log p(y) = 0,
so the nuisance term vanishes exactly for any prefix-level A, and the expected update is A(y)-correlated policy gradient E_p[A grad log p] (exactly unbiased for the on-policy objective). The only change is that the weight is not clipped tightly: weights are capped at r<=8 (clamped, not dropped) purely to bound variance in the extreme tail; the residual bias is confined to tokens where p>8q, and is small. Log-ratio is also clipped at e^{+-20} for numerical safety. The default correction is applied when `correction=None`; an explicit correction still overrides it.

## Configuration
rl.is.level=token
rl.is.high=8.0
rl.is.outside=clamp
