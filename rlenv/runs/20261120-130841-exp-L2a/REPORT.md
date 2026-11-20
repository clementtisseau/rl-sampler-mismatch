# Sampler/trainer mismatch in REINFORCE

## What is wrong
Rollouts are drawn from q, but the loss is `-A * log p(y_t | y_<t)` with no correction (default `correction=None`). Take one context with logits z, so p = softmax(z). The gradient of log p(y) with respect to z is e_y - p. Its expectation under the sampler is

  E_{y~q}[e_y - p] = q - p.

On-policy (q = p) this is zero. That is why the score function needs no baseline and why group-centred advantages are valid. Under mismatch it is not zero. The expected update of a token with advantage A is A (q - p), not A * 0, so the sampled-token gradient has a systematic component that does not average out. Summed over a sequence, the expected gradient is

  E_q[A ∇ log p(y)] = sum_y q(y) A(y) ∇ log p(y),

which is not ∇J(p) = sum_y p(y) A(y) ∇ log p(y). It is the gradient of a different objective: it rewards likelihood of q-samples with weight A. Positive-advantage tokens are pulled towards q, and negative-advantage tokens are pushed away from it. Advantages are not zero-mean per context or per token. Longer and failed sequences get a different token-mean advantage, and the 1/n_tokens normalisation weights them unevenly. So the bias has a consistent sign.

The bias is not noise. The sampler is rebuilt from the trainer every step, so q = q(p_theta). The drift moves p, which moves q, and the error compounds. Here the sampler is a perturbed or quantised copy of p, so q has systematically different entropy and tail mass. The drift becomes entropy and likelihood distortion, and eventually collapse.

## Why clipping or dropping does not fix it
Use a clipped weight w = min(r, c) with r = p/q. Then

  E_q[w (e_y - p)] = m - p * sum_v m_v,  with m_v = min(p_v, c q_v).

This is zero only if m is proportional to p. In general it is not, so the bias is only reduced. Tighter clipping makes m deviate further from p, which explains why stronger token-level clipping can be worse. PPO/DAPO-style masks and dropping have the same problem, because they remove exactly the tokens where p and q disagree most.

## Fix
Use the exact per-token importance weight r_t = p_t / q_t, detached, with no clipping or dropping:

  loss = -(1/N) sum_t A r_t log p_t,  with r_t = sg(exp(log p_t - log q_t)).

The sampler logprob of the sampled token is already recorded. The default `correction=None` now means a token-level `ISCorrection(level="token")`. This has no band, so only the numerical clip of log r to ±20 remains.

Derivation. In one context, E_q[r (e_y - p)] = sum_v q_v (p_v/q_v)(e_v - p) = sum_v p_v (e_v - p) = p - p = 0.

The expected update is therefore identical to the on-policy one for any A that depends on the context only through the sampled continuation. The score-function identity E_p[∇ log p] = 0 is restored, and the bias A (q - p) vanishes. Per token, E_q[r A ∇log p] = E_p[A ∇log p].

For a full sequence, the sequence weight is the product of per-token ratios. The per-token form is the standard causal estimator. The weight on token t includes only the ratios up to t, and the remaining future ratios have conditional expectation 1. Using only r_t is therefore the correct estimator for the gradient of the policy-gradient term at t, up to the prefix-ratio terms that this loss drops. Those are the same dropped terms as in the usual token-level policy-gradient approximation. The loss is exact per context and carries no clipping bias.

The cost is variance when r is heavy-tailed. This is why weights are not clipped. Clipping trades a variance problem for a bias that compounds over training. In the regime studied here (mild noise or quantisation), r is concentrated near 1 and the variance is moderate.

## Configuration
rl.is.level=token
rl.is.high=inf
rl.is.outside=drop
rl.is.sign=both
