# Sampler mismatch in REINFORCE: what goes wrong and the fix

## Summary

- The current loss, with no correction, is a score-function estimator whose samples come from the sampler q, not the trainer p. For a fixed prefix h the expected update is sum_x q(x|h) Q(h,x) grad log p(x|h). On-policy (q = p) the constant part of Q has zero expected gradient; under mismatch it does not. A constant advantage c then contributes c * E_q[grad log p] = -c * grad KL(q || p).
- For bad prefixes (c < 0) this term pushes the trainer away from the sampler's samples with a gradient that never decays. The sampler is then refreshed from the trainer, so the drift compounds. The sampled-token estimate of KL(q || p) (`kl_samp_train`) grows without bound under the current loss.
- Fix: weight each token by the importance ratio p/q. This makes the expected update equal to the on-policy one for every prefix, and it makes the per-sample gradient vanish as the trainer suppresses a sampled token. Because p/q has heavy tails, the weight is capped from above at C = 10 (`min(p/q, 10)`). Two errors remain: the cap, which only biases tokens whose trainer/sampler ratio exceeds 10, and the continuation bias of section 7.
- The fix is the default in `losses.py`: `loss_fn_rl` uses this correction whenever `correction=None`. No configuration changes are needed.

## 1. Setting

Fix a training step. The sampler distribution q(.|h) is a fixed function of the snapshot (trainer weights plus the mismatch: noise, quantization, stale weights, quantized KV cache), so it does not depend on the trainer parameters theta. The trainer is p_theta(.|h). A completion y ~ q gets a sequence-level advantage A(y) (group-centred reward). The loss, written per token with weights w_t, is

  L = -(1/N) sum_b A_b sum_t w_bt log p_theta(y_bt | y_b,<t),

where N is the number of completion tokens. The current code has w_t = 1.

## 2. Expected update at one position

Condition on the prefix h and look at one token. Define the sampler-side advantage at a token as Q_q(h,x) = E_q[A | h, y_t = x], where the continuation after x is also drawn from q. The expected per-token gradient of L (the descent direction is minus this) is

  E_q[ w(x) A grad log p(x|h) | h ] = sum_x q(x|h) w(x) Q_q(h,x) grad log p(x|h).

The on-policy quantity, for q = p, is

  G_p(h) = sum_x p(x|h) Q_p(h,x) grad log p(x|h) = grad_theta E_p[A | h] (policy-gradient theorem).

Two facts matter.

(i) Score identity. sum_x p(x|h) grad log p(x|h) = grad sum_x p(x|h) = 0. Under q = p a constant added to the advantage therefore changes nothing in expectation: this is why baselines are free on-policy.

(ii) Score under the sampler. Because q does not depend on theta, sum_x q(x|h) grad log p(x|h) = -grad KL(q(.|h) || p_theta(.|h)). In general this is nonzero.

In logit space, grad_z log p(x|h) = e_x - p, so the uncorrected update is

  G_q(h) = q o Q_q - (q . Q_q) p,

compared with the target G_p(h) = p o Q_p - (p . Q_p) p.

## 3. Where the mismatch enters

Write the advantage as Q = c + Q~ with c constant over x. The uncorrected update splits as

  G_q(h) = c (q - p) [in logit space] + sum_x q Q~ grad log p,

and the first term equals -c grad KL(q || p) in parameter space. Since the group-centred advantage has zero mean over a group but not zero conditional mean at most prefixes, c is generally nonzero, so this bias is present throughout training. Two sources of error follow:

- (a) Constant-advantage bias: c * E_q[grad log p] = -c grad KL(q || p). This is the term that the on-policy baseline invariance hides.
- (b) Continuation bias: Q_q(h,x) differs from Q_p(h,x) because the tokens after x are drawn from q, not from p.

## 4. Why it compounds

Gradient descent on L moves theta by -eta * grad L. The constant-advantage term adds to grad L the piece -c E_q[grad log p] = c grad KL(q || p), so

  theta <- theta - eta c grad KL(q || p).

For c < 0 (a bad prefix) this is gradient ascent on KL(q || p): the trainer is pushed away from the sampler's samples. The push does not decay as the trainer suppresses a sampled token x. For a softmax, d log p(x) / d z_x = 1 - p(x) tends to 1 as p(x) goes to 0, so the unlikelihood gradient on sampler-typical tokens stays O(1) and KL(q || p) is unbounded. The sampler is then refreshed as trainer plus the same fixed noise, so the mismatch is re-applied to a trainer that has already drifted, and the effect accumulates.

Empirically, with the current loss on the CPU proxy (dev/mini.py, noise 0.6, 1500 steps), kl_samp_train (a sampled estimate of KL(q || p)) grows linearly from about 3 to about 44, and the reward falls to about 0.04 (collapse).

## 5. The fix: token-level importance weight p/q

Let w(x) = p(x|h)/q(x|h). Then

  E_q[ w(x) f(x) | h ] = sum_x q(x) (p(x)/q(x)) f(x) = sum_x p(x) f(x) = E_p[f | h].

Applied to the constant part, the bias vanishes exactly, for any c:

  E_q[ w(x) c grad log p(x|h) ] = c sum_x p(x) grad log p(x|h) = c * 0 = 0.

Applied to the full advantage, the update per prefix becomes sum_x p(x) Q_q(h,x) grad log p(x|h), which equals the on-policy G_p(h) except for the continuation bias (b) discussed in section 7.

The weight also acts on the unlikelihood drift directly. The per-sample gradient is w grad log p(x) = grad p(x) / q(x), which is proportional to p(x). Once the trainer suppresses a sampled token, its gradient vanishes, so the drift in section 4 stops. The current loss, with w = 1, has the gradient grad log p(x), which does not vanish.

## 6. Variance and the cap

The unclipped ratio has heavy tails: E_q[w^2] = sum_x p(x)^2 / q(x) can be very large when q(x) is small. The estimator is unbiased but noisy. We therefore cap the weight from above at C:

  w_C(x) = min(p(x)/q(x), C).

The bias of the capped estimator is

  E_q[w_C f] - E_p[f] = -sum_x (p(x) - C q(x))^+ f(x),

which is nonzero only on tokens with p/q > C. Its size is at most the trainer's mass on those tokens times the magnitude of f there. A smaller cap gives more bias; a larger cap gives more variance. In the proxy runs, the result got worse as C decreased (C = 1.2 < 2 < 5 < 10), and no cap (C = infinity) did much worse when the vocabulary had many unlikely tokens (0.38 against 0.61). We use C = 10.

The lower side is not capped. For p/q near zero the weight is already small and self-limiting, so no bias is introduced there.

The constant part of the cap bias can be removed exactly if the sampler's full vocabulary log-probabilities are available (`sampler.vocab_logprobs=full`). Adding sum_y (p(y) - C q(y))^+ log p(y), with the coefficient stopped, cancels the constant-advantage part of the cap bias. We tested this in the proxy and it made no measurable difference, so it is not in the default loss.

## 7. What remains

- Continuation bias. Per-token weights correct the current token only. The continuation after it is still drawn from q, so Q_q differs from Q_p. The full reward-to-go correction uses w_t = prod_{s >= t} p_s/q_s. We tried it (unclipped, and clipped on the suffix product). Unclipped, one of two seeds collapsed to zero reward and the other stayed near 0.14 eval; clipped, it was worse than the per-token cap (0.44 and 0.26 eval against 0.71 and 0.61). We therefore do not use it.
- Residual drift. With the cap, kl_samp_train still grows slowly (about 1.7 to 3.2 over 1500 steps in the proxy). The reward kept rising and did not collapse.
- The CPU proxy is a small model. The numbers below are evidence for the direction of the effect, not a claim about Qwen3-0.6B.

## Evidence (dev/mini.py, eval reward at the end of training)

| setting | PG (current) | p/q uncapped | cap 2 | cap 5 | cap 10 (default) |
| --- | --- | --- | --- | --- | --- |
| noise 0.6, seed 0, 600 steps | 0.37 | 0.66 | 0.69 | 0.70 | 0.71 |
| noise 0.6, seed 1, 600 steps | 0.40 | 0.50 | 0.57 | 0.60 | 0.63 |
| noise 0.6 + 200 extra tokens, seed 0 | 0.33 | 0.38 | 0.52 | 0.57 | 0.61 |
| noise 0.6, 1500 steps, seed 0 | 0.04 | - | 0.74 | - | 0.82 |
| noise 0.3, 600 steps | 0.80 | - | - | - | 0.80 |
| noise 1.0, 600 steps | 0.23 | 0.47 | 0.50 | 0.49 | 0.51 |

## Configuration

None. The corrected loss is the default when `correction=None`, and no `rl.is.*` or `sampler.*` keys are needed.
