# Sampler/trainer mismatch in REINFORCE

## What is wrong
Without correction the loss is L = -sum_t A_i log p(x_t | prefix), tokens drawn from q. Fix a prefix, token x ~ q. The expected gradient w.r.t. logits z (p = softmax z) is

E_q[A(x) grad_z log p(x)] = sum_x q(x) A(x) (e_x - p).

Write A(x) = b + a(x), where b is the prefix-conditional mean advantage (a baseline: group centering, length effects, etc.). Then

E_q[...] = sum_x q(x) a(x)(e_x - p)  +  b (q - p).

On-policy (q = p): sum_x p (e_x - p) = 0, so the baseline term vanishes, which is why baselines/group centering are free and the update is the true policy gradient. When q != p the term b(q - p) is non-zero: it is b * grad E_q[log p] = -b * grad KL(q||p). It is a systematic drift that is independent of reward signal: for b>0 it pulls p towards q, for b<0 (typical for failed, long rollouts: group-centered advantages are zero-mean per sequence but not per token, and failures carry more tokens and negative b) it pushes p away from q, raising mismatch/entropy. The reward term is also evaluated with q-weights, sum q a (e_x-p), not the policy gradient sum p a (e_x - p). Because the sampler is refreshed from the trainer, the pushed-away p makes q-p larger next step: positive feedback, hence compounding collapse. This is a bias, not zero-mean noise, so averaging doesn't help.

## Fix
Weight each token by the detached ratio w = p(x)/q(x), using the sampler logprobs already in the batch:

E_q[w A (e_x - p)] = sum_x p(x) A(x)(e_x - p) = true policy gradient of the current p,

and the baseline term is b * sum_x p(e_x - p) = 0 exactly again, for any q with support covering p. So baseline invariance, which the loss silently relied on, is restored. (Equivalently the surrogate -A * p/q has gradient A r grad log p, and E_q[grad r] = grad sum p = 0.)

Clipping w (token-level, tight band, or dropping tokens) breaks this identity: the leftover b * sum_x q(x)(w_clip - r)(e_x - p) is again a bias, so tight clipping merely rescales the drift, and dropping tokens with r small/large, particularly for one sign of A, makes b effectively non-zero. Therefore `loss_fn_rl` with default arguments uses the token-level ratio w = p/q with only a loose clamp at 50 (purely to avoid variance blowups/overflow; the biased mass is only tokens with p/q>50). If an explicit `correction` is passed it is used as before.

## Configuration
rl.is.level=token
