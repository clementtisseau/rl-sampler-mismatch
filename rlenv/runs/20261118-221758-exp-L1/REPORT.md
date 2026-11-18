# Sampler/trainer mismatch in REINFORCE

## What is wrong
Rollouts come from q, the trainer is p. The loss uses the sampled-token score A * grad log p(y). In logit space,
grad_z log p(y) = e_y - p(.|s). It has zero mean under p, which is why a baseline or group centering is harmless:
E_p[e_y - p] = 0. Under q-sampling the mean is

  E_q[e_y - p] = q - p != 0.

So every token carries a systematic term (advantage-independent part) that pushes p along (q-p). It is not zero-mean noise.
Group centering only cancels it if the term were identical for all samples (it is not: sequences differ in length and
token-sum normalisation weights them unequally, and with plus/minus rewards the mean advantage is not 0). Because the sampler is
refreshed from the trainer, q = p + perturbation each step and the bias acts every step and compounds: the first-order
effect is a drift that, e.g., puts mass on tail tokens or (when negative-advantage samples dominate) pushes down sampled
tokens and sprays mass over the unlikely vocabulary, collapsing the policy. Token-level IS weights p/q fix the mean in
principle (E_q[(p/q)(e_y-p)] = 0), but once the weight is clipped/dropped (TIS, MIS, PPO-style bands) the
mean is no longer zero: E_q[min(p/q,C)(e_y - p)] != 0, so clipping reintroduces a bias, and stronger clipping
is worse (more bias). That is why those methods only slow the degradation.

## Fix: exactly centered, clipped, full-vocabulary score
Let the sampler record its full next-token log-probs q(.|s) (`sampler.vocab_logprobs=full`). With w_v = min(p_v/q_v, C)
(C = 3, bounds the variance), define the logit-space gradient

  g = w_y (e_y - p) - m,   m = E_{v~q}[w_v (e_v - p)] = (q_v w_v)_v - p * sum_v q_v w_v.

m is computable in closed form from the vocabulary. Then E_{y~q}[g] = 0 exactly, for any clipping, so the
advantage-independent bias term vanishes by construction, while the clipping keeps variance bounded. The remaining
signal is E_q[A g], the advantage-correlated part, which equals the policy gradient where p/q < C (and a
clipped, still correct-direction version elsewhere). The surrogate is loss = -A * sum_v sg(g_v) log p_v, whose gradient
w.r.t. logits is -A g (sum g = 0 so log-softmax normalisation is irrelevant). Nothing needs to be switched on: the loss
uses the vocabulary path whenever `sampling_vocab_logprobs` is in the batch; `correction=None` works. If the batch
lacks it, the loss falls back to the plain sampled-token score (the old behaviour), so the configuration below is
needed for the fix to be active.
On the dev CPU setting (noise 0.3, 200 extra tokens) the baseline collapses to reward 0 while this loss reaches 0.86.

## Configuration
sampler.vocab_logprobs=full
