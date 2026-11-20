# Sampler/trainer mismatch: a biased score, and a q-centered fix

## What is wrong
Samples come from q, the trainer is p. The loss gradient is g = E_q[ A(y) * grad log p(y) ] (per token, A = group-centered advantage).
On-policy (q=p) the score has zero mean, sum_a p(a) grad log p(a) = 0, so any baseline that depends only on the prefix h drops out and the
update is the policy gradient. In logit space the per-token gradient is A (e_y - p).

For q != p the score is no longer zero-mean: E_{a~q}[grad log p(a)] = grad_theta sum_a q(a) log p(a) = -grad CE(q,p) != 0. Write A(y) = V(h) + (A - V(h)), V(h)=E[A|h].
Then
  g = E[ V(h) * (q-p)·grad logp ]  +  Cov_q(A, grad log p).
The second term is the intended signal. The first is a systematic bias: in logit space it is V(h)(q(.|h) - p(.|h)).
In prefixes with positive expected advantage, p is pulled toward q (self-reinforcing the sampler's quirks, e.g. noise-inflated
tail or over-sharp tokens); in negative-V prefixes it is pushed away from q. Because the sampler is rebuilt from p each step, the bias is
applied every step and compounds: it is not zero-mean noise, it is a drift with a fixed sign set by q-p. (Group centering only zeros V at the root; V(h) is nonzero at deeper tokens, and token-sum normalization weights sequences by length.)
Clipping/dropping tokens by p/q keeps preferentially the tokens with q>=p and so makes the (q-p) bias worse; IS weights p/q make it unbiased
(sum_a q (p/q) grad log p = 0) but with high variance, and every clip reintroduces bias.

## Fix: center the score under q (control variate)
Use the score  s~(y) = grad log p(y) - E_{a~q} grad log p(a). Then E_q[s~] = 0 exactly for any q and p, so the V(h) term vanishes just as on-policy,
and E_q[A s~] = Cov_q(A, grad log p) (no bias term, no importance weights, no variance blow-up).
In logit space the gradient is A (e_y - q): the REINFORCE gradient with respect to the sampler's own distribution, which is zero-mean under q.
Surrogate implemented (q detached):
  loss = - A * [ log p(y) - sum_{a in topk_q} q(a) log p(a) - tail_q * log P_tail ],   P_tail = 1 - sum_{a in topk_q} p(a), tail_q = 1 - sum_{topk} q(a).
q is known from the sampler's top-20 logprobs (`sampler_topk_*` in the batch, which both trainers provide). The mass outside top-k is assumed to be
distributed like p there; then sum_{a in tail} q(a) grad log p(a) ~= tail_q * grad P_tail / P_tail = tail_q * grad log P_tail. This is exact when q=p on the tail,
and when q=p everywhere the correction term has zero gradient, so the loss reduces to ordinary REINFORCE on-policy.
If the top-k keys are missing the loss falls back to plain REINFORCE. An optional ISCorrection weight still multiplies the centered score.
In the CPU dev setting (noise 0.3, 600 steps, seed 1) eval reward was 0.66 with the old loss and 0.79 with this one.

## Configuration
rl.is.level=null
