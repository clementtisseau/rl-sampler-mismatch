# Sampler mismatch: a baseline bias in REINFORCE, and its removal

## What is wrong with the current loss

Notation: state s (prompt plus prefix), action a (token), p = trainer policy, q = sampler policy,
A = sequence advantage, Q(s,a) = E[A | s,a]. The loss is -(1/N) sum_i A_i sum_t log p(a_it | s_it).

On-policy (q = p) the expected gradient at a state is
sum_a p(a) Q(s,a) grad log p(a).
Since sum_a p(a) grad log p(a) = grad sum_a p(a) = 0, any constant b(s) can be added to Q at no cost in expectation.
Baselines (and group centering) only reduce variance.

With q != p the loss gives, at state s,
g_q(s) = sum_a q(a) Q(s,a) grad log p(a),
and this identity fails:
sum_a q(a) grad log p(a) = -grad_theta KL(q || p(.|s)) != 0.
Write Q = V(s) + Adv(s,a). Then
g_q = V(s) * (-grad KL(q||p)) + sum_a q Adv grad log p.
The first term is a bias that does not depend on the reward signal. It is
- a forward-KL pull of p toward q when V > 0, and
- a push of p away from q when V < 0. This is unbounded: it moves mass onto tokens that q rarely samples, such as the tail of the vocabulary.

It is nonzero whenever the advantages averaged over the sampled actions at a state are nonzero. Two effects cause that.
1. Plain rewards have V(s) = E[A|s] far from 0. Group centering removes this at the prompt level.
2. Group centering only gives sum_i A_i = 0 per prompt. The token-level loss weights sequence i by its length: sum_i A_i L_i != 0. At deeper positions the alive set is a biased subset of the group. Long rollouts, such as rambling failures, typically have A < 0. The effective per-state V is then negative and the loss steadily pushes p away from q.

The bias compounds because the sampler is refreshed from the trainer each step: q_{k+1} = p_k + mismatch. p moves in the direction -grad KL, i.e. away from the previous sampler. The result is entropy growth, tail mass and drift, and finally collapse.

The second-order part of the error is sum_a (q-p) Adv grad log p. It is multiplied by the small centered advantage and is not removed here.

Importance weights w = p/q repair the identity, since sum_a q (p/q) grad log p = 0. Clipping or dropping tokens breaks it again, because
sum_a q (min(p/q, c)) grad log p = sum_a q (min(p/q, c) - p/q) grad log p != 0.
That residual is itself a bias with a sign set by where the clipping acts, which explains why stronger token clipping can make things worse. Naive IS has a heavy-tailed variance.

## Modification

The bias term depends only on the part of the advantage that is constant over actions at a state. We remove it exactly, with no q information and no clipping.
For each prompt group g (rows sharing the same prompt tokens) and each position t, the per-token advantage is

A_it = A_i - mean_{j in g, alive at t} A_j.

Then for each (g,t) the weights sum_i A_it over the sampled alive rows equal 0. The state-constant component V(s) is removed
to the extent the sample mean estimates it, and the drift term V * grad KL(q||p) vanishes.
- Group centering already does this at t = first position when all lengths are equal. Per-position centering extends it to variable lengths.
- The contrast between actions is kept. At a position where some rows emit EOS and others continue, both are in the alive set, so the contrast between them survives.
- This is a position-dependent baseline: it is a valid baseline on-policy, so it adds no bias when q = p.
- It is combined with whatever correction is passed. With correction=None it is still active.
- Rows alone in their group (group size 1) are left unchanged. A lone survivor at a position gets A = 0.

Implementation: `position_centered_advantages` in `losses.py` hashes the prompt tokens (positions before the first completion token), builds a [B,B] same-group matrix, and computes the alive-weighted mean per position with a matmul. The loss then uses per-token advantages inside the chunked computation.

In the CPU dev setting (noise 0.5, plus_minus rewards), the plain loss collapses (reward goes to 0). With centering, reward stays at about 0.6, the same as under group-centered rewards.

## Configuration

rl.is.level=null
