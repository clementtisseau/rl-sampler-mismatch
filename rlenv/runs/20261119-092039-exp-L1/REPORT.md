# REINFORCE under sampler mismatch: self-normalised token ratio weights

## What is wrong with the current loss

Let p = p_theta (trainer), q = sampler, y ~ q a rollout, A its advantage, and
h(y) = A(y) sum_t grad log p(y_t | y_<t) the per-rollout score term. The loss
in losses.py with correction=None is L = -sum A sum_t log p(y_t) / N, so its
expected gradient is

    g_q = E_{y~q}[ h(y) ].

The quantity we want is the gradient of the reward under the trainer,

    g* = E_{y~p}[ h(y) ] = grad_theta E_p[A].

These differ. Write rho_t = p(y_t | y_<t) / q(y_t | y_<t), with log-ratio
l_t = log rho_t = logp_t - logq_t (already returned by the sampler and the
trainer), and rho(y) = prod_t rho_t = p(y)/q(y). Then exactly

    g* = E_q[ rho(y) h(y) ],     g_q = E_q[ h(y) ] = E_p[ h(y) / rho(y) ].

So the uncorrected loss is the correct gradient under the wrong measure: each
rollout is weighted by q/p = 1/rho instead of 1. To first order in the
mismatch,

    1/rho(y) - 1 ~ - sum_t l_t,

so the bias is a sum of per-token log-ratios over the whole completion. It is
systematic (its mean is not zero), it grows with the length of the
completion, and it is first order in the sampler error.

A second term appears even with zero-mean advantages. For a fixed prefix,
the score of the sampled token has expectation

    E_{v~q}[ grad log p(v) ] = sum_v q_v (e_v - p) = q - p   (in logit space),

which is zero only when q = p. A constant advantage c contributes c (q - p)
per position. Group centering removes the constant part of the sequence
advantage, but not this per-position term, and it does not remove a
length-weighted sum sum_i A_i L_i, which the per-token normalisation keeps.
Uncentred rewards (for example 2r - 1 with few solved rollouts) make c
large and negative, so the step pushes every sampled token down, mass flows
to unsampled tokens, and the policy degrades.

It compounds because the sampler is refreshed from the trainer each step:
the same mismatch re-enters the gradient at every step, and the biased step
moves the trainer further along a direction that is not the reward gradient.
Clipping only changes which part of the bias is kept.

## The modification

Default path (correction=None), three changes in losses.py:

1. Per-token ratio weights, detached: w_t = rho_t = exp(logp_t - logq_t),
   unclipped (log-ratio clamped to +-20 only for float safety).
2. Self-normalisation: the loss is divided by the total ratio mass
   Z = sum_{i,t} mask_it rho_it (detached) instead of the token count N:

       L = - sum_i A_i sum_t rho_it log p_it / Z.

3. Batch-centred advantages: A_i <- A_i - mean_j A_j before the loss.

Corrections passed explicitly (rl.is.*) keep their original behaviour.

### Why per-token ratio weights remove the first-order bias

For any f that depends on y_<=t only,

    E_q[ rho_t f ] = E_q[ p(y_t|y_<t)/q(y_t|y_<t) f ] = E_p[ f ].

With f = grad log p_t at a fixed prefix this gives E_q[rho_t grad log p_t] =
sum_v p_v grad log p_v = 0, the score identity under the trainer, so the
per-position term (q - p) disappears. The remaining bias of the token-ratio
estimator comes only from the dependence of A on the future tokens of the
completion through the factors rho_s with s > t. That is a covariance of the
future log-ratios with the advantage, which is second order in the
per-step mismatch in expectation, and does not grow with the length of the
prefix the way the uncorrected bias does.

### Why not the sequence (prefix) product

The exact sequence weight rho(y) = prod_t rho_t has variance that grows
exponentially with the length T, roughly exp(T Var log rho), so the
sequence-level estimator is unusable at these lengths. A prefix product
rho_<=t is also unbiased only for the prefix part of the future and has the
same explosion. Experimentally (mini, noise 0.6, 4 seeds), prefix-product
weights collapse to eval reward 0.00 to 0.35, which is worse than the
uncorrected loss.

### Why clipping is the wrong fix

Clipping w = min(rho, c) replaces rho by a bounded function. The bias is
E_q[(rho - c)_+ h], which has a sign set by the advantage. Its size grows as
c shrinks, so stronger token clipping adds more bias, which is what is
observed.

### Why self-normalisation is the right variance control

E_q[rho_t] = 1 at every position, so Z is approximately N and the
normalisation changes the scale only by the random deviation of the batch
mass from its mean. The estimator sum rho A g / sum rho is a ratio estimator:
consistent as the batch grows, with the usual O(1/N) finite-sample bias. Each
weight is at most rho_max / Z, so the effective sample size is at most the
number of tokens, and no clipping bias is introduced. The overall
sampler-to-trainer mass error cancels in the ratio, which makes the update
scale insensitive to a systematic drift in the total ratio mass.

### Why batch-centring the advantages

If A is constant c, the token-ratio term has zero expectation per position
but the self-normalised ratio of sums c sum rho grad log p / sum rho keeps a
finite-sample drift of order c / sqrt(N) in a fixed direction. Centring over
the batch makes the advantage a valid baseline (the batch mean is independent
of the sampled token up to O(1/N)), so the constant term that produced the
collapse with uncentred rewards is removed. Group-centred rewards, the
train_rl default, already sum to zero over the batch, so this step is a
no-op there.

## Evidence (mini setup, eval reward at the end of 400 steps, 4 seeds)

| setting | uncorrected loss | new default |
| --- | --- | --- |
| noise 0.6, group-centred | 0.37, 0.49, 0.36, 0.58 | 0.63, 0.56, 0.57, 0.66 |
| noise 0.6, uncentred 2r-1 | 0.00 on all seeds (collapse) | 0.67, 0.66, 0.38, 0.54 |
| dev default (noise 0.3, 600 steps) | 0.66 final, peak reward 0.67 | 0.85 final, peak reward 0.79 |

Naive token IS (no self-normalisation, no centring) gives 0.52 to 0.64 with
group-centred rewards. Prefix-product weights collapse. Noise 1.0 is hard for
every method on the mini task because the sampler almost never solves it;
the new loss avoids collapse to zero there (0.49, 0.41 on two seeds) but
does not learn much.

Known limits: with uncentred rewards and noise 0.6, one seed still drifts
down (0.38). The bias analysis above is first order, so large mismatch
(noise 1.0) is outside its regime.

## Configuration

No configuration changes. The modification is active with the default
arguments, that is with no correction passed, which is the train_rl default.
