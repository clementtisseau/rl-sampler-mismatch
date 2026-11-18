# Sampler/trainer mismatch in the REINFORCE loss

## 1. What is wrong with the current loss

With no correction, `loss_fn_rl` uses w = 1, so the loss is

    L(θ) = -(1/N) Σ_i A_i Σ_t log p_θ(x_it | h_it),

where x ~ q is drawn from the sampler, p_θ is the trainer, A_i is the group-centred advantage and N the token count.
Its expected gradient (per sequence, dropping 1/N) is

    g_q = E_{x~q}[ A(x) Σ_t ∇log p(x_t | h_t) ].

Expand the per-position score under q. For a fixed prefix h,

    Σ_x q(x|h) ∇log p(x|h) = Σ_x p(x|h) ∇log p(x|h) + Σ_x (q(x|h) − p(x|h)) ∇log p(x|h)
                           = 0 + Σ_x (q − p) ∇log p.

**On-policy (q = p)** the first term is ∇Σ_x p(x|h) = ∇1 = 0. The score has zero mean, so any advantage-independent
term drops out and the loss has the correct expected gradient ∇E_p[R].

**Mismatched (q ≠ p)** the score has non-zero mean under q:

    E_q[∇log p(x|h)] = −∇_θ KL(q(·|h) ‖ p_θ(·|h)).

This is a systematic force that moves the trainer toward the sampler's distribution (a cross-entropy/forward-KL
pull toward q). It does not average out: it is present whenever the advantage has a sample-independent component
at that position, and group centring only removes the group mean of A, not the correlation between A and the
samples. Since the sampler is refreshed from the trainer every step, q = p_θ + (mismatch) moves with θ. The pull
toward q therefore follows the mismatch, and the bias compounds. This is why the problem is a bias rather than
zero-mean noise, and why more training makes it worse.

Clipping does not remove this bias. For w = min(r, c) with r = p/q, the expected score is
Σ_x q·min(p/q, c)·∇log p = Σ_x min(p, c·q)·∇log p ≠ 0. The residual grows as c decreases, which matches the
observation that stronger token-level clipping can make things worse.

## 2. The fix: per-token ratio weights, self-normalised

Use the detached per-token ratio r_t = p(x_t|h_t) / q(x_t|h_t) as the weight on each score term. This is the only
per-token weight that gives an exact zero for the constant-advantage term. For any prefix h:

    E_{x~q}[ (p(x|h)/q(x|h)) ∇log p(x|h) ] = Σ_x p(x|h) ∇log p(x|h) = ∇ Σ_x p(x|h) = 0.

So the weighted per-position score has the same expectation as the on-policy one, and the sampler-induced
force toward q is removed exactly. Only the sampled token's logprobs (`sampling_token_logprobs`) are needed, so no
full-vocabulary sampler distribution is required. The ratio is bounded in the loss by the same e^±20 guard as
`is_weight`.

Self-normalisation. Let w_t be the detached ratios over the batch's completion tokens. The loss is

    L = −( Σ_{i,t} A_i w_it log p_it ) / ( Σ_{i,t} w_it ),

i.e. the denominator is the sum of the weights, not the token count. Its gradient is the self-normalised
importance-sampling estimator

    ĝ = Σ w_it A_i ∇log p_it / Σ w_it.

Because E_q[w] = 1 per token, the normaliser concentrates at the token count. The constant-advantage bias stays zero
to O(1/n) (the ratio of the numerator, whose expectation is zero, to its normaliser). The normalisation also
removes the overall scale of the weights (heavy-tailed ratios from rare tokens get bounded influence), which is what
keeps the gradient magnitude stable when the mismatch is large. On-policy (w = 1) the estimator reduces exactly to
the original loss, so this does not change the on-policy objective.

The default `correction=None` is this estimator. Any explicit `correction` is used unchanged (the old path), so
the other points of the `rl.is.*` grid remain available.

## 3. Residual bias and what was ruled out

Per-token weights are exact for the advantage-free part, but not for a sequence-level advantage. Writing the
expected update over prefixes, a per-token weight leaves a first-order bias from the prefix distribution and from
the future advantage (A depends on the later tokens, which are still drawn from q). Only the full trajectory ratio
R = Π_t r_t removes it exactly, since E_q[R f(x)] = E_p[f(x)] for any f. I tried that (`rl.is.level=seq_sum`,
`high=inf`), and also a self-normalised version. Both were unstable in the dev setting (mini, noise 0.45): their
variance grows with sequence length, and the per-sequence log-ratio is already about 1.5 nats at this mismatch. So
the chosen estimator trades the residual sequence-level bias for a variance that stays bounded per token.

## 4. Empirical notes (dev/mini.py, 600 steps, lr 0.2)

These runs are only a proxy and are not part of the report's argument. Mismatch = relative weight noise.

- Noise 0.45: the unweighted loss (w = 1) collapsed in all seeds tried (eval reward to about 0). Per-token
  unnormalised ratio: seeds 0 and 1 stable, seed 2 collapsed partway. TIS (clamp at 2): no collapse in training
  reward, but eval reward fell to about 0.3 (the clipping bias). Self-normalised per-token (the submitted default):
  seeds 0 and 2 stable, seed 1 stable to about step 400 and then collapsed late. Seq-level exact ratio: collapsed.
- Noise 0.3 (the visible setting): the unweighted loss already reached eval about 0.70. The submitted default reached
  about 0.78 eval on seed 0 and about 0.68–0.70 on seed 1 (the unweighted loss was about 0.70 on seed 0), so no
  clear regression was observed.

The mini proxy is much shorter than real Countdown rollouts, so the effect of the sequence-length variance and the
residual sequence-level bias are not established on real runs.

## 5. Summary

The w = 1 loss has expected gradient E_q[A ∇log p], which carries the bias −∇KL(q‖p) (per position, for a constant
advantage), a pull toward q at every position. Per-token ratios p/q give Σ_x p ∇log p = 0 exactly at each
position, so the bias is removed without clipping. Self-normalisation keeps this to O(1/n) and bounds the influence
of large ratios. The residual sequence-level bias is the price for avoiding the variance of full trajectory ratios.
