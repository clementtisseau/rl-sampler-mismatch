# Sampler-expected score centering for REINFORCE under sampler mismatch

## What is wrong with the current loss

`loss_fn_rl` differentiates `w * log p_theta(y_t | h_t) * A` at every completion
token, where `y_t ~ q(. | h_t)` is the token the sampler drew, `A` is the
sequence advantage, and `w` is an optional detached IS weight. Fix a prefix
`h` and write `p_v = p_theta(v|h)`, `q_v = q(v|h)`, `s_v = grad_theta log p_v`,
`A_v = E[A | h, y = v]`, `Abar_q = sum_v q_v A_v`. With no correction, the
expected update at this position is

    E_q[A s_y | h] = sum_v q_v A_v s_v
                   = Abar_q * sbar_q(h) + Cov_q(A, s_y | h),
    sbar_q(h) = sum_v q_v s_v = grad_theta sum_v q_v log p_v.

On-policy (q = p) the first term vanishes because
`sum_v p_v s_v = grad_theta sum_v p_v = 0`. Under mismatch it does not: since
`sum_v p_v s_v = 0`, we have `sbar_q = sum_v (q_v - p_v) s_v`, which is first
order in the mismatch `q - p`. The term `Abar_q * sbar_q(h)` is the drift of
the hint. Its coefficient is the expected advantage at the prefix, which is
generically nonzero (good prefixes, bad prefixes), while the direction
`sbar_q(h)` is not a reward direction. It is the gradient of `sum_v q_v log p_v`,
which raises the trainer's log-probability of whatever the sampler
over-represents and lowers that of what it under-represents. Under a fixed
sampler this direction is systematic, so it does not average out over the
batch or over steps. Because the sampler is refreshed from the trainer at
every step, the same mismatch is re-injected at every step, and the bias
accumulates in the parameters. That is the observed degradation.

Why the existing knobs only slow it down. A per-token IS weight `w_v = p_v/q_v`
on the sampled token would give `E_q[w A s_y] = sum_v p_v A_v s_v`, which is the
on-policy expectation and has no drift. It is exact only where the weight is
not truncated. Clipping or dropping `w` (TIS, PPO, DAPO, ...) changes the
coefficient of each `s_v` to `q_v * clip(p_v/q_v) * A_v`, which again leaves
`sum_v c_v A_v s_v` with `c_v` not equal to `p_v`, so a drift term of the same
form remains for the clipped tokens. Stronger clipping moves more mass into
this regime, which is consistent with the observation that it can be worse.

## The modification

Replace the sampled-token score by a centered score: subtract the
sampler-expected log-probability of the prefix,

    Phi(h) = sum_v q(v|h) log p_theta(v|h)        (q is stop-gradient)
    grad Phi(h) = sbar_q(h)

    c_y(h) = log p_theta(y|h) - Phi(h)           gradient: s_y - sbar_q(h)

and use the loss `loss = -sum_t w_t * A * c_{y_t}(h_t) / n_tokens`, with `w_t` the
existing IS weight (`1` when no correction is set). With `correction=None` this
is `-A * (log p(y) - Phi)` per token.

Expected update under q at prefix h:

    E_q[A (s_y - sbar_q) | h] = sum_v q_v A_v s_v - Abar_q * sbar_q
                              = sum_v q_v (A_v - Abar_q) s_v
                              = Cov_q(A, s_y | h).

So the drift term `Abar_q * sbar_q` is removed exactly, and what remains is
the covariance term of the hint, evaluated under q.

Properties:

1. **Drift-free.** If the advantage does not depend on the next token at this
   prefix (`A_v` constant in v), the expected update is `0` for any q, p. The
   original update is `A * sbar_q != 0` in that case. In other words, the
   centered estimator only moves the policy where the advantage carries
   information about the token, which is the requirement "no update that does
   not come from the reward signal".
2. **On-policy unchanged in expectation.** For q = p, `sbar = 0`, so the
   subtracted term has zero mean: `E_p[A (s_y - sbar_p)] = E_p[A s_y]`. Only the
   variance changes. We checked this: with zero noise, the centered and plain
   runs have the same reward curve within noise (see below).
3. **Residual bias.** Let `T = sum_v p_v A_v s_v` be the true on-policy
   gradient. The centered estimator is `sum_v q_v (A_v - Abar_q) s_v`, so the
   remaining error is `sum_v (q_v - p_v) A_v s_v - Abar_q sbar_q`. The original
   error was `sum_v (q_v - p_v) A_v s_v`, i.e. the same term plus the drift.
   The residual `sum_v (q_v - p_v) A_v s_v` is a reweighting of the advantage
   over tokens the sampler under- or over-samples. It is reward-driven (it is
   proportional to `A_v`), so it does not accumulate as a fixed direction,
   and it is removed by a full-vocabulary IS correction, which we do not
   assume.

## Why it needs the sampler's distribution

`Phi(h)` needs `q(.|h)` over the vocabulary, not just `q(y|h)`. The loss uses
`sampler.vocab_logprobs`:

- `full`: `sampling_vocab_logprobs` [B, T, V] gives exact `Phi`.
- `k`: `sampler_topk_ids`/`sampler_topk_logprobs` [B, T, k]. We use `q`
  renormalized over the recorded top-k tokens, so the tail mass outside the
  top-k is dropped from `Phi`. This is what makes `full` infeasible at real
  vocabulary sizes (B*T*V float32 is ~160 GB for the dev run at Qwen's
  V = 151,936), and it is the setting we recommend.

Position alignment: the logprobs at position t are the distribution of token
`t+1`, which matches `targets = roll(tokens, -1)` in `loss_fn_rl`. The head is
evaluated in the same chunks as the sampled-token logprob, so only
`[B, chunk, V]` is live as before.

If the sampler does not record its distribution, `expected_logp` returns
`None` and the loss reduces to plain REINFORCE (the old behavior). The
modification is therefore active under the default arguments only when the
configuration below is set.

## Configuration

sampler.vocab_logprobs=32

## Checks run (dev/mini.py, CPU, 600 steps unless noted)

Eval reward (clean sampler), plain REINFORCE (`null`) vs centered loss:

- Noise 0.6, no extra tokens, lr 0.2: plain goes 0.48 (step 0) to 0.0 by
  step 25. Centered with full vocab stays between 0.50 and 0.62 at every
  checkpoint.
- Noise 0.3, lr 0.6: plain goes 0.56 to 0.24 at step 25 and 0.0 by step 50.
  Centered with full vocab rises to about 0.8 and stays there.
- Noise 0.3, 200 extra vocabulary tokens, lr 0.2: plain goes 0.39 to 0.0 by
  step 25. Centered with top-32 rises to about 0.8 and stays there. Top-8 is
  stable but lower (about 0.65), which matches the tail mass it drops.
- Noise 0.6, 200 extra tokens, lr 0.2: plain goes 0.30 to 0.0 by step 25.
  Centered with full vocab stays between 0.40 and 0.51.
- Noise 0 (on-policy): plain and centered (full vocab) match at every
  checkpoint up to step 225 and end at 0.895 and 0.880 (step 400).
