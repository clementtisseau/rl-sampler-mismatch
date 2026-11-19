# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

Under a sampler/trainer mismatch, the REINFORCE gradient `A * grad log p(y)`
on tokens drawn from the sampler `q` is not "the policy gradient plus noise".
At every position it contains a **deterministic drift term `A * (q - p)`** in
logit space. That term does not come from the reward, does not average to zero,
and gets applied at every step. Because the sampler is rebuilt from the trainer
after each update, the drift never shrinks, so it accumulates until training
collapses.

The fix is to subtract, at each position, the score's expectation under the
sampler:

    score_t = log p(y_t | h_t) - sum_v sg(q(v | h_t)) * log p(v | h_t)
    loss    = - sum_i A_i sum_t score_{i,t} / n_tokens

The gradient of `score_t` with respect to the trainer logits is `e_y - q`
instead of `e_y - p`. This is the sampler's own score function, so it has
exactly zero mean under the distribution that produced the token. The expected
update becomes the exact policy gradient of the reward obtained by the sampler,
with no drift. When the two policies match (`q = p`), the extra term has zero
gradient and the loss is plain REINFORCE. The method uses no importance
weights, no clipping and no dropped tokens.

`q` is needed over the vocabulary. It is reconstructed from what samplers return
anyway: the top-20 logprobs and the sampled token's logprob, which
`train_rl.py` already puts in the batch. The leftover mass is handled exactly
in aggregate, as described below.

## 1. What is wrong with the current loss

Notation: at position `t` with prefix `h_t`, the trainer has logits `z_t` and
`p_t = softmax(z_t)`. The sampler has `q_t`. The token `y_t ~ q_t`, and `A` is
the (group-centered) advantage of the sequence. The current loss is
`-sum A sum_t log p(y_t)`. Its gradient with respect to the logits at
position `t` is

    g_t = A (e_{y_t} - p_t).

Split it into two parts:

    g_t = A (e_{y_t} - q_t)   +   A (q_t - p_t)
          \_______________/       \___________/
            (I) score of q        (II) drift d_t = q_t - p_t

**(I) is the right object.** If `zeta` are the sampler's logits, then
`grad_zeta log q(y) = e_y - q`. Hence

    E_q[ sum_t A (e_{y_t} - q_t) ]  =  grad_zeta J(q),     J(q) = E_{tau ~ q}[R(tau)],

which is the exact likelihood-ratio policy gradient of the reward the sampler
obtains. It is evaluated on the sampler's own trajectory distribution, so
there is no state-distribution mismatch. It has the usual baseline property:
`E_{y~q}[e_y - q] = 0`, so any part of `A` that does not depend on `y_t`
contributes nothing in expectation.

**(II) is bias.** `d_t = q_t - p_t` depends only on the prefix, not on `y_t`.
It does not average out over tokens. Its contribution to the expected update is

    E[ sum_t A d_t(h_t) ].

- With uncentered rewards (`plus_minus`, `plus_zero`, ...) this term includes
  `E[A] * E[sum_t d_t]`. That is a push of the whole policy along `+(q - p)`
  (toward the sampler) or `-(q - p)` (away from it), depending on whether most
  rollouts succeed or fail.
- Group centering removes the constant part. What is left is
  `Cov_q(A, sum_t d_t(h_t))`, the correlation between a trajectory's reward
  and how much sampler and trainer disagree along it. This is generically
  nonzero. For example, low-reward trajectories are those that wandered into
  prefixes where the perturbed sampler puts extra mass on unlikely tokens.

In both cases the term is a fixed direction set by the mismatch, not by the
task. It is not the gradient of any reward.

**Why it compounds.** Zero-mean noise in the update accumulates like
`sqrt(steps)` and is damped by the true gradient. The drift has nonzero mean,
so it accumulates linearly. It also never goes away, because after each update
the sampler is rebuilt from the trainer: `q_{k+1} = Perturb(p_{k+1})`. Moving
`p` along `±(q - p)` therefore does not close the gap. The perturbation is
reapplied on top, and `d` stays about the same size or grows. Weight noise and
quantization perturb logits roughly at random. By Jensen
(`E[exp(z + eps)] > exp(z)`), this inflates the sampler's mass on the many
unlikely tokens, so `d` has a systematic component that "flattens toward the
tail". When drift pushes `p` toward `q`, the next `q` is the perturbation of a
flatter `p` and is flatter still. The fixed point of `p <- Perturb(p)` is
high-entropy garbage, which is the observed collapse. When the drift pushes the
other way, `p` takes an unbounded walk along `-(q - p)`. Either way the
objective being followed is not the reward.

**Why IS corrections and clipping only slow it down.** For any detached
per-token weight `w(y)`, the expected logit gradient at a position is

    E_{y~q}[ w(y) (e_y - p) ] = q.w - (E_q w) p        (elementwise product q.w)

This is zero, meaning the drift is gone, iff `q.w ∝ p`, i.e. iff
`w ∝ p/q` exactly, with no truncation.

- With truncation (`w = min(p/q, c)`) the residual drift is
  `min(p, c q) - (sum min(p, c q)) p`. It is nonzero wherever `p > c q`.
- Dropping out-of-band tokens (MIS, PPO-style masks) leaves the residual
  `p.1_band - p(band) p`.
- Tighter bands clip or drop more tokens, so the residual grows. This is why
  stronger token-level clipping can make things worse.
- Untruncated IS removes the drift but has unbounded variance, with heavy
  tails exactly on the unlikely tokens where `p/q` is extreme.
- Sequence-level ratios multiply `T` such factors, so their variance is worse.

All these methods trade bias for variance through a weight on the sampled
token. None of them can make the score zero-mean under `q`, because they only
rescale the sampled token's gradient. The drift lives in the `-p` part of
`e_y - p`, which no weight on `y` can turn into `-q`.

## 2. The modification

Fix the score itself, not its weight. At each position subtract its
expectation under the sampler, which is a function of the prefix only:

    b_t = E_{v~q_t}[ grad log p(v | h_t) ] = sum_v q_t(v) (e_v - p_t) = q_t - p_t = d_t.

So the drift is exactly the score's mean under `q`, and it can be removed with
a prefix-dependent control variate. As a surrogate:

    S_t(theta) = log p_theta(y_t | h_t) - sum_v sg(q_t(v)) log p_theta(v | h_t)
               = log p(y_t) + CE(sg(q_t), p_t)

    L(theta) = - (1 / n_tokens) sum_i A_i sum_t mask_{i,t} S_{i,t}(theta)

Gradient with respect to the logits, using `sum_v q(v) = 1`:

    grad_z S_t = (e_{y_t} - p_t) - sum_v q(v) (e_v - p_t) = e_{y_t} - q_t.

### Properties

1. **No drift.** `E_{y_t ~ q_t}[e_{y_t} - q_t | h_t] = 0` for every prefix,
   so term (II) is gone exactly. If the reward does not depend on the policy,
   the expected update is zero, whatever the mismatch is and whatever
   baseline is used.

2. **Unbiased for a consistent objective.** The expected logit-space update is
   `E_q[sum_t A (e_{y_t} - q_t)] = grad_zeta J(q)`, the on-policy gradient of
   the sampler's return. The batch is drawn from `q`, so this is ordinary
   on-policy REINFORCE for the sampler. The trainer applies it through its own
   Jacobian, `dtheta = (dz/dtheta)^T grad_zeta J(q)`.
   - When the sampler is a fixed transformation of the trainer (weight noise,
     quantization, a quantized KV cache), `zeta = z + Delta` with `Delta`
     varying slowly, so `dzeta/dtheta ≈ dz/dtheta`. The update is then the
     gradient of `J(q_theta)`, the reward of the policy that is actually
     deployed and evaluated.
   - Under staleness, `q` is the policy from `k` steps ago. The update is
     still a valid ascent direction for that policy's return, which is the
     usual lag of asynchronous RL. It does not pull `p` along `p_old - p`.

   The stationary points are those of a single well-defined objective, so the
   mismatch cannot accumulate.

3. **Exactly REINFORCE on-policy.** If `q = p`, then
   `grad sum_v sg(p(v)) log p(v) = p - p = 0`, so the gradient is unchanged.
   The modification only acts through the mismatch.

4. **Variance like on-policy REINFORCE.** `||e_y - q|| <= sqrt(2)`. There are
   no ratios, so nothing explodes on tail tokens, and no clipping or dropping
   that would bring the bias back.

5. **Interpretation.** `-A * E_q[log p]` is an advantage-weighted cross-entropy
   to the sampler. On rollouts that went badly (`A < 0`) it pulls `p` toward
   `q`. On rollouts that went well it pushes away. Together with the sampled
   token's term, this is exactly what cancels the `A (q - p)` push of the
   vanilla loss.

### Computing `sum_v q(v) log p(v)` from what samplers return

Inference engines return the sampled token's logprob and the top-k logprobs
(here `k = 20`: `batch["sampler_topk_ids"]` and
`batch["sampler_topk_logprobs"]`). Let `S = topk ∪ {y}`. Then `q` is known
exactly on `S`, and `q_rest = 1 - q(S)`. Spread `q_rest` over the remaining
vocabulary in proportion to `p`:

    q_hat(v) = q(v)                     for v in S
    q_hat(v) = q_rest p(v) / p(rest)    for v not in S

    sum_v q_hat(v) log p(v) = sum_{v in S} q(v) log p(v) + q_rest log p(rest)   (+ a constant)

The gradient of the score becomes `e_y - q_hat`. The residual drift is
`E_q[e_y - q_hat] = q - q_hat`, which:

- is exactly zero on every top-k token, and
- is exactly zero for the total off-top-k mass, since `q_hat(rest) = q_rest`.

So the leading collapse mode, sampler mass leaking into the tail, is cancelled
exactly. Only the shape inside the tail is approximated, with
`||q - q_hat||_1 <= 2 q_rest`, which is the probability mass outside the top 20.

`log p(rest)` is computed as a `logsumexp` over the vocabulary with the entries
in `S` masked out. This avoids cancellation in `1 - sum p`, and it also works
when the vocabulary is tensor-parallel sharded. If the batch has no top-k
fields, `S = {y}` is used as a fallback. The fallback is exact on the sampled
token and on the aggregate of the rest, but its cancellation is weaker.

## 3. Implementation (`losses.py`)

- `sampler_expected_logp(...)` computes `sum_v q_hat(v) log p(v)` per position
  as above. Gradients flow only through `p`; `q` is stop-gradient.
- `mask_vocab` scatters the mask into a vocab axis that may be TP-sharded. It
  is the counterpart of `take_vocab`.
- `loss_fn_rl` uses `score = log p(y) - E_q[log p]` instead of `log p(y)`.
  This is always on: it needs no flag and works with `correction=None`.
  - It uses the top-k fields when they are in the batch.
  - The signature and metrics are unchanged.
  - An optional `ISCorrection` still multiplies the (now centered) per-token
    score, or the advantages for sequence-level corrections. Corrections are
    not needed, and by the analysis in section 1 any non-constant weight
    reintroduces a small drift `q.w - (E_q w) q`. The recommended setting is
    no correction.

Checks run in the CPU setup:

- The logit gradient equals `-A (e_y - q) / n_tokens` to float precision when
  `k` covers the vocabulary.
- The sharded version, with TP over the vocabulary, matches the unsharded
  one.
- On `dev/mini.py` with noise 0.3, the vanilla loss collapsed to reward 0 when
  extra unlikely tokens were present. The modified loss trained stably to a
  sampler reward of about 0.85 in that setting. Without extra tokens it
  improved from about 0.58 to about 0.83.

## Configuration

rl.is.level=null
