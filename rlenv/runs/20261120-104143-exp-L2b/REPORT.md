# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## Summary

The REINFORCE score `log p(y)` is only centered (zero mean) under the
distribution that produced `y`. Under a mismatched sampler `q != p` it is not:
at every token the loss gets an extra expected update `b(h) (q - p)` in logit
space, where `b(h)` is the expected advantage at that prefix. This term is
not noise. It is the gradient of `-b(h) KL(q || p)`: it pulls `p` toward the
sampler on prefixes that tend to succeed and pushes `p` away from it,
without bound, on prefixes that tend to fail. The sampler is rebuilt from
the trainer at every step, so the gap `q - p` is reproduced each time and
the push keeps going in the same direction. That is the collapse.

The fix is to **center the score under the sampler instead of the
trainer**. The per-token loss becomes

    l_t = -A * w(y) * [ log p(y) - sum_v sg(pi_v) log p(v) ],   pi ∝ q * w(p/q)

(`pi = q` without a correction, `w = 1`). Its logit gradient is
`A w(y) (e_y - pi)`, which has zero mean under `y ~ q`. The expected update
therefore no longer depends on `b(h)` or on any constant added to the
advantage. Only the part of the advantage that depends on the sampled token
moves `p`, as in on-policy REINFORCE. When `q = p` the extra term has zero
gradient, so the loss is exactly the original one.

The sampler returns its top-20 logprobs per position (`sampler_topk_ids`,
`sampler_topk_logprobs`). `losses.py` uses them, plus the sampled token's
logprob, to build `pi`. This is on by default and works with
`correction=None`.

## 1. Setup at one token position

Fix a prefix `h`. Let `z` be the trainer logits, `p = softmax(z)`, and let
`q` be the sampler's next-token distribution at `h`. The token is drawn as
`y ~ q`. Let `A(y) = E[A | h, y]` be the expected advantage after `y`, with
the continuation also drawn from the sampler. Define

    b(h) = E_{y~q}[A(y)]                      (value of the prefix)

For the current loss `-A log p(y)`, the logit gradient of one sample
(descent direction, i.e. the update) is

    g(y) = A * ∇_z log p(y) = A (e_y - p).

The parameter update at this position is `J_h^T g(y)`, where `J_h = ∂z/∂θ`
is fixed once `h` is fixed. So we can analyze `E_q[g]` in logit space.

## 2. Expected update: on-policy vs. mismatched

Write `A(y) = b + (A(y) - b)`:

    E_q[g] = sum_y q(y) (A(y) - b) (e_y - p)  +  b sum_y q(y) (e_y - p)
           = sum_y q(y) (A(y) - b) e_y        +  b (q - p)             (1)

(The `- p` part of the first term drops out because
`sum_y q(y)(A(y) - b) = 0`.)

* **On-policy (`q = p`).** The second term is `b (p - p) = 0`. The update is
  `Cov_p(A(y), e_y)`, which does not depend on `b`. This is why REINFORCE
  is free to use any baseline. Group centering only changes the variance.
  A token whose advantage does not depend on its choice gets zero expected
  update.

* **Mismatched (`q != p`).** The second term is

      b (q - p) = b ∇_z E_q[log p] = -b ∇_z KL(q || p).                 (2)

  So in expectation the loss contains a hidden term `b(h) KL(q || p)` at
  every prefix (the loss being minimized):
  - `b(h) > 0` (prefix tends to succeed): minimize `KL(q||p)`, i.e. pull the
    policy toward the sampler's errors.
  - `b(h) < 0` (prefix tends to fail): **maximize** `KL(q||p)`, i.e. push
    `p` away from `q`. This has no bound: `KL(q||p) -> inf` as `p` puts zero
    mass on tokens the sampler likes.

Why this is a systematic bias and not zero-mean noise:

1. It does not average out over the batch. Group centering makes the
   advantages sum to zero per prompt, so `b = 0` only at the first
   completion token. As the prefix grows, the outcome becomes more fixed and
   `b(h) -> A`. Late tokens of failed rollouts therefore carry `b ≈ A < 0`
   and late tokens of successful ones carry `b ≈ A > 0`. These act at
   *different* prefixes with different `q_h - p_h`, so they do not cancel.
   With `plus_minus` or `zero_minus` rewards, `b` is not even centered at
   the prompt.
2. It has the same size as the signal. Per token, the signal in (1) is
   `Cov_q(A, e_y)`, which is small when one token does not change the
   outcome (most tokens). The drift is `|b| ||q - p||`, with
   `|b| ~ |A| = O(1)` on every token.
3. It compounds. The sampler is a fixed map of the trainer,
   `q = S(p)` (noise added to the same weights, quantization of the same
   weights, a stale copy). After each update the sampler is refreshed, so
   `q - p` comes back at every step with the same structure. The drift adds
   up coherently over steps (linearly), not like noise (as the square root).
   The repulsive part (2) moves `p` along `p - q`. This lowers `p` on tokens
   where `q > p`. Those are exactly the tokens the sampler over-samples,
   and they then get large `∇_θ log p` and ratios `q/p`. This loop is the
   collapse.
4. The plain loss has **no fixed point** under mismatch. Even when the
   policy is optimal (`A(y)` constant in `y` at every prefix), `E_q[g] =
   b (q - p) != 0`. Training cannot settle. It drifts until something
   breaks.

## 3. Why IS corrections and clipping only slow it down

With a detached per-token weight `w(y)`, the update is `A w(y)(e_y - p)` and

    E_q[w (e_y - p)] = m - |m| p,     m_y = q_y w_y,  |m| = sum_y m_y.     (3)

This is zero for every `b` only if `m ∝ p`, i.e. `w = c p/q`: exact,
unclipped IS. That is the only per-token weight that removes the drift,
and its variance `E_q[r^2] = 1 + chi^2(p||q)` is unbounded. Every bounded
variant leaves a drift `b (m - |m| p)`:

* TIS (`w = min(r, C)`): `m = min(p, C q)`. The residual pulls toward or
  pushes away from `m/|m|`, which differs from `p` on all tokens with
  `p > C q`.
* MIS / PPO / DPPO (drop outside the band): `m = p · 1[in band]`, so the
  residual is `b (p 1_in - P_in p)`. Mass is moved between in-band and
  out-of-band tokens in the direction set by `sign(b)`.

A tighter band gives a larger `|m - |m| p|`. So **stronger token-level
clipping can increase the drift**, which matches what was observed.
Sequence-level weights multiply `A`. They reweight whole rollouts but leave
the per-token direction `e_y - p` unchanged, so (1) still holds token by
token unless the weights are exact, and those have exponential variance.

## 4. The fix: center the score under the sampler

The drift comes from `e_y - p` having mean `q - p` under `q`. Replace it
by a direction that has zero mean under `q`. For any detached distribution
`pi` over the vocabulary,

    ∇_z [ sum_v pi_v log p_v ] = pi - p (sum_v pi_v) = pi - p,             (4)

so the per-token loss

    l = -A w(y) [ log p(y) - sum_v sg(pi_v) log p_v ]                     (5)

has logit update `A w(y) (e_y - pi)`. Take expectations under `y ~ q` and
shift the advantage by an arbitrary constant `beta`:

    E_q[(A(y) - beta) w(y) (e_y - pi)]
        = E_q[A w (e_y - pi)] - beta (q w - pi E_q[w]).                   (6)

The `beta` term vanishes for every `beta` if and only if

    pi = q w / E_q[w]      (pi = q when there is no correction, w = 1).    (7)

**Theorem (baseline invariance restored).** With `pi` from (7), the
expected per-token update is unchanged by any additive change of the
advantage at that prefix, including `b(h)`. In particular:

* There is no drift. If `A(y)` does not depend on `y`, the expected update
  is exactly 0, for any `q`. The optimal policy is again a fixed point of
  the expected dynamics. The hidden `b(h) KL(q||p)` term of (2) is gone.
* What remains is pure signal. With `w = 1`,

      E_q[A (e_y - q)] = sum_y q(y) (A(y) - b) e_y = Cov_q(A(y), e_y).

  This raises the logits of tokens that beat the prefix value and lowers
  those that do not. It is an ascent direction for the trainer's own
  objective: to first order, `δJ_p = sum_y p_y (A_y - b_p) δz_y =
  sum_y p_y q_y (A_y - b_p)(A_y - b_q) ≈ sum_y p_y q_y (A_y - b)^2 >= 0`
  for `q ≈ p`.
* It reduces to the original loss on-policy. If `q = p` then `pi = p`, and
  by (4) the extra term has zero gradient. Training is unchanged when there
  is no mismatch.
* It composes with IS corrections. With a token correction `w`, (7) gives
  `pi ∝ q w(p/q)`, which `losses.py` computes per vocabulary entry with the
  same `log_mass` as the correction. For exact IS (`w = p/q`) we get
  `pi = p` and the extra term vanishes again, because exact IS is already
  drift-free. For clipped or dropped weights, (7) cancels exactly the
  residual (3). Clipping then only changes the variance, not the bias.
  (For sequence-level corrections the per-token direction is centered with
  `pi = q`. The sequence weight still multiplies `A`.)

So the change does not suppress the symptom (large ratios). It removes the
cause: the part of the update that the mismatch creates for every token,
whatever that token's advantage.

Why this is better than a per-token baseline: in principle (1) could also be
fixed by subtracting a per-prefix baseline `b(h)` (a value model), but that
needs a critic and is only as good as the critic. Centering the score under
`q` makes the update invariant to *every* baseline, so no critic is needed.

## 5. Estimating `pi` from what the sampler returns

`pi` needs `q` over the vocabulary. The sampler returns `q` on its top-K
tokens (K = 20, as inference engines do) and at the sampled token `y`. Let
`S = top-K ∪ {y}`, `Q = q(S)`, `P = p(S)`. Off `S`, the remaining sampler
mass is spread in proportion to the trainer:

    q̂_v = q_v                     for v in S
    q̂_v = p_v (1 - Q) / (1 - P)   for v not in S.

Then the tail ratio `p/q̂ = (1-P)/(1-Q)` is the same for all tail tokens, so
the tail weight `w_tail` is a single number, and

    pi_v = q_v w_v / Z (v in S),   pi_v = kappa p_v (v not in S),
    Z = sum_S q_v w_v + (1 - Q) w_tail,   kappa = (1 - Q) w_tail / (Z (1 - P)).

To get the gradient `pi - p` of (4) without scattering into the
vocabulary, which may be sharded, `losses.py` uses only the gathered `|S|`
entries:

    center = sum_{v in S} sg(a_v) log p_v,    a_v = pi_v - kappa p_v.

Check: `∇_z center = a - p sum_S a = pi_S - kappa p_S - p ((1 - pi_tail) - kappa P)`,
and `pi_tail = kappa (1 - P)`, so this is
`pi_S - kappa p_S + (kappa - 1) p = pi - p`. This holds exactly, with no
`O(1)` cancellation.

Properties of the estimate:
- The remaining drift is `b (q - q̂)`. It is supported on the tail only,
  with norm at most `2 |b| (1 - Q)`. That is negligible for an LLM's top-20
  and exactly zero when the top-K covers the vocabulary (the CPU mini
  setup). A unit check with `K = V` gives `||E_q[grad]|| ~ 1e-8` for no
  correction, TIS, MIS, PPO and DPPO, against `||q - p|| = 0.12` for the
  plain loss.
- If the top-k keys are missing from the batch, `S = {y}`. The correction
  is then partial: the drift is reduced at the sampled token's mass, never
  increased.
- Guards: `1 - P` is floored at `1e-6`, `kappa` is capped at 100, and the
  coefficients are zeroed at masked or non-finite positions (the sampler's
  top-k buffer is uninitialized on prompt positions).

## 6. What changed in `losses.py`

- New `sampler_centering(...)`: builds `S`, the detached coefficients `a_v`
  above, and the correction-aware `pi` (through the existing `log_mass` /
  `is_weight`, so every `rl.is.*` point is supported).
- `loss_fn_rl`: the per-token score is
  `w(y) * (log p(y) - sum_S a_v log p_v)` instead of `w(y) * log p(y)`.
  Everything else (advantages, sequence-level corrections, metrics,
  chunking and rematerialization) is unchanged. The signature is unchanged.
  The change is active with default arguments and `correction=None`.

## Configuration

No configuration change is needed: the method is on by default and works
with no IS correction (`rl.is.level=null`, the trainer default).
