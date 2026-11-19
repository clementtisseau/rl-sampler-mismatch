# A sampler-centered score for REINFORCE under a sampler/trainer mismatch

## Summary

The REINFORCE loss `-A * log p(y)` depends on the identity
`E_{y~p}[grad log p(y)] = 0`. Because of it, any part of the advantage that
does not depend on the sampled token (a baseline, the mean advantage, length
weighting, how lucky the prefix was) has no effect in expectation. When the
tokens come from a sampler `q != p` the identity fails:
`E_{y~q}[grad log p(y)] = -grad KL(q || p)`. Every token then adds a
deterministic term `mean-advantage * (-grad KL(q||p))`, which pulls `p` onto
the sampler's errors or pushes it away from them. The sampler is rebuilt from
the trainer at every step, so the term points the same way each step and adds
up. That is the collapse.

The fix is to center the score of every token under the sampler:

    per token:  w(y) log p(y)  ->  w(y) log p(y) - sum_y' sg(q(y') w(y')) log p(y')

Under `y ~ q` its gradient has mean zero exactly, for any detached IS weight
`w`, including none, clipped, or dropped. On-policy (`q = p`, `w = 1`) the
subtracted term has zero gradient, so there the loss is plain REINFORCE.
`losses.py` always applies this, with or without a `correction`. The sampler's
next-token distribution comes from its top-20 logprobs, which the batch
already carries (`sampler_topk_ids` and `sampler_topk_logprobs`). No
configuration change is needed.

## Setup and notation

At a completion position with prefix `h`, the trainer has logits `z` and
`p = softmax(z)`. The sampler draws `y ~ q(.|h)`. The current loss is

    L = -(1/N) sum_i A_i sum_t w_it log p(y_it | h_it),       w = 1 without correction,

where `N` is the number of completion tokens. With respect to the logits,

    grad_z log p(y) = e_y - p.

Training follows `-grad L`. Its expectation over rollouts drawn from `q`, the
"expected update", is a sum over positions:

    U = (1/N) sum_t E_{h~q} E_{y~q(.|h)} [ A * w(y) * grad log p(y|h) ].

## 1. On-policy: the token-independent part of A drops out

Write `A = Abar(h) + (A - Abar(h))`, where `Abar(h) = E[A | h]` is the mean
advantage of rollouts that pass through the prefix `h`. If `q = p` and `w = 1`:

    E_{y~p}[grad log p(y)] = sum_y p(y) grad log p(y) = grad sum_y p(y) = grad 1 = 0     (*)

so the `Abar(h)` part contributes nothing and

    U_on = (1/N) sum_t E_h Cov_p( A, grad log p(y) | h ),

which is the policy gradient. This is why REINFORCE does not depend on
baselines. A constant shift of the rewards (plus_minus or zero_minus instead
of group_centered), longer sequences getting more tokens in the sum, and
`Abar(h) != 0` for "good" or "bad" prefixes all add variance but no bias.

## 2. Mismatched: (*) fails, and it fails systematically

With `y ~ q`:

    d(h) := E_{y~q}[grad log p(y)] = sum_y q(y) grad log p(y) = -grad_theta CE(q, p) = -grad_theta KL(q || p),

which is `q - p` in logit space and is nonzero whenever `q != p`. So

    U = (1/N) sum_t E_h [ Cov_q(A, grad log p(y) | h)  -  Abar(h) * grad KL(q(.|h) || p(.|h)) ].
                          \___ learning signal ___/     \____________ drift ____________/

The drift term does not depend on whether the sampled tokens were good. It
appears even when the reward carries no information about the tokens:

- `Abar > 0`: `p` descends on `KL(q||p)`. The trainer does SFT on the
  sampler's own errors, such as the noise or quantization error, and absorbs
  them.
- `Abar < 0` (most rollouts fail, as in plus_minus early in training, or long
  failed answers): `p` ascends on `KL(q||p)`. In logits
  `z <- z + eta|Abar|(p - q)`, so `p` moves away from the sampler. Mass shifts
  to tokens the sampler under-weights, often tail tokens.

`Abar` is nonzero in every reward mode:
- plus_minus gives `Abar = 2 E[r] - 1`.
- group_centered makes `sum_i A_i = 0` per group, but the loss weights
  sequence `i` by its length `L_i`. The effective per-token mean
  `sum_i A_i L_i / N` is therefore nonzero, and negative when failures are
  longer.
- Even with equal lengths, `Abar(h_t)` at later positions equals
  `V(h_t) - V(prompt)`, the value gained or lost so far, which is not zero.

Why it compounds instead of averaging out:
- A fixed weight perturbation, int8/fp8 weights, and a quantized KV cache all
  make `q` a deterministic function of the trainer's weights:
  `q = p_{theta + delta(theta)}`. So `d(h) = -grad KL(q||p)` points in roughly
  the same direction at every step, and the drift grows linearly with steps,
  not like `sqrt(steps)` as zero-mean noise would.
- The sampler is re-derived from the updated trainer, so the drift never
  "catches up" with `q`. In the `Abar < 0` case it increases the very mismatch
  `KL(q||p)` that drives it, which is a positive feedback loop.
- Its size `|Abar| * ||q - p||` does not shrink as learning saturates, while
  the learning signal does. Late in training the drift dominates and the
  reward collapses.
- Stale weights behave the same way, with `q` equal to an earlier `p`.

### Why IS corrections and clipping do not remove it

With a detached token weight `w(y) = w(p(y)/q(y))`:

    E_{y~q}[w(y) grad log p(y)]  =  sum_y q(y) w(y) (e_y - p)  =  q.w - p * E_q[w]     (logit space).

This is zero only for the exact ratio `w = p/q`, which gives `p - p`. That is
naive IS, whose variance blows up whenever `q << p` somewhere. Every bounded
weight gives a nonzero mean score, and so a drift `Abar * (q.w - p E_q[w])`:
- truncation (TIS) under-weights exactly the tokens with `p >> q`;
- dropping (MIS, PPO-style) zeroes some tokens but keeps `p`'s full
  normalization term;
- sign-dependent clipping (PPO, DAPO, TOPR) treats `A > 0` and `A < 0`
  differently, which creates a mean component of `A` even where there was
  none.

The tighter the band, the further `w` is from `p/q`, and the larger this
term. This explains why stronger token-level clipping can make things worse.
Sequence-level ratios are unbiased in principle, but their variance is
exponential in length, and clipping them is biased again.

## 3. The fix: center the score under the sampler

Replace the per-token score by

    s(y) = w(y) log p(y) - B(theta),       B(theta) = sum_y' sg[ q(y') w(y') ] log p(y'),

and use `L = -(1/N) sum_i A_i sum_t s(y_it)` (sequence-level weights still
multiply `A_i`). Then `grad B = E_{y'~q}[w(y') grad log p(y')]` and:

**Lemma.** For any detached `w` that depends on the token and the prefix,
`E_{y~q}[grad s(y)] = 0`.

*Proof.* `E_q[w grad log p] - grad B = E_q[w grad log p] - E_q[w grad log p] = 0`. ∎

So the expected update is

    U_new = (1/N) sum_t E_h Cov_q( A, w(y) grad log p(y) | h ),

and the drift term is removed exactly: `Abar(h)` and any baseline or shift of
`A` are again irrelevant, as in the on-policy case. If the reward carries no
information about the token choices, the expected update is 0 instead of
`-Abar grad KL(q||p)`. The trainer does not move when there is nothing to
learn, so nothing accumulates across sampler refreshes.

With `w = 1`, the gradient on the logits is `A (e_y - q)`, and its expectation
is

    E_q[A (e_y - q)] = sum_y q(y) (Q_q(h,y) - V_q(h)) e_y,

the policy gradient of the sampler policy with respect to its logits. This is
the correct improvement direction for the policy that produced the data, and
it is first-order close to `p`'s policy gradient. What remains of the
mismatch (Q-values and the prefix distribution under `q` rather than `p`) is
an error in the size of the signal that vanishes with the signal itself. It
is not a constant force.

**Consistency on-policy.** If `q = p` and `w = 1`, then
`grad B = sum_y p grad log p = 0` by (*). The loss gradient equals REINFORCE
exactly (checked numerically: max difference 1e-8). The modification is an
exact control variate. It changes nothing where REINFORCE is correct.

**Equivalent form.** With `w = 1`, `-A * s(y) = -A log p(y) - A * CE(q, p)`,
which equals `-A log p(y) - A KL(q||p)` up to a constant. The added term
`-A * KL(q || p)` cancels the implicit `+A * KL(q||p)` that sampling from `q`
puts into REINFORCE's expected loss. Because it is weighted by the same `A`,
it cancels per sample in expectation, not just on average over the batch.

**Compatibility with IS.** The lemma holds for every `w`, so the existing
`rl.is.*` corrections (TIS, MIS, PPO, ...) can be layered on top without
re-creating the drift. Their clipping then only trades variance against an
error in the learning signal, not against a drift.

## 4. Implementation (`losses.py`)

- `centered_baseline` computes `B(theta)` per position inside the existing
  rematerialized vocab chunks, and the token term becomes
  `score = w * logp - B`. It is always on, for `correction=None` and for every
  `ISCorrection`. `w(y')` over the vocabulary is `exp(log_mass(...))`, the
  same function that defines the token weight, so `B` uses the same band,
  clamp/drop and sign rules as the sampled token.
- **Sampler distribution.**
  - Inference engines return only the top-K logprobs of `q`; here K = 20, in
    `sampler_topk_ids` and `sampler_topk_logprobs`, aligned with
    `sampling_token_logprobs`.
  - On those K tokens `q_hat = q`. The remaining mass `1 - Q_K` is spread in
    proportion to `p`: `q_hat(y) = c p(y)` with `c = (1-Q_K)/(1-P_K)`.
  - The sum is computed as a full-vocabulary sum with `q_hat = c p`, corrected
    on the K head tokens by a gather (`take_vocab`), so there is no scatter on
    a tensor-parallel vocab axis.
  - Remaining mean score: in logits, `E_q[e_y - q_hat] = q - q_hat`. This is 0
    on the top-K tokens, and on the tail it reduces to the shape error of the
    tail inside mass `1 - Q_K`, instead of `q - p` everywhere. When K covers
    the vocabulary (as in `dev/mini.py`) the centering is exact.
  - The tail sums come from subtracting two sums, so they are accurate only to
    float32 rounding. `c` is therefore capped at `MAX_TAIL_SCALE = 100`. An
    uncapped `c` could reach 1e27 when `1 - P_K` rounds to 0, and it produced a
    NaN in one run. When the cap binds, `q_hat` sums to `S < 1`, the
    logit-space residual becomes `(q - q_hat) - (1 - S) p`, and its size is at
    most the sampler's tail mass.
  - If no top-k is present, the sampled token's own `q` is used (K = 1) as a
    best-effort fallback. It removes little of the bias, because the lemma
    needs `q` at more than the sampled token.
- Masked and padded positions are excluded with `where`, not multiplication,
  so values left over in padded top-k rows cannot produce NaNs.
- Checks:
  - with exact `q` and the expectation over the token taken by enumeration,
    the expected gradient of the new loss is 0 (|g| < 3e-8). The old loss
    gives 0.42 with no correction and 0.086 with TIS clipped at 1.5;
  - on-policy gradients match the original loss;
  - with truncated top-K (V = 50, the expectation over the token taken
    exactly), the maximum remaining mean gradient is:

    | K | Remaining mean gradient |
    | --- | --- |
    | 1 | 0.088 |
    | 5 | 0.038 |
    | 20 | 0.0028 |
- On the CPU dev setting (`dev/mini.py`, noise 0.3, 600 steps), with
  everything else unchanged, final eval reward:

  | Seed | Original loss | New loss |
  | --- | --- | --- |
  | 0 | 0.66 | 0.80 |
  | 1 | 0.66 | 0.74 |

  With 200 extra unlikely vocabulary tokens (`MINI_EXTRA_TOKENS=200`), so
  that top-20 does not cover the vocabulary:
  - the original loss collapses to reward 0 by step 50;
  - the new loss reaches eval reward 0.81 (seed 0) and 0.79 (seed 1);
  - the new loss with TIS (`high=2, clamp`) reaches 0.86.

## Configuration

No configuration changes are required: the modification is active with the
default arguments and with `correction=None`.
