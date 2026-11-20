# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The loss now centers each token's score under the **sampler's** next-token
distribution instead of implicitly centering it under the trainer's:

    per-token term:   A * ( log p(y) - sum_v sg(q~(v)) log p(v) )

with q~ the sampler distribution reconstructed from the top-20 logprobs it
already returns (`sampler_topk_ids`, `sampler_topk_logprobs`) plus the
sampled token's logprob. With respect to the logits z this term's gradient is
`A (e_y - q~)`, where plain REINFORCE has `A (e_y - p)`. When q = p nothing
changes. When q != p the expected update for a constant advantage is exactly
zero again (zero up to the tail approximation, see below). That removes the
systematic drift `-E[A|s] grad KL(q||p)`, which is what makes training
collapse. It needs no importance ratios, so there are no large weights and no
clipping. It is on by default (`correction=None`), and it also composes with
any `rl.is.*` correction (section 5).

## 1. Expected update of the current loss

Notation: a state s is a prompt plus a prefix. The trainer gives
p(.|s) = softmax(z), and the sampler gives q(.|s). The current loss is
`-(1/N) sum_i A_i sum_t log p(y_t | s_t)`, with tokens y_t ~ q(.|s_t). For a
softmax, the gradient of `log p(y)` with respect to z is

    d log p(y) / dz = e_y - p.

**On-policy (q = p).** At each state the expected score is zero:

    E_{y~p}[ e_y - p ] = p - p = 0,         i.e. E_p[grad log p] = grad sum_y p = 0.

So whatever part of the advantage does not depend on y (the group mean, the
sign convention plus_minus/plus_zero/zero_minus, a length or prefix effect)
has zero expected effect. Only differences in reward between actions move the
policy. Write A = Q(s,y) - b(s) with Q(s,y) = E[A | s, y] and
b(s) = E_q[A | s]. Then the expected logit update is

    E_{y~p}[ A (e_y - p) ] = p ⊙ (Q(s,.) - V(s)),

which is the exact policy gradient.

**Mismatched (q != p).** The score no longer has zero mean:

    E_{y~q}[ e_y - p ] = q - p  != 0,       i.e. E_q[grad log p(y)] = sum_y q grad log p = -grad_theta KL(q || p)   (q fixed).

So the expected logit update at s becomes

    E_{y~q}[ A (e_y - p) ] = E_q[ A (e_y - q) ]  +  E_q[A|s] (q - p)
                           = q ⊙ (Q_q(s,.) - V_q(s))  +  b(s) (q - p)
                             \_____ policy gradient ___/   \__ drift __/

The second term is a **bias, not noise**. It does not average out over
samples, and it is not a policy-gradient term at all: it equals
`-b(s) grad_z KL(q||p)`. In states where the mean advantage is positive it
pulls p toward q. Where it is negative (failed rollouts, negative
plus_minus/zero_minus rewards, long failing completions under per-token
normalization) it pushes p away from q. Group centering only removes the
part of b(s) that is constant across the whole batch. b(s) is the advantage
conditioned on the prefix, so it is generically nonzero: a prefix that has
already gone wrong has b(s) < 0.

**Why it compounds.** The sampler is the trainer passed through a fixed
distortion D (weight noise, quantization, a stale copy): q = D(p). Moving
p toward or away from q is a move along the distortion direction, and the
next step's sampler is D of the moved p, so the same displacement is
requested again. With q_theta = p_{theta+delta}, for example, the drift
gradient is about `±b * F * delta`, where F is the Fisher matrix. It is
constant in direction from step to step, so the parameters walk along
∓delta linearly in the number of steps. The reward term pushes back only
once the damage is visible in the reward, which is too late. Negative-b
states are where the policy is already failing. There the term lowers the
logits of exactly the tokens that q over-weights relative to p, and raises
what q under-weights. This is unlikelihood training on the distortion. It
sharpens or distorts p in a direction the next sampler amplifies again, so
the mismatch KL(q||p) grows and so does the drift. That feedback loop is the
collapse.

## 2. Why IS and clipping do not fix it

Use a token-level weight w(y) = f(p(y)/q(y)), so the logit gradient is
`A w (e_y - p)`. Its expected update for a constant advantage is

    E_q[ w (e_y - p) ] = sum_y q(y) w(y) e_y - E_q[w] p.

This vanishes for every q only if q(y) w(y) ∝ p(y), that is, only for the
untruncated ratio w = p/q. Exact IS removes the drift, but it does so with
weights that are unbounded exactly where the mismatch is large. That gives
huge variance, especially at sequence level, where the ratio is a product
of T per-token ratios. Any clipping or truncation f != identity leaves a
residual drift:

    E_q[ w (e_y - p) ] = - sum_y (p(y) - q(y) w(y)) (e_y - p)   (for E_q w ≈ 1).

Its magnitude is the clipped mass. Tighter clipping removes more mass, so
the residual drift is larger. This is why "stronger token-level clipping can
even make it worse". PPO-style sign rules zero out exactly one side of the
correction, so the residual is systematic again. IS-type methods therefore
trade drift for variance. They can slow the walk, but they cannot stop it.

## 3. The fix: center the score under the sampler

The drift comes entirely from using p, not q, as the centering term of the
score. Replace the score with one whose expectation under the sampling
distribution is zero for every q:

    s(y) = grad log p(y) - E_{v~q}[ grad log p(v) ]
         = grad [ log p(y) - sum_v sg(q(v)) log p(v) ]          (q detached)

With respect to the logits:

    d/dz [ log p(y) - sum_v q(v) log p(v) ] = (e_y - p) - (q - p) = e_y - q.

Expected update under the sampler, for any advantage:

    E_{y~q}[ A (e_y - q) ] = q ⊙ (Q_q(s,.) - V_q(s)).

Properties:

* **No drift.** The term `b(s)(q - p)` is gone: E_{y~q}[e_y - q] = 0 for
  every q. A constant or prefix-only advantage has zero expected effect
  again, as it does on-policy. Baselines, reward modes and length effects
  become pure variance, never bias.
* **The right direction.** `q ⊙ (Q_q - V_q)` is exactly the policy gradient
  of the sampler's own expected reward J(q), in logit space. If the sampler
  logits are the trainer logits plus a perturbation that changes slowly with
  theta (z_q = z_p + delta), then dz_q/dtheta ≈ dz_p/dtheta, and a step on
  theta along `(dz_p/dtheta)^T (e_y - q) A` changes q by an on-policy
  REINFORCE step on q. The coupled system trains the policy that is actually
  sampled, with on-policy dynamics. Its fixed points are stationary points
  of J(q), so nothing accumulates step after step. The residual mismatch
  enters only through the Jacobian difference dz_q/dtheta - dz_p/dtheta.
  That difference multiplies a zero-mean, reward-driven signal, so it is not
  a self-reinforcing direction.
* **Bounded weights.** No ratio p/q appears anywhere. The centering term is
  a convex combination of `log p(v)` with weights q(v) in [0, 1]. Variance
  is the same as REINFORCE: the per-token logit gradient has norm at most
  sqrt(2)|A|.
* **Exact reduction.** If q = p, then `grad sum_v sg(p(v)) log p(v) = sum_v p grad log p = 0`,
  so the loss gradient is the old REINFORCE gradient (checked numerically
  to 1e-8).
* **No extra hyperparameters.** The fix is active with default arguments.

### Using only the top-k sampler logprobs

The sampler returns q on a set K: its top-20 tokens. I add the sampled token
y and deduplicate. I reconstruct

    q~(v) = q(v)                          for v in K,
    q~(v) = q_rest * p(v) / p(not K)      otherwise,   q_rest = 1 - q(K),

so that the centering term is computable without the full q:

    sum_v q~(v) log p(v) = sum_{v in K} q(v) log p(v) + q_rest * log p(not K) + const,

where log p(not K) is a logsumexp over the vocabulary with K masked out
(exact, with no `1 - p(K)` cancellation). Its logit gradient is q~ - p. The
drift that remains is

    E_{y~q}[ e_y - q~ ] = q - q~,

which is zero on K and on the tail has total mass at most 2 q_rest. For an
LLM, the top-20 tokens of the sampler cover essentially all of q's mass at
almost every position, so the drift is removed up to that tiny remainder.
If top-k logprobs are missing, K = {y}, and the drift is still reduced
substantially (in a 12-token test: max |drift| 0.10 for REINFORCE, 0.017
with top-4, 0.032 with only y, and 1e-9 with full K).

## 4. Implementation (`losses.py`)

* `sampler_baseline(logp_vocab, logp, b, positive, c)` returns
  `sum_v m(v) log p(v)`, with m = q~ detached (or q~ w, see below). It
  works on each chunk of the vocabulary-sized log-softmax that the loss
  already computes, and gathers K with `take_vocab` (TP-safe).
* In `loss_fn_rl` the per-token score `w * logp` becomes
  `w * logp - baseline`. Everything else is unchanged: per-token
  normalization, sequence-level weights and metrics.
* The signature and default behaviour are preserved, and
  `correction=None` is the intended setting.

## 5. With an IS correction (optional)

With a token weight w = f(p/q), the expected score is
`E_q[w grad log p] = sum_v q(v) w(v) grad log p(v)`. So the matching
centering mass is m(v) = q~(v) w(v), which is `exp(log_mass(...))` of the
existing correction grid, evaluated at every v in K and on the tail block.
The term `w(y) log p(y) - sum_v sg(m(v)) log p(v)` then has zero expected
gradient for every correction, including clamped, dropped, PPO-sign and DPPO
variants. Clipping no longer causes drift; it only changes which direction
is followed. Sequence-level weights scale the whole centered score.
Without this generalization, TIS on top of the centered score collapsed.
With it, all the corrections I tried were stable. The uncorrected version
(the default) trained best.

## 6. Check on the CPU proxy (`dev/mini.py`)

Default setting (noise 0.3, plus_minus rewards, 600 steps). The final eval
reward of the clean policy:

| loss | seed 0 | seed 1 |
| --- | --- | --- |
| REINFORCE (old) | 0.66 (train reward stuck around 0.58) | 0.66 |
| centered (new) | 0.88 | 0.82 |

With 200 extra "unlikely" vocabulary tokens (so top-20 is only partial),
the old loss collapsed to 0 reward within 100 steps on 2 of 2 seeds. The new
loss reached 0.70 to 0.77 eval reward with no collapse. At noise 0.5 the old
loss collapsed to 0. The new one kept improving for 1200 steps.

## Configuration

None needed: the fix is active with the default `rl.is.level=null`, which is
the recommended setting.
