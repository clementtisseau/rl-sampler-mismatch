# Sampler-centered REINFORCE: removing the drift caused by a mismatched sampler

## Summary

Tokens are sampled from the sampler `q`, but the loss uses the trainer's score
`grad log p(y)`. Under `q` that score does not have mean zero. It has mean
`q - p` in logit space. So every advantage that does not average to zero
moves the policy along `+-(q - p)`, even when the reward says nothing about
the token. This is a deterministic vector field over the parameters. It does
not depend on the reward, and because the sampler is rebuilt from the trainer
at every step it never averages out. Instead it accumulates.

The fix replaces the score of each sampled token with a score centered under
the sampler:

    grad log p(y_t)   ->   grad log p(y_t) - E_{v ~ m_t}[grad log p(v)],
    m_t(v) ∝ q_t(v) w(p_t(v)/q_t(v))   (m_t = q_t when there is no correction),

that is, in code, `score = w * (log p(y) - sum_v sg(m_v) log p_v)`. This score
has mean zero under the sampler at every prefix, exactly, for any detached
weight `w`. The drift term therefore disappears, and the learning signal (the
covariance between advantage and token) is kept. The loss uses the sampler's
top-k next-token logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`), which
both `train_rl.py` and `dev/mini.py` already put in the batch. The
modification is on with default arguments. With `correction=None`, `m = q`.

## 1. What is wrong with the current loss

Notation: at position t with prefix s_t the trainer has logits z_t and
p_t = softmax(z_t). The sampler draws y_t ~ q_t. In logit space the per-token
score is

    ∇_{z_t} log p_t(y_t) = e_{y_t} - p_t .

The current loss (no correction) gives the update

    g = (1/N) Σ_i A_i Σ_t (e_{y_it} - p_it) .

Split the score into a part centered under the sampler and a remainder:

    e_y - p = (e_y - q) + (q - p),       E_{y~q}[e_y - q | s_t] = 0 .

Taking expectations over the sampler's rollouts:

    E[g] = E[ Σ_t A (e_{y_t} - q_t) ]   +   E[ Σ_t Ā(s_t) (q_t - p_t) ]
           ---------------------------      -----------------------------
           signal: Σ_t Cov_q(A, e_{y_t})    drift D

where Ā(s_t) = E_q[A | s_t] is the mean advantage of rollouts that pass
through prefix s_t. With the identity q - p = -∇_z KL(q || p) for fixed q:

    D = - Σ_t E[ Ā(s_t) ∇_{z_t} KL(q_t || p_t) ] .

What this means:

* **It is a bias, not noise.** D does not depend on whether the sampled token
  was good. It stays when the advantage is independent of y_t, and it is zero
  only when q = p. On-policy (q = p), `E_p[e_y - p] = 0` and D = 0. That is
  why REINFORCE needs no correction there.
* **It has a fixed direction.** At prefixes with Ā < 0 (failing continuations,
  which is the common case when reward is low and failed answers are long), the
  update ascends KL(q||p). It pushes p away from the sampler. At Ā > 0 it pulls
  p toward the sampler. Group centering makes Σ_i A_i = 0 per group, but not
  Ā(s_t) per prefix and not Σ_i A_i·len_i (failed rollouts are usually
  longer), so it does not cancel D.
* **It compounds.** q is a fixed function of the trainer's weights (weights
  plus fixed noise, quantized weights or KV, a stale copy), refreshed every
  step. So q - p = F(θ) is a vector field over parameters, and D integrates
  along the training trajectory instead of averaging out.
* **Mechanism, per token.** On-policy, a negative-advantage rollout raises
  every token it did not sample by |A| p_v. That rise is cancelled because v
  is itself sampled, and pushed down, with frequency p_v. Under the sampler, v
  is sampled with frequency q_v instead. A token with p_v > q_v (the trainer
  likes it more than the sampler does) then rises at net rate |A|(p_v - q_v).
  It is under-sampled, so this is rarely corrected. The rise becomes a runaway
  once the token is sampled and penalized too rarely to balance it. In a large
  vocabulary there are many such low-probability tokens. In `dev/mini.py`
  with extra unused tokens, the current loss sends the policy's mass to them
  (tail mass goes to 1 and reward to 0).

**Why IS corrections and clipping only slow this down.** With a detached
token weight w = w(p/q) the update is A·w·(e_y - p), and

    E_{y~q}[ w (e_y - p) ] = Σ_v q_v w_v (e_v - p) = Z (m - p),
    m_v = q_v w_v / Z,  Z = Σ_v q_v w_v .

The drift is now along m - p instead of q - p. Only the untruncated ratio
(w = p/q) gives m = p and no drift, and its variance is unbounded. Clipping or
dropping makes m a mixture between q and p, so part of the drift remains.
Tighter token-level bands push m back toward q and can move it further from p
(for example, dropping tokens where p >> q removes exactly the mass that
should balance them). So stronger clipping can make things worse.
Sequence-level weights rescale A and do not change the token score, so they
leave the q - p direction in place.

## 2. The fix: center the score under the distribution that was sampled

For any detached weight w (including w ≡ 1), define

    m_t(v) = q_t(v) w(p_t(v)/q_t(v)) / Σ_u q_t(u) w(p_t(u)/q_t(u)) ,

which is `log_mass(log p, log q)` from `losses.py`, normalized over the
vocabulary. The new per-token score is

    ψ_t = w(y_t) [ ∇ log p_t(y_t) - Σ_v m_t(v) ∇ log p_t(v) ]  =  w(y_t) (e_{y_t} - m_t)   (logit space).

**Zero drift, exactly.** For every prefix,

    E_{y~q}[ w(y)(e_y - m) ] = Σ_v q_v w_v e_v - (Σ_v q_v w_v) m = Z m - Z m = 0 .

So when the advantage carries no information about y_t, the expected update
is exactly zero, whatever the mismatch. The drift D is gone. With no
correction (w ≡ 1, m = q) the expected update is

    E[g] = Σ_t E[ Cov_q(A, e_{y_t} | s_t) ] ,

which is pure signal. It differs from the on-policy gradient Σ_t Cov_p(A, e_y)
only because the covariance is taken under q instead of p. That difference is
first order in (q - p) times the signal, and it vanishes whenever the signal
does. It has no fixed direction that could accumulate. The centered term is
a state-dependent baseline (a control variate) for the score itself. A
baseline that does not depend on the action leaves the gradient unbiased only
if the score has mean zero under the sampling distribution. Here we choose
the baseline that makes that true for q.

Consistency checks:
* q = p: m = p and ψ = e_y - p. This is plain REINFORCE, so nothing changes
  on-policy.
* Untruncated IS, w = p/q: m ∝ q·(p/q) = p, so the loss reduces to naive IS,
  which already has no drift. Clipped or dropped IS (TIS, MIS, PPO-style
  bands, DPPO) keeps its variance control and has no drift with m in place
  of p.
* It is a pure loss change. The rollout, the advantages and the optimizer are
  unchanged. Only one scalar per position is subtracted from log p(y).

**As a loss.** The gradient of Σ_v sg(m_v) log p_v with respect to z is
m - p. So `w * (log p(y) - Σ_v sg(m_v) log p_v)` has gradient
w (e_y - p - (m - p)) = w (e_y - m), as required. I checked numerically that
the autodiff gradient equals -A w (e_y - m)/N to within 1e-8, with and
without corrections.

## 3. Using only the sampler's top-k logprobs

Inference engines return the top-k logprobs (k = 20), not the full q. Let K
be the top-k set, q_rest = 1 - Σ_{k∈K} q_k, and p_rest = 1 - Σ_{k∈K} p_k.
Outside K, take q to have p's shape:

    q̂_v = q_v (v ∈ K),      q̂_v = p_v q_rest / p_rest (v ∉ K).

The tail mass is therefore exact. The ratio is constant on the tail,
p/q̂ = p_rest/q_rest, so the tail of m is

    m_tail = q_rest w(p_rest/q_rest) / Z,  Z = Σ_{k∈K} q_k w(p_k/q_k) + q_rest w(p_rest/q_rest),

with m_v = m_tail p_v/p_rest inside the tail. The tail sum does not need the
full vocabulary, because Σ_v p_v ∇log p_v = 0 gives

    Σ_{v∉K} p_v ∇log p_v = - Σ_{k∈K} p_k ∇log p_k ,

so the centering term is

    Σ_{k∈K} sg(m_k) log p_k + sg(m_tail/p_rest) (sg(Σ_v p_v log p_v) - Σ_{k∈K} sg(p_k) log p_k).

The detached full sum Σ_v p_v log p_v is already computed for the entropy
metric. It only fixes the loss value and has zero gradient. The remaining
error is a drift Ā·(q - q̂) that lives only inside the tail and sums to zero.
It changes the shape of the tail but not its mass. The tail mass is exactly
the quantity the collapse feeds on. q_rest and p_rest are computed as
-expm1(logsumexp(·)) and floored at 1e-6, because a near-empty tail is float32
noise. Log ratios are clipped to ±20, as in `is_weight`. For DPPO's TV band,
each tail token has TV ≤ p_v, which is tiny, so the tail is treated as in-band.
If the batch has no top-k logprobs, the loss falls back to m = p, which is
the old loss.

## Implementation (`losses.py`)

* `sampler_centering(...)` returns Σ_v sg(m_v) log p_v per position, computed
  inside the existing rematerialized vocabulary chunks. It reuses `log_mass`,
  so every point on the `rl.is.*` grid gets its matching m. The sign is taken
  from the advantage for `sign=ppo/neg/pos`.
* In `loss_fn_rl`, `score = w * (logp - center)`. The signature, the batch
  layout, the metrics and the sequence-level corrections are unchanged. It is
  on by default with `correction=None` (m = q).
* Caveat: with sequence-level weights W (`seq_mean`, `seq_sum`), W depends on
  y_t through the ratio, so E[W(e_y - q)] is only approximately zero. The
  q - p drift is still removed, and the default (no correction) is exact.

## Dev check (CPU `dev/mini.py`, 600 steps, mean reward over the last 10 steps)

| setting | current loss | centered loss |
|---|---|---|
| default (noise 0.3, plus_minus), seed 0 / 1 | 0.58 / 0.61 (stalls) | 0.86 / 0.85 |
| 40 extra vocabulary tokens | 0.00 (collapse: tail mass → 0) | 0.82 |
| 200 extra tokens, noise 0.5 | 0.00 (collapse: tail mass → 1) | 0.49 |
| 40 extra tokens, TIS (high=2, clamp) | 0.59 | 0.66 |
| group_centered, fixed length (little drift) | 0.82 | 0.81 |

The last row matches the theory. With equal-length rollouts and centered
advantages, Ā is close to zero, so there is little drift to remove and the
two losses perform the same.

No configuration changes are needed: the method is meant to run with the
default `rl.is.level=null`.
