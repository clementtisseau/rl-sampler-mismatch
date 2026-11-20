# Sampler-centered REINFORCE: removing the baseline-induced drift under sampler/trainer mismatch

## Summary

The loss in `losses.py` is now REINFORCE in which every token's score is
**centered under the sampler's own next-token distribution** q:

    loss = - (1/N) Σ_i A_i Σ_t [ w_t log p(y_t)  -  Σ_v sg(q_t(v) w_t(v)) log p_t(v) ]

(w ≡ 1 without a correction). The subtracted term is a control variate: its
gradient is exactly the expectation, under y ~ q, of the gradient of the first
term, so the per-token update has zero mean under the sampler whatever the
advantage offset. q is read from the sampler's top-k logprobs
(`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20, which the trainer already
records); the remaining vocabulary is one bucket. It is active with default
arguments, works with `correction=None` and with every `ISCorrection`. No
configuration change is needed.

## 1. What is wrong with the current loss

Notation: at a prefix (state) s, the trainer has logits z, p = softmax(z); the
sampler draws y ~ q (q ≠ p because of noise, quantization, staleness). The
token's score in logit space is

    ∇_z log p(y) = e_y - p .

The current loss is −A Σ_t log p(y_t) with A the (group-centered) sequence
advantage. REINFORCE relies on one identity: **the score has zero mean under
the sampling distribution**, E_{y~p}[e_y − p] = 0, which is what makes a
baseline legal (it changes variance, not the expected update). Under the
sampler this identity fails:

    E_{y~q}[ e_y - p ] = q - p  =  -∇_z KL(q ‖ p)                      (1)

(indeed KL(q‖p) = Σ q log q − Σ q log p and ∇_z(−Σ_v q_v log p_v) = p − q).

Write the advantage at token t as A = Ā(s) + (A − Ā(s)), where
Ā(s) = E_q[A | s] = V^q(s) − b is the expected advantage of the prefix
(the value of the state under the sampler, minus the group baseline b).
The expected per-token update of the current loss at state s is

    E_q[A (e_y - p) | s]
      = Σ_y q(y) (Q^q(s,y) - V^q(s)) e_y          (a)  policy-gradient signal
      - Ā(s) ∇_z KL(q ‖ p)                         (b)  mismatch term        (2)

Derivation: E_q[A(e_y − p)] = Σ_y q_y A_y e_y − (Σ_y q_y A_y) p
= Σ_y q_y (A_y − Ā) e_y + Ā q − Ā p, with A_y = Q^q(s,y) − b.

Term (a) is the correct on-policy logit-space gradient (for the sampler's
policy). Term (b) is a **systematic, non-zero-mean bias**, proportional to how
far the state's value is from the baseline, and pointing along the KL gradient:

- In states that do worse than the baseline (Ā < 0) the update **ascends**
  KL(q‖p): the trainer is pushed *away* from the sampler. Concretely, logits of
  tokens that q over-samples relative to p (typically the many low-probability
  tokens whose mass noise/quantization inflates) are pushed further down, and
  those q under-samples are pushed up: p sharpens away from q.
- In states that do better (Ā > 0) it descends KL, pulling p toward q.

Neither has anything to do with the reward: it is a force that exists only
because q ≠ p. With group-centered rewards Ā is zero on average over the
group's *first* token, but not per state: failing trajectories visit many
low-value prefixes (Ā < 0), and with `plus_minus`/`zero_minus`-like offsets
Ā is non-zero almost everywhere. The bias does not average out over steps,
because the sampler is re-derived from the trainer every step: q = Φ(p) for a
fixed perturbation Φ (noise, int8/fp8, KV quantization) — so q − p is a
deterministic function of the current weights and term (b) points in a
consistent direction step after step. Moving p away from q does not shrink the
gap; the next sampler is Φ(p_new), which re-creates (and, as p becomes sharper
and its low-probability logits more extreme, usually enlarges) the
mismatch. The process compounds: KL(sampler‖trainer) grows, (b) grows with it,
and training collapses. This matches the observations that the bias is
systematic and that it accumulates.

Why IS corrections and clipping only slow it down: with a token weight
w = p/q one gets E_q[w(e_y − p)] = Σ q·(p/q)(e_y − p) = 0, so the exact ratio
does remove (b) — but the exact ratio has unbounded variance, so all practical
methods truncate/clamp/drop it. For any truncated weight w̄,

    E_q[ w̄ (e_y - p) ] = c - (Σ_v c_v) p,   c_v = q_v w̄_v ,               (3)

which is non-zero as soon as w̄ ≠ p/q on any token, so Ā·(3) remains a
systematic drift. Clipping removes exactly the tokens where |p/q − 1| is large,
i.e. where the bias lives, and *tighter* token clipping makes c deviate more
from p, which is why stronger clipping can make things worse. Sequence-level
weights multiply A but leave each token's E_q[e_y − p] = q − p untouched.

## 2. The modification

Subtract from every token's (possibly IS-weighted) score its expectation under
the sampler, computed exactly over the vocabulary rather than estimated from
the one sampled token:

    g_t(y) = w(y) log p(y) - CV,     CV = Σ_v sg( q(v) w(v) ) log p(v),     (4)

where w(v) = w(p(v)/q(v)) is the correction's weight (w ≡ 1 for
`correction=None`), and sg is stop-gradient. Then

    ∇_z CV = Σ_v c_v (e_v - p) = c - (Σ c) p = E_{y~q}[ w(y)(e_y - p) ]   by (3),

so the gradient of g_t has **zero mean under q for any advantage that does not
depend on y**:

    E_q[ (A_y)(∇_z g_t) ] = Σ_y q_y w_y (A_y - Ā_w) e_y  - (Σ_y q_y w_y (A_y - Ā_w)) p,
    Ā_w = Σ q w A / Σ q w.                                                (5)

For w ≡ 1 this is exactly term (a) of (2): Σ_y q_y (Q^q − V^q) e_y − 0·p,
i.e. the logit-space update becomes e_y − q instead of e_y − p:

    E_q[ A (e_y - q) ] = Σ_y q_y (A_y - Ā) e_y = q ⊙ (Q^q(s,·) - V^q(s)).  (6)

Properties:

1. **Baseline invariance is restored.** Adding any constant (or any function
   of the prefix) to A leaves the expected update unchanged, since
   E_q[e_y − q] = 0. The mismatch term Ā·∇KL(q‖p) of (2) is gone exactly — not
   damped as with clipping. The trainer is no longer pushed away from (or
   pulled toward) the sampler for reasons unrelated to reward.
2. **What remains is a correct policy-gradient direction.** (6) is the exact
   gradient, with respect to the logits, of E_{y~q}[A] for the softmax policy
   q — the on-policy policy gradient of the sampler at that state — applied to
   the trainer's logits through the trainer's Jacobian. Because q = Φ(p) is a
   small fixed perturbation of p, an improvement of the logits that improves
   q improves p (to first order the perturbation is constant and
   logit updates pass through), so there is no direction in which errors
   systematically accumulate. Remaining discrepancies (state distribution and
   advantages come from q's rollouts) are second order in the mismatch and,
   crucially, carry the sign of the reward signal, not of Ā.
3. **On-policy it is a no-op in expectation.** If q = p, CV = Σ p log p and
   ∇CV = p − p = 0: the loss gradient is identical to standard REINFORCE.
   So nothing is lost when the engines agree, and the correction scales with
   the mismatch.
4. **Lower variance, not higher.** Unlike importance weights, CV involves no
   ratios of the sampled token; it is a deterministic function of the state
   (q and p on the vocabulary). It is a standard control variate, so it can
   only remove variance correlated with the score.
5. **Composable with any correction.** With an ISCorrection the same
   construction uses c_v = q_v w(p_v/q_v) and cancels the residual bias (3) of
   the truncated weight; the expectation becomes the w-weighted on-policy
   signal (5). Sequence-level weights scale A and are unaffected.

Loss-level form (w ≡ 1). g_t = log p(y) − Σ_v q(v) log p(v) = log p(y) +
H(q, p): REINFORCE on the sampled token plus a per-token cross-entropy toward
the sampler weighted by −A. The latter exactly cancels the KL-shaped drift
(1)–(2) in expectation.

## 3. Using only the sampler's top-k

The sampler records the top-k logprobs (k = 20) per position, as real
inference engines return. Let K be those tokens and R the rest, with masses
q_R = 1 − Σ_K q and p_R = 1 − Σ_K p. We treat R as a single bucket whose
sampler mass is distributed like the trainer's:

    c_v = q_v w(p_v/q_v)                   v ∈ K
    c_u = q_R w(p_R/q_R) · p_u / p_R        u ∈ R.

The CV is still Σ_v sg(c_v) log p(v). Its tail part is computed without any
[V]-sized scatter: Σ_v sg(p_v) log p(v) has zero gradient (Σ_v p_v ∇log p_v =
p − p = 0), so

    Σ_{u∈R} sg(p_u) log p(u) = Σ_v sg(p_v) log p(v) − Σ_{v∈K} sg(p_v) log p(v),

and the tail contribution is sg(q_R w_R / p_R) times that. With w ≡ 1 the
resulting logit gradient is e_y − q̃ with q̃ = q on K and q_R p_u/p_R on R;
the residual bias is q_R · (q_{·|R} − p_{·|R}), confined to the tail and
bounded by its sampler mass q_R (small for k = 20; zero in `dev/mini.py`).
The tail weight is bounded too: each tail logit receives at most
q_R p_u/p_R ≤ q_R. p_R is floored at 1e-6 (float32 resolution of 1 − Σ_K p).
If no top-k is present in the batch, K is the sampled token alone (the CV then
corrects the sampled direction and the rest as one bucket; it is still exact
when q = p).

## 4. Implementation (`losses.py`)

- `sampler_centered_terms` computes, per position, the plain term w_y log p(y)
  and the control variate from (4) using the top-k logprobs gathered from the
  trainer's log-softmax (`take_vocab`, TP-safe) and the sampler's top-k
  logprobs; weights reuse `is_weight` so all `ISCorrection` bands, signs and
  TV statistics apply per candidate token.
- `loss_fn_rl` keeps its signature and chunked/rematerialized structure; the
  per-sequence sum of centered terms replaces the score, multiplied by the
  (optionally sequence-weighted) advantage. Masked positions use `jnp.where`
  so padding cannot inject NaNs. A metric `centering` reports the mean control
  variate.

## Sanity check (CPU dev setting, `dev/mini.py`, 600 steps)

Not needed for the argument, only a quick check that the implementation behaves
as the derivation predicts: with 40 extra tail tokens (`MINI_EXTRA_TOKENS=40`,
so the top-20 does not cover the vocabulary and the tail bucket is exercised),
plain PG with the original loss collapses to 0 eval reward by step 100, while
the centered loss climbs steadily to about 0.8 (0.83 on the default
vocabulary, where the original loss stalls near 0.7). It also stays stable
with TIS and when the top-k keys are removed (sampled-token fallback).

No configuration change is needed: the modification is on by default, also
with `rl.is.level=null`.
