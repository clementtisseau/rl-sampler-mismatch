# Sampler-centered score: removing the mismatch bias from REINFORCE

## 1. Setup and notation

At completion position t with prefix x_t the trainer has logits z_t and policy
p = softmax(z_t). The rollout token y_t is drawn from the sampler q(.|x_t)
(noisy, quantized, KV-quantized or stale weights). The current loss
(`correction=None`) has gradient

    g = sum_i A_i sum_t w_{i,t} grad log p(y_{i,t} | x_{i,t}),   w = 1 (or w(p/q) with a correction)

The calculations below are per position and in logit space, where
grad_z log p(y) = e_y - p (e_y is a one-hot vector). The parameter gradient is
J_t^T times this, with J_t = dz_t/dtheta.

## 2. What is wrong with the current loss

REINFORCE stays unbiased under any baseline only because the score has zero
mean **under the distribution the token was drawn from**:

    E_{y~p}[grad_z log p(y)] = sum_y p_y (e_y - p) = 0.

The rollouts come from q, though, and under q the score does not have zero mean:

    E_{y~q}[grad_z log p(y)] = sum_y q_y (e_y - p) = q - p            (1)

In parameter space this is grad_theta sum_y q_y log p_theta(y) = -grad_theta KL(q || p_theta).

Split the advantage into its prefix-conditional mean and the remainder:
A = V(x_t) + (A - V(x_t)) with V(x_t) = E[A | x_t]. Taking expectations of g:

    E[g] = sum_t E[ J_t^T ( sum_y q_y (Q(x_t,y) - V(x_t)) e_y ) ]       (the learning signal)
         + sum_t E[ V(x_t) J_t^T (q_t - p_t) ]                         (2: mismatch term)

with Q(x,y) = E[A | x, y]. The second term is the bias. It has these properties:

* **It does not depend on the learning signal.** It is present even when every
  token is equally good (Q = V), where the correct update is zero. Rewards
  that carry no information still move the policy.
* **Its sign is the sign of the baseline.** If V < 0 (most rollouts fail, or
  `plus_minus` / `zero_minus`), the loss ascends KL(q || p) and pushes p away
  from the sampler: it lowers p wherever q > p and sharpens p wherever p > q.
  If V > 0, p chases q.
* **It is systematic and it compounds.** The mismatch is a fixed
  transformation of the trainer (fixed noise delta, a fixed quantizer), so
  q - p ~ J delta points in a consistent direction at every step. Because
  the sampler is re-synced from the trainer every step, the gap never closes.
  Each step adds roughly eta V J^T J delta along the same direction, so the
  drift grows linearly over training until the policy degenerates. Zero-mean
  noise would average out; this does not.
* **Group-centering does not remove it.** sum_i A_i = 0 within a group cancels
  (2) only at the first completion token, where all group members share x_t.
  At every later position the prefixes differ and V(x_t) = E[A | x_t] != 0
  (a prefix that already contains a mistake has negative value). The bias is
  weaker but still systematic, and it acts on hundreds of tokens per sequence.
* Tokens where q puts mass that p does not (the tail of a large vocabulary
  under noise or quantization) are where |q - p| and the bias are largest.

### Why IS corrections and clipping only slow it down

With a detached token weight w(r), r = p/q, write m_v = q_v w(p_v/q_v),
Z = sum_v m_v and mbar = m/Z (this is exactly `log_mass` in `losses.py`). Then

    E_{y~q}[w(y) (e_y - p)] = m - Z p = Z (mbar - p).                   (3)

The mismatch term becomes V Z (mbar - p). It vanishes only when mbar = p,
which requires the exact, untruncated ratio w = p/q. That ratio has unbounded
variance and is unavailable at sequence level. Every truncated or clipped
weight (TIS, MIS, PPO, DAPO, DPPO) leaves mbar strictly between q and p, so the
bias shrinks but keeps the same direction and still compounds.

Tighter token-level clipping can make it worse. With a band [1-eps, 1+eps],
every token whose ratio leaves the band gets m_v = q_v(1 +/- eps) (clamp) or 0
(drop) instead of p_v. A tighter band affects more tokens, so mbar moves back
towards q, or with drop towards a renormalized distortion of p that can be
further from p than q is. The `ppo` sign rule makes m depend on sign(A), so
even the cancellation between positive and negative advantages in (2) is lost.

## 3. The modification: center the score under the sampling distribution

Each token's score is replaced by its centered version under the distribution
it is actually drawn from (q, reweighted by whatever weight w the correction
applies):

    s_t = w(y_t) [ grad log p(y_t) - sum_v mbar_v grad log p(v) ],   mbar ∝ q_v w(p_v/q_v) (detached)

In logit space s_t = w(y_t)(e_{y_t} - mbar). As a surrogate loss this is

    loss = - sum_i A_i sum_t w_{i,t} [ log p(y_{i,t}) - b_{i,t} ],   b_t = sum_v sg(mbar_v) log p(v)

so the change is that b_t is subtracted inside the score. Since grad_z b_t = mbar - p, the effect is to replace the
centering point p of the REINFORCE score by mbar. With `correction=None`, w = 1
and mbar = q.

### Derivation that it removes the bias

    E_{y~q}[ s_t ] = sum_y q_y w_y (e_y - mbar) = m - Z mbar = 0       (4)

This holds exactly, for any weight function w, including none, clamps, drops
and sign-dependent bands. Consequences:

1. **Baseline invariance is restored.** For any c(x_t) that depends only on the
   prefix, E[c(x_t) s_t] = E[c(x_t) E[s_t | x_t]] = 0. The mismatch term (2)
   vanishes identically, and the expected update is

       E[g] = sum_t E[ J_t^T sum_y m_y (Q(x_t,y) - <Q>_mbar) e_y ].       (5)

   Shifting the rewards by a constant (`plus_minus`, `plus_zero`,
   `zero_minus`, `group_centered`) no longer changes the expected update.
2. **No drift without signal.** If Q(x, .) is constant (nothing to learn at a
   token), (5) is exactly zero, whatever q is. The update is driven only by
   reward differences between tokens. The compounding mechanism of Section 2
   needs a nonzero update when there is no signal, and that update is now gone.
3. **The remaining mismatch is second order.** Compared with the on-policy
   gradient p ⊙ (Q - <Q>_p), (5) is the same improvement direction with
   per-token weights m instead of p. The error is proportional to
   (q - p) x (Q - <Q>), a product of two small quantities that vanishes with
   the learning signal. The old error is (q - p) x V and does not vanish.
   Near p = q, (5) is a positive reweighting of the true policy gradient and
   remains an ascent direction.
4. **The method strictly generalizes the existing estimators.** On-policy
   (q = p) gives mbar = p and recovers REINFORCE exactly. With the full ratio
   w = p/q, mbar ∝ q (p/q) = p and it recovers naive token-level IS exactly.
   For truncated weights it adds the missing correction Z(mbar - p) from (3).
5. **Variance.** Subtracting a per-token control variate with zero mean under
   the sampling distribution does not add variance. Here it also removes the
   large V(x_t)(e_y - p) component, so the gradient norm tends to drop.

### Implementation details (`losses.py`)

* Inference engines (and `train_rl.py` / `dev/mini.py`) return the sampler's
  top-k next-token logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`,
  k = 20). On the top-k set K, m_k = q_k w(p_k/q_k), computed by the existing
  `log_mass` in log space.
* For the tail outside K, only its mass Q_tail = 1 - sum_K q is known. It is
  spread in proportion to the trainer's tail, q_v = p_v Q_tail / P_tail. All
  tail tokens then share the ratio P_tail / Q_tail, so ratio bands treat the
  tail exactly as one outcome with mass m_tail = Q_tail w(P_tail/Q_tail). The
  tail term of the baseline is (mbar_tail / P_tail) sum_{v not in K} sg(p_v) log p_v.
  It is computed as the full-vocabulary sum sg(p) . log p minus the top-k part,
  inside the existing rematerialized vocab chunks, so no extra [B, T, V] memory
  is used. If this assumption about the tail shape is wrong, the error is at
  most Q_tail times the distance between the true and assumed tail shapes,
  which is small because the top 20 tokens carry almost all the mass. For the
  DPPO TV band each tail token's |q - p| is tiny, so the tail is treated as in-band.
* Numerical safeguards: the ratios are clipped to e^{+/-20}; P_tail is floored
  at 1e-6 in the tail coefficient (the tail gradient is then negligible
  anyway); if every weight is zero, Z is set so that no baseline is subtracted.
* The modification is on by default (`correction=None`). It also composes
  with every token-level `rl.is.*` correction through mbar ∝ q w. With
  random p, q and every correction family I checked numerically that
  E_{y~q}[s] = 0 to float precision (1e-8).
* Limitations: with sequence-level corrections (`seq_mean`, `seq_sum`) the
  weight of a token depends on the ratios of all the other tokens and on
  selection over the whole sequence. Then there is no per-token centering
  distribution that makes (4) hold, so the loss keeps the plain p-centered
  score there and behaves exactly as before. Without top-k sampler logprobs
  in the batch it also falls back to the plain score.

## 4. Sanity checks (dev/mini.py, CPU; not required, given for context)

Columns are the final eval reward (mean training reward over the last 20 steps).

| setting | original loss | centered score |
| --- | --- | --- |
| default (noise 0.3, plus_minus), seed 0 | 0.66 (0.58) | 0.82 (0.87) |
| default, seed 5 | 0.73 (0.60) | 0.88 (0.87) |
| + 40 extra vocab tokens, seed 1 | 0.00, collapsed | 0.80 (0.82) |
| noise 0.5, 40 extra tokens | 0.00, collapsed | 0.58 (0.54) |
| noise 0.5, 40 extra tokens, group_centered | 0.49 (0.44), declining | 0.61 (0.50) |
| noise 0.5, 40 extra tokens, TIS | 0.00, collapsed | 0.67 (0.31) |
| noise 0.5, 40 extra tokens, PPO | 0.00, collapsed | 0.26 (0.17) |
| noise 0.3, 100 extra tokens, group_centered | 0.75 (0.70) | 0.72 (0.65) |

No configuration changes are needed: the method is active with the default
`rl.is.level=null`.
