# Sampler-centred REINFORCE: removing the mismatch drift from the RL loss

## 1. Setting and notation

At one completion position, with prefix s fixed, the trainer has logits z ∈ R^V and
next-token distribution p = softmax(z); the sampler that produced the rollout
has a different distribution q (noisy / quantized / stale weights, quantized
KV cache). The token y is drawn from **q**. The sequence gets an advantage A
(group-centred reward). Let

    A(y) := E[A | s, y]        (future tokens also come from q)
    Ā_q  := Σ_y q(y) A(y),     Ā_p := Σ_y p(y) A(y).

The current loss is, per token, `-A · w · log p(y)` (w = 1 without a correction),
summed over positions and divided by the number of tokens. Its gradient with
respect to the logits at that position is

    g(y) = A · w · ∇_z log p(y) = A · w · (e_y − p),          (1)

where e_y is the one-hot vector of y (∇_z log softmax(z)_y = e_y − p). Every
parameter gradient is J_θᵀ g, with J_θ = ∂z/∂θ fixed given the prefix, so it
suffices to analyse g.

## 2. What is wrong with the current loss

### On-policy (q = p)

    E_{y~p}[A(y)(e_y − p)] = Σ_y p(y)A(y) e_y − p Ā_p = p ⊙ (A − Ā_p).     (2)

Every component is p(v)(A(v) − Ā_p): a token is pushed up iff it is better
than average *under the distribution that is being updated*. In particular,
the part of the advantage that does not depend on y (a constant c, e.g. the
baseline error, or the fact that the advantage of a sequence is shared by all
its tokens) contributes c · E_p[e_y − p] = c (p − p) = 0. This is the usual
"a baseline does not bias REINFORCE" property; it is what makes group
centring, ±1 rewards, sequence-level advantages etc. harmless: they only
change the variance.

### Mismatched (q ≠ p)

    E_{y~q}[A(y)(e_y − p)] = q ⊙ A − p Ā_q
                           = q ⊙ (A − Ā_q)  +  Ā_q (q − p).                  (3)
                             └ signal ┘       └──── drift ────┘

The first term is the same useful signal as (2) with q in place of p. The
second term is new: it is proportional to the **prefix-conditional mean
advantage Ā_q, not to any difference between tokens**, and it points along
q − p. Since ∇_z KL(q‖p) = p − q,

    drift = Ā_q (q − p) = − Ā_q ∇_z KL(q ‖ p).

So the loss secretly contains, at every token, the extra objective
"− Ā_q · KL(q‖p)": with a positive mean advantage it pulls p towards the
sampler, with a negative one it pushes p away from the sampler. Properties:

* It is **not zero-mean noise**: it does not vanish with more samples, its
  size is |Ā_q|·‖q − p‖ and nothing in the advantage cancels it, because Ā_q
  is a per-prefix quantity (all tokens of a failing sequence share A < 0;
  group centring makes advantages zero-mean only across the group, at the
  prompt, not at each later prefix; `plus_minus` rewards are not centred at
  all).
* It **compounds**. The sampler is rebuilt from the trainer each step with the
  same systematic perturbation (q_θ = p_{θ+δ}, quantize(θ), ...). Moving p
  towards (or away from) q moves θ along +δ (or −δ) in the Fisher metric
  (∇_θ KL(p_{θ+δ}‖p_θ) ≈ −F δ); the new sampler is again offset by δ from the
  new trainer, so the same push is applied again: a deterministic drift of
  roughly η Ā_q F δ per step, which accumulates linearly and eventually
  dominates the reward signal (collapse). For the "away from q" sign
  (Ā_q < 0, e.g. failing rollouts) there is not even a fixed point: tokens with
  q > p are pushed down further, increasing the mismatch.

### Why IS corrections and clipping only slow it down

With a detached weight w(y) = w(p(y)/q(y)):

    E_{y~q}[w A (e_y − p)] = (q⊙w) ⊙ A − p Σ_y q(y)w(y)A(y),

whose constant-advantage part is c · (q⊙w − p · E_q[w]). It is zero iff
q⊙w ∝ p, i.e. only for the untruncated ratio w = p/q. Every clipped,
truncated, masked or sign-dependent weight (TIS, MIS, PPO/DAPO, DPPO, TOPR...)
leaves a drift c·(m − |m| p) with m = q⊙w: e.g. for TIS m = min(p, Cq), the
drift lives exactly on the tokens with p > Cq, and tighter clipping (smaller C)
makes m further from p, i.e. a *larger* drift — which is why stronger
token-level clipping can make things worse. The untruncated ratio removes the
drift only in expectation, with variance E_q[(p/q)²] − 1 = χ²(p‖q) that
explodes on rare tokens; sequence-level ratios are products of these and are
worse. So all of them reduce but do not remove the systematic term.

## 3. The modification: centre the score under the sampler

The drift exists because the score e_y − p is centred under p while y comes
from q. The fix is to centre it under the distribution the token was actually
drawn from. Per token, replace log p(y) by

    log p(y) − Σ_v c_v log p(v),      c detached,  c = q  (no correction).      (4)

Its logit gradient is (e_y − p) − (c − p) = **e_y − c**. With c = q:

    E_{y~q}[A(y)(e_y − q)] = q ⊙ A − q Ā_q = q ⊙ (A − Ā_q).                  (5)

* The drift term of (3) is gone exactly, for every q, every prefix and every
  advantage scheme: a y-independent advantage now gives zero expected update,
  as on-policy. Nothing systematic remains to compound across steps.
* What remains is exactly the on-policy form (2) with the sampler's
  distribution: raise the tokens that are better than the sampler's average.
  Its inner product with the on-policy direction p⊙(A − Ā_p) is positive
  whenever q is close to p, so the learning signal is kept. It needs no ratio,
  so it has no ratio variance and nothing to clip.
* When q = p, Σ_v sg(p_v) ∇ log p(v) = Σ_v p_v(e_v − p) = 0, so the loss is
  exactly REINFORCE: the change only acts through the mismatch.
* Equivalent view: the term added to the loss is +A·Σ_v q_v log p(v)
  = −A·(H(q) + KL(q‖p)). Its logit gradient is A(q − p), which cancels exactly
  the expected logit gradient −A(q − p) that the plain loss −A log p(y) has
  under y ~ q (the drift of (3)). Unlike a KL penalty with a tuned
  coefficient, its coefficient is the advantage itself, so the cancellation is
  exact and needs no tuning.

### With an IS correction (optional)

If a token-level weight w(p/q) is used, the drift-free centre is the
reweighted sampler distribution:

    c_v = q(v) w(v) / Σ_u q(u) w(u)   ⇒   E_q[w (e_y − c)] = q⊙w − c E_q[w] = 0.

(For w = p/q this gives c = p, i.e. naive IS is already centred; for clipped
or masked weights it removes the residual drift c·(m − |m|p) of §2.)
Sequence-level weights are constant within a sequence, so c = q. Hence the
centring is drift-free for every point of the `rl.is.*` grid, and the default
(`correction=None`) is plain sampler-centred REINFORCE.

### Computing c with what an inference engine returns

Only the sampler's top-K logprobs (K = 20: `sampler_topk_ids`,
`sampler_topk_logprobs`) are available. On the top-K set, c is exact. The
remaining sampler mass q_T = 1 − Σ_{k∈K} q_k is spread like the trainer's
tail, q(v) ≈ q_T p(v)/p_T for v ∉ K (p_T the trainer's tail mass), computed
with a masked log-softmax. Then

    Σ_v c_v log p(v) = [Σ_{k∈K} q_k w_k log p_k + q_T w_T E_{p|tail}[log p]] / Z,

with w_T = w(p_T/q_T) and Z the normaliser. The residual drift is confined to
the shape of the tail inside a mass q_T (tiny for LLM next-token
distributions with K = 20): the drift becomes
Ā_q · q_T (q_{·|tail} − p_{·|tail}) instead of Ā_q (q − p). If the batch has no
top-K entries the centre falls back to the sampled token (plain REINFORCE).

Implementation (`losses.py`): `sampler_center` computes the detached centre
term per position inside the existing rematerialised chunks (one extra masked
log-softmax over [B, chunk, V]); the score becomes `w * (log p(y) − centre)`.
The signature, the correction grid, normalisation and metrics are unchanged.

## Sanity check (dev/mini.py, CPU)

Default setting (noise 0.3, ±1 rewards, 600 steps): plain loss, final train
reward 0.58 (eval 0.66); centred loss 0.85 (eval 0.85). With 40 extra
vocabulary tokens (seed 1): plain loss collapses to 0.00, centred loss 0.85.

## Configuration

No configuration change is needed: the centring is active with the default
arguments (`correction=None`).
