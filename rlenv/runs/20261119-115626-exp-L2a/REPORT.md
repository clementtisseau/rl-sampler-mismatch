# Sampler-mismatch bias in REINFORCE, and a score-centering fix

## What is wrong with the current loss

With `correction=None`, `loss_fn_rl` is `L = -(1/N) sum_b A_b sum_t logp(y_bt | s_bt)`, where the tokens y are drawn from the sampler q, not from the trainer p. Write s_t for the prefix (prompt and earlier tokens), `A(y)` for the sequence advantage, and `Q_q(s,v) = E_q[A | s, y_t = v]`, `V_q(s) = E_q[A | s] = sum_v q(v|s) Q_q(s,v)`. The expected gradient is

  g_q = E_{y~q} [ A(y) sum_t grad log p(y_t | s_t) ]
      = sum_t E_{s_t~q} sum_v q(v|s_t) Q_q(s_t, v) grad log p(v|s_t).

**On-policy (q = p).** Since sum_v p(v|s) grad log p(v|s) = grad sum_v p(v|s) = 0 at every state, the same expression equals sum_t E_{s~p} sum_v p(v|s) (Q_p - V_p)(s,v) grad log p(v|s). This is the policy gradient grad J with a state-dependent baseline. Constant advantages contribute nothing.

**Mismatch (q != p).** Split Q_q = (Q_q - V_q) + V_q:

  g_q = sum_t E_{s~q} [ sum_v q(v|s)(Q_q - V_q)(s,v) grad log p(v|s)  +  V_q(s) * grad b(s) ],

with b(s) := sum_v q(v|s) log p(v|s) and grad b(s) = sum_v q(v|s) grad log p(v|s) (q is detached). The second term is reward-free. For a constant advantage c it is the whole gradient: g_q = c * sum_t E_q[grad b(s_t)], which is the gradient of an expected cross-entropy. It is zero only when q = p at the visited states. Maximizing it is maximum likelihood on the sampler's own tokens, so each step pulls the trainer toward the current sampler.

This term is not zero-mean noise. It is a systematic direction determined by q - p. The sampler is refreshed from the trainer at each step, so q_k = p_k + delta_k, and the pull toward q_k is applied at every step. It accumulates instead of averaging out. Group centering does not remove it: centering subtracts a per-prompt mean of A, while the bias depends on V_q(s) at each prefix, and that varies between states inside a sequence and across prompts. Clipping or truncating the trainer/sampler ratio only reshapes the same reward-free term, and stronger clipping changes it in a way that adds bias.

## The modification

Subtract the sampler-expected score from each token's score. The per-token term becomes

  score_t = logp(y_t | s_t) - sum_v sg[q(v|s_t)] logp(v|s_t),

and the loss is `-(1/N) sum_b A_b sum_t w_bt score_t`, with w the usual (detached) IS weight, which is 1 when `correction=None`. Its gradient is grad logp(y_t|s_t) - grad b(s_t). The baseline is a function of the state only and has zero expectation under q, so this is a control variate on the score that needs no ratio and no clipping.

**Expected update.** Since E_q[grad logp(y_t|s_t) | s_t] = grad b(s_t) for every q,

  E_q[ A(y) sum_t (grad logp(y_t|s_t) - grad b(s_t)) ]
    = sum_t E_{s~q} sum_v q(v|s) (Q_q(s,v) - V_q(s)) grad log p(v|s).

Two facts follow.

1. **Constant advantages give zero update for any q.** If Q_q = V_q = c, the right-hand side is 0. The reward-free drift term is gone exactly, which is what compounds across steps.

2. **On-policy the modification is inert.** If q = p, then grad b(s) = sum_v p grad log p = 0 identically, so the loss and its gradient are unchanged, and the expected update is the policy gradient with baseline V_p (as above).

Under mismatch, the residual error relative to grad J is of order |q - p| times the within-state advantage variation |Q - V|. The original loss's error is of order |q - p| times the full advantage level |V|, and that part is what drifts.

**Top-k support.** The sampler returns only the top-K log-probabilities (`sampler_topk_*`, K=20 in train_rl.py), so the implementation uses b_K(s) = sum_{v in K} sg[q(v|s)] log p(v|s). The error is the tail term -sum_{v not in K} q(v|s) grad log p(v|s). It is small when the tail mass of q is small. It is exact when K covers the vocabulary, as in dev/mini.py's default 12-token vocabulary. If `sampler_topk_*` is absent from the batch, the code falls back to the original score.

**Why this is not an IS correction.** IS multiplies by p/q, which is exact in expectation but has high variance. Clipping then biases it, which is why the baseline-corrected loss does not degrade the way clipped IS does. The fix changes the baseline, not the weights.

## Checks (CPU mini setting, `dev/mini.py`)

- Gradient of the loss at a mismatched sampler (noise 0.3, 4096 rollouts), with constant advantage A = 1: original |grad| = 1.39, new |grad| = 0.07. The residual is Monte Carlo noise. With A = 1 and q = p, the original and new gradients are identical (0.018 both).
- Noise 0.6, 2 seeds, default 12-token vocabulary: the original loss collapses to reward 0 in both seeds; the new loss ends at mean training reward 0.37 to 0.40.
- Same with 300 extra vocabulary tokens (so top-20 truncation is active): original collapses to 0; new ends at 0.24 to 0.27.
- Default mini setting (noise 0.3, 600 steps): original final eval reward 0.66; new 0.83.

## Configuration

No `rl.is.*` changes are needed. The modification is active under the default arguments of `loss_fn_rl`, and it works with `correction=None` and with an ISCorrection.
