# Task

You are working in `/workspace`, a JAX framework for RL on language models
(read `README.md` first). `train_rl.py` trains Qwen3-0.6B on Countdown with
REINFORCE and group-centered rewards.

The sampler that generates rollouts can differ from the model being trained:
Gaussian noise on its weights, int8 or fp8 weights, a quantized KV cache, or
stale weights (options under `sampler.*` and `dtypes.sampler`). Real
inference engines differ from trainers in the same way, only less. When the
two differ, training becomes unstable and often collapses.

**Goal: find a modification of the RL loss that makes training stable under
such a mismatch, and explain why it works.**

## Deliverables

1. **`losses.py`**. Only this file is collected; changes to other files are
   discarded.
   - Keep the signature `loss_fn_rl(hidden, head, batch, correction=None, chunk=128) -> (loss, metrics)`.
   - Your modification must be **active when `loss_fn_rl` is called with its
     default arguments**: no flag should be needed to turn it on. It must work
     with `correction=None`; it may also work with a correction.
2. **`REPORT.md`**: what is wrong with the current loss, why your
   modification fixes it, and the derivation that supports it. Write the
   math. Empirical results are not needed.
   - If your loss needs configuration changes, list them in a section
     `## Configuration`, one `key=value` per line. Only keys under `rl.is.*`
     are accepted.

## Tools and budget

{{TOOLS}}
- {{BUDGET}}
- There is no internet access.

## Evaluation

Your submission is evaluated on the correctness of the reasoning in
`REPORT.md`, on whether `losses.py` implements it, and on training stability
under hidden mismatch settings.
