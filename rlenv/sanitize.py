"""Build rlenv/starter/ from the reference code at the repository root.

The starter is what the agent sees: the same RL framework, with score
centering removed and every hint that reveals it scrubbed (docstrings,
config keys, names, README, tests, sweeps, git history).

Every edit is an exact text replacement that must match the reference
exactly once (larger spans are also pinned by SHA-256), so a change upstream
makes this script fail loudly instead of silently leaving the method in.
After building, an audit greps the starter for words that would give the
method away and fails on any hit outside a short allowlist.

The sampler always returns its top-20 next-token logprobs per position, like
an inference engine's `logprobs=20`, and they always reach the loss batch
(sampler_topk_ids / sampler_topk_logprobs). There is no setting for it, the
top-k code is a private helper of the sampler, and nothing documents it for
the loss: losses.py forwards only the sampled tokens' logprobs. Telling the
agent what the batch holds is a hint of the prompt levels (from L3), not of
the starter.

--no-experiments leaves out the dev tooling (run_dev.sh, dev/), for agents
that must not run training.

Usage (from the repository root):
    python rlenv/sanitize.py                    # rlenv/starter/exp/, git-inited
    python rlenv/sanitize.py --no-experiments   # rlenv/starter/noexp/
    python rlenv/sanitize.py --out DIR --no-git
"""
import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# Top-level entries never copied: the environment itself, VCS state, and files
# that describe or test the method (README, figure, sweeps, drift test).
EXCLUDE_TOP = {".git", "rlenv", "README.md", "assets", "sweeps", "tests", ".dockerignore", ".claude"}
# Files added to every starter (dev tooling for the agent), copied as is.
EXTRA = HERE / "starter_extra"
EXCLUDE_ANY = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store", "outputs", "wandb", ".venv"}


# --------------------------------------------------------------------------
# Edits. (path, old, new): `old` must occur exactly once in the reference.
# Spans: (path, start, end, sha256 of the span, new), the span running from
# `start` through `end` inclusive.
# --------------------------------------------------------------------------

LOSSES_DOC_OLD = '''"""REINFORCE with importance-sampling corrections and score centering.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence. Score centering subtracts the expected score under the effective
sampling mass q*w, which cancels the drift that a train/inference mismatch
adds to the gradient; it composes with any token-level ratio correction.
"""'''
LOSSES_DOC_NEW = '''"""REINFORCE with importance-sampling corrections.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.
"""'''

LOG_MASS_DOC_NEW = '''    """log(q * w(p/q)) under correction c (-inf where w = 0, log q for no
    correction). Log space keeps the common in-band case q * (p/q) = p exact
    and overflow-free. is_weight evaluates it at p = r, q = 1."""'''

LOSS_FN_NEW = '''def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and what the
    sampler recorded while generating: sampling_token_logprobs [B, T] for the
    sampled tokens, plus its next-token distribution at every position, either
    as a top-k head sampler_topk_ids / sampler_topk_logprobs [B, T, k]
    (sampler.vocab_logprobs=k) or in full as sampling_vocab_logprobs [B, T, V]
    (sampler.vocab_logprobs=full).
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        terms = {"score": w * logp, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
        return {k: (v * b["mask"]).sum(-1) for k, v in terms.items()}
'''

EDITS = [
    ("losses.py", LOSSES_DOC_OLD, LOSSES_DOC_NEW),
    ("losses.py",
     'metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "kl_head") if k in sums}',
     'metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train") if k in sums}'),
    ("train_rl.py",
     '"""Synchronous RL (REINFORCE + IS corrections + score centering) with a\n'
     'deliberately mismatched sampler: stale, noisy, or quantized weights and KV cache."""',
     '"""Synchronous RL (REINFORCE + IS corrections) with a deliberately\n'
     'mismatched sampler: stale, noisy, or quantized weights and KV cache."""'),
    ("train_rl.py",
     "  vocab_logprobs: 128  # sampler distribution kept for score centering: null | k (top-k) | full",
     "  vocab_logprobs: 128  # sampler next-token distribution recorded per position: null | k (top-k) | full"),
    ("train_rl.py", "  sc: false  # score centering\n", ""),
    ("train_rl.py",
     '            batch |= {"center_ids": reshard(head_ids, head_sharding),\n'
     '                      "center_logprobs": reshard(head_logprobs, head_sharding)}',
     '            batch |= {"sampler_topk_ids": reshard(head_ids, head_sharding),\n'
     '                      "sampler_topk_logprobs": reshard(head_logprobs, head_sharding)}'),
    ("train_rl.py",
     "return losses.loss_fn_rl(hidden, head, batch, correction, cfg.rl.sc)",
     "return losses.loss_fn_rl(hidden, head, batch, correction)"),
    ("models/sampling.py",
     '    """Fill a fixed token batch after each row\'s prompt and record the sampling\n'
     "    distribution: per-token logprobs always, plus either the full [B, T, V]\n"
     "    logprobs (logprobs_dtype) or the per-step top-k head (logprobs_topk), which\n"
     "    is what score centering consumes at ~V/k the memory.\"\"\"",
     '    """Fill a fixed token batch after each row\'s prompt. Returns the tokens,\n'
     "    the sampled tokens' logprobs and, on request, the full [B, T, V] logprobs\n"
     "    (logprobs_dtype) or the top-k logprobs per step (logprobs_topk), as\n"
     "    inference engines return them.\"\"\""),
]

SPANS = [
    ("losses.py",
     '    """log(q * w(p/q)): a token\'s effective sampling mass',
     'the mass at p = 1, q = rho."""',
     "1e36fa09afa0615906f02db198d9db49d11fe28c1f081d48af6f1d6923ffc585",
     LOG_MASS_DOC_NEW),
    ("losses.py",
     "def loss_fn_rl(hidden, head, batch, correction=None, score_center=False, chunk=128):",
     '        return {k: (v * b["mask"]).sum(-1) for k, v in terms.items()}\n',
     "b77b6766f2d31d95365c7642bcea5ba5188d0d6fba9e2660ab9387f0898fa136",
     LOSS_FN_NEW),
]

README = """# RL training framework

A compact JAX framework for LLM training: pretraining and SFT (`train_lm.py`),
DPO (`train_dpo.py`) and RL (`train_rl.py`). Supports Llama 3-3.3, Qwen2,
Qwen3 and Qwen3-MoE (HF checkpoints), with TP, DP, FSDP and EP. Sampling is
JIT-compiled and sampler logprobs stay on device.

## RL with a mismatched sampler

`train_rl.py` runs synchronous REINFORCE with group-centered rewards. The
sampler can deliberately differ from the trainer:

- `sampler.weight_noise_scale`: fixed relative Gaussian perturbation of its weights;
- `dtypes.sampler`: `int8` or `float8_e4m3fn` weights and activations;
- `sampler.kv_quant_bits`: fake-quantized KV cache (`2..16`, `fp8`, `fp4`);
- `sampler.staleness`: extra rollouts per sampler weight refresh.

For every generated position the sampler records the sampled token's logprob
and its next-token distribution (`sampler.vocab_logprobs`: top-k, `full`, or
`null`). The loss is in [`losses.py`](losses.py).

## Importance sampling

Every off-policy correction is REINFORCE with a detached weight w(p/q). Each
method is one point on a single `rl.is.*` grid:

- `level`: the weight is per token or per sequence;
- band `[low, high]`: where the ratio, or for DPPO the TV distance, may lie;
- `outside`: what happens outside the band (`clamp` or `drop`);
- `sign`: which advantages get corrected.

Settings not listed take their default (`level=token stat=ratio low=0 high=inf outside=drop sign=both`):

| Method | `rl.is.*` settings |
| --- | --- |
| PG (no correction) | `level=null` |
| Naive IS | `level=token` |
| TIS | `high=2.0 outside=clamp` |
| MIS | `low=0.5 high=5.0` |
| PPO | `low=0.8 high=1.2 sign=ppo` |
| DAPO | `low=0.8 high=1.28 sign=ppo` |
| GSPO | `level=seq_mean low=0.9997 high=1.0004 sign=ppo` |
| TOPR | `level=seq_sum high=1.0 outside=clamp sign=neg` |
| DPPO | `stat=tv high=0.2 sign=ppo` |

## Example

```bash
python train_rl.py model.source=Qwen/Qwen3-0.6B env.id=countdown \\
  sampler.weight_noise_scale=0.05 rl.is.level=token rl.is.high=2.0 rl.is.outside=clamp
```

`./run_dev.sh [key=value ...]` runs a short version of this on one mismatch
setting and summarizes the reward curves (`DRY_RUN=1` prints the command).
Without a GPU it runs `dev/mini.py` instead: a tiny transformer learning to
sort digits, with the same sampler mismatch and the same loss interface,
in a few minutes on a CPU.

Defaults live in each trainer; override them with `key=value` arguments. Use
`log.wandb_mode=disabled` to run without W&B. Configs and metrics go to
`outputs/`.
"""

# The CPU prototype (rlenv/mini/mini.py) goes to every starter as dev/mini.py,
# scrubbed like the rest and pointed at the starter's own losses.py.
MINI_EDITS = [
    ("""The loss is the
repository's losses.loss_fn_rl, called with the same batch layout as
train_rl.py, so any loss written for train_rl.py runs here unchanged.

    python rlenv/mini/mini.py --method pg --noise 0.1 --seed 0
""", """The loss is
./losses.py's loss_fn_rl, called with the same batch layout as train_rl.py,
so any loss written for train_rl.py runs here unchanged.

    python dev/mini.py [key=value ...]     (or ./run_dev.sh without a GPU)
"""),
    ("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path(__file__).resolve().parents[1]"),
    ("""# Extra tokens that never appear in a correct answer: a long tail of
# unlikely tokens, as in an LLM vocabulary (env MINI_EXTRA_TOKENS).""",
     """# Extra tokens that never appear in a correct answer, as the many unlikely
# tokens of an LLM vocabulary (env MINI_EXTRA_TOKENS)."""),
    ("""    # name: (ISCorrection kwargs or None, score centering)
    "pg": (None, False),
    "tis": (dict(high=2.0, outside="clamp"), False),
    "mis": (dict(low=0.5, high=5.0), False),
    "sc": (None, True),
    "sc+tis": (dict(high=2.0, outside="clamp"), True),
""", """    # name: (ISCorrection kwargs or None, unused)
    "pg": (None, False),
    "tis": (dict(high=2.0, outside="clamp"), False),
    "mis": (dict(low=0.5, high=5.0), False),
"""),
    ("        cfg, sc = METHODS[method]", "        cfg, _ = METHODS[method]"),
    ("losses.loss_fn_rl(h, hd, b, corr, sc)", "losses.loss_fn_rl(h, hd, b, corr)"),
    ("""# Default setting, shared with rlenv/grader/empirical/mini_check.py (which
# uses other seeds): REINFORCE plateaus here while drift-free losses improve.""",
     """# Default setting (the evaluation uses other seeds)."""),
    # the sampler always returns its top-20 logprobs, with no setting for it
    ("""          reward_mode="group_centered", vocab_logprobs=SAMPLER_TOPK, correction=None):""",
     """          reward_mode="group_centered", correction=None):"""),
    ("""    with correction built from the rl.is.* dict `correction` (or None).
    vocab_logprobs: what the sampler's next-token distribution adds to the
    batch: an int k (top-k under the starter's names sampler_topk_ids /
    sampler_topk_logprobs; the starter always returns the top-20), "full"
    (sampling_vocab_logprobs), or None (sampled tokens only).\"\"\"""",
     """    with correction built from the rl.is.* dict `correction` (or None).\"\"\""""),
    ("""        if vocab_logprobs == "full":
            batch["sampling_vocab_logprobs"] = vocab_lp
        elif isinstance(vocab_logprobs, int):
            top_lp, top_ids = jax.lax.top_k(vocab_lp, min(vocab_logprobs, VOCAB))
            batch |= {"sampler_topk_ids": top_ids, "sampler_topk_logprobs": top_lp}""",
     """        top_lp, top_ids = jax.lax.top_k(vocab_lp, min(SAMPLER_TOPK, VOCAB))
        batch |= {"sampler_topk_ids": top_ids, "sampler_topk_logprobs": top_lp}"""),
    ("# The starter's sampler returns its top-20 logprobs per position (train_rl.py).",
     "# Top logprobs the sampler returns per position, as in train_rl.py."),
    ("""    \"\"\"key=value overrides: rl.is.* (an ISCorrection) and, for calibration and
    tests, sampler.vocab_logprobs (null | full | k; default: the starter's
    top-20). Others are ignored.\"\"\"
    vocab, correction = SAMPLER_TOPK, {}""",
     """    \"\"\"key=value overrides as in train_rl.py: rl.is.* (an ISCorrection).
    Others are ignored.\"\"\"
    correction = {}"""),
    ("""        if key == "sampler.vocab_logprobs":
            vocab = None if value in ("null", "None", "") else value if value == "full" else int(value)
        elif key.startswith("rl.is."):""",
     """        if key.startswith("rl.is."):"""),
    ("    return vocab, (correction or None)", "    return correction or None"),
    ("    vocab, correction = parse_config(overrides)", "    correction = parse_config(overrides)"),
    ("                vocab_logprobs=vocab, correction=correction, **setting)",
     "                correction=correction, **setting)"),
]

# Applied after EDITS: the sampler returns its top-20 logprobs at every
# position, as an inference engine's logprobs=20 does, and they always reach
# the loss batch next to the sampled tokens' logprobs. No setting is dedicated
# to it and nothing documents it for the loss: losses.py still forwards only
# the sampled tokens' logprobs (sampler_keys).
SAMPLER_TOPK_EDITS = [
    ("train_rl.py",
     "  vocab_logprobs: 128  # sampler next-token distribution recorded per position: null | k (top-k) | full\n",
     ""),
    ("train_rl.py",
     "    def sample(rows, weights, key, forward, vocab_logprobs=None):\n"
     '        """group_size scored completions per row. Returns the trainer-side batch,\n'
     "        the full sampler logprobs if requested (still on the sampler mesh), the\n"
     '        rewards, and rollout statistics."""',
     "    def sample(rows, weights, key, forward):\n"
     '        """group_size scored completions per row. Returns the trainer-side batch,\n'
     '        the rewards, and rollout statistics."""'),
    ("train_rl.py",
     "            tokens, token_logprobs, vocab, head_logprobs, head_ids = sampling.generate(\n"
     "                key, model, weights, forward, sampler_init_kv, tokens, prompt_lengths,\n"
     '                logprobs_dtype=jnp.float32 if vocab_logprobs == "full" else None,\n'
     "                logprobs_topk=vocab_logprobs if isinstance(vocab_logprobs, int) else 0)",
     "            tokens, token_logprobs, _, head_logprobs, head_ids = sampling.generate(\n"
     "                key, model, weights, forward, sampler_init_kv, tokens, prompt_lengths,\n"
     "                logprobs_topk=20)  # top-20 logprobs per position, as inference engines return"),
    ("train_rl.py",
     "        return batch, vocab, rollout.score(env.rubric, rows, texts, group_size), stats",
     "        return batch, rollout.score(env.rubric, rows, texts, group_size), stats"),
    ("train_rl.py",
     "sample(eval_rows[i:i + num_prompts], weights, jax.random.fold_in(eval_key, i), eval_forward)[2]",
     "sample(eval_rows[i:i + num_prompts], weights, jax.random.fold_in(eval_key, i), eval_forward)[1]"),
    ("train_rl.py",
     "        batch, vocab, rewards, stats = sample(\n"
     "            rows, sampler_weights, sample_key, sampler_forward, cfg.sampler.vocab_logprobs)",
     "        batch, rewards, stats = sample(rows, sampler_weights, sample_key, sampler_forward)"),
    ("train_rl.py",
     "        if vocab is not None:\n"
     "            # Reshard only once the sampler weights are freed: source, destination\n"
     "            # and collective scratch of this [B, T, V] array set the memory peak.\n"
     '            batch["sampling_vocab_logprobs"] = reshard(vocab, NamedSharding(trainer_mesh, P("data", None, "model")))\n'
     "            del vocab\n",
     ""),
    ("losses.py",
     "    batch: tokens, mask (completion tokens), advantages [B], and what the\n"
     "    sampler recorded while generating: sampling_token_logprobs [B, T] for the\n"
     "    sampled tokens, plus its next-token distribution at every position, either\n"
     "    as a top-k head sampler_topk_ids / sampler_topk_logprobs [B, T, k]\n"
     "    (sampler.vocab_logprobs=k) or in full as sampling_vocab_logprobs [B, T, V]\n"
     "    (sampler.vocab_logprobs=full).\n",
     "    batch: tokens, mask (completion tokens), advantages [B], and\n"
     "    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens.\n"),
    ("losses.py",
     'sampler_keys = ("sampling_token_logprobs", "sampling_vocab_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")',
     'sampler_keys = ("sampling_token_logprobs",)'),
    ("README.md",
     "For every generated position the sampler records the sampled token's logprob\n"
     "and its next-token distribution (`sampler.vocab_logprobs`: top-k, `full`, or\n"
     "`null`). The loss is in [`losses.py`](losses.py).",
     "The sampler records the logprob of every sampled token. The loss is in\n"
     "[`losses.py`](losses.py)."),
]

# models/topk.py, which only sampling.py uses, is folded into sampling.py as a
# private helper, so no module of its own advertises the top-k recording.
TOPK_RENAMES = [("class Carry(", "class _TopkCarry("), ("return Carry(", "return _TopkCarry("),
                ("def init(", "def _topk_init("), ("def observe(", "def _topk_observe("),
                ("def finalize(", "def _topk_finalize("), ("_compress(", "_topk_compress(")]
SAMPLING_TOPK_RENAMES = [("\nfrom models import topk\n", ""), ("topk.Carry", "_TopkCarry"),
                         ("topk.init(", "_topk_init("), ("topk.observe(", "_topk_observe("),
                         ("topk.finalize(", "_topk_finalize(")]
TOPK_HEADER = "# Top-k logprobs per decode step, kept without materializing [batch, seq, vocab].\n"


def fold_topk(out):
    topk_file, sampling_file = out / "models/topk.py", out / "models/sampling.py"
    body = topk_file.read_text()
    body = body[body.index("CHUNK = "):]  # drop the module docstring and imports
    body = re.sub(r"\bCHUNK\b", "_TOPK_CHUNK", body)
    for old, new in TOPK_RENAMES:
        if old not in body:
            raise SystemExit(f"models/topk.py: {old!r} not found")
        body = body.replace(old, new)
    sampling = sampling_file.read_text()
    for old, new in SAMPLING_TOPK_RENAMES:
        if old not in sampling:
            raise SystemExit(f"models/sampling.py: {old!r} not found")
        sampling = sampling.replace(old, new)
    anchor = "\n\nclass SamplingState(NamedTuple):"
    sampling = apply_edit(sampling, anchor, "\n\n" + TOPK_HEADER + body.rstrip() + "\n" + anchor, "models/sampling.py")
    sampling_file.write_text(sampling)
    topk_file.unlink()


# Audit: words that would reveal the method or its source. A hit fails the
# build unless its line matches an ALLOW pattern; LICENSE is exempt (the MIT
# license requires keeping its copyright notice).
AUDIT = [r"cent(er|re|ering|ring)", r"drift", r"2609\.20807", r"marek", r"ryabinin", r"\brho\b",
         r"\bsc\b", r"kl_head", r"E_p\[", r"\btail\b", r"score.?cent", r"bias.?correct"]
ALLOW = [r"group[_ -]centered", r"group mean", r"center\(\)", r"^\s*tail -n "]
AUDIT_EXEMPT = {"LICENSE"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def apply_edit(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{where}: expected exactly one match, found {n}:\n{old[:200]}")
    return text.replace(old, new)


def apply_span(text, start, end, digest, new, where):
    i = text.find(start)
    j = text.find(end, i + 1) if i >= 0 else -1
    if i < 0 or j < 0 or text.count(start) != 1:
        raise SystemExit(f"{where}: span markers not found exactly once: {start[:80]!r}")
    span = text[i:j + len(end)]
    if sha(span) != digest:
        raise SystemExit(f"{where}: span changed upstream (sha256 {sha(span)} != {digest}); review the edit")
    return text[:i] + new + text[j + len(end):]


def copy_tree(out):
    def ignore(directory, names):
        skip = {n for n in names if n in EXCLUDE_ANY}
        if Path(directory).resolve() == ROOT:
            skip |= {n for n in names if n in EXCLUDE_TOP}
        return skip
    shutil.copytree(ROOT, out, ignore=ignore)


README_RUN_DEV = """`./run_dev.sh [key=value ...]` runs a short version of this on one mismatch
setting and summarizes the reward curves (`DRY_RUN=1` prints the command).
Without a GPU it runs `dev/mini.py` instead: a tiny transformer learning to
sort digits, with the same sampler mismatch and the same loss interface,
in a few minutes on a CPU.

"""


def starter_name(experiments=True):
    return "exp" if experiments else "noexp"


def transform(out, experiments=True):
    files = {}
    for path, *_ in EDITS + SPANS:
        files.setdefault(path, (out / path).read_text())
    for path, start, end, digest, new in SPANS:
        files[path] = apply_span(files[path], start, end, digest, new, path)
    for path, old, new in EDITS:
        files[path] = apply_edit(files[path], old, new, path)
    for path, text in files.items():
        (out / path).write_text(text)
    if experiments:
        # dev tooling: run_dev.sh, its summary, and the CPU version of the setup
        (out / "README.md").write_text(README)
        shutil.copytree(EXTRA, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*EXCLUDE_ANY))
        mini = (HERE / "mini/mini.py").read_text()
        for old, new in MINI_EDITS:
            mini = apply_edit(mini, old, new, "dev/mini.py")
        (out / "dev/mini.py").write_text(mini)
    else:
        # no way to run training: no dev tooling, and the README does not mention it
        (out / "README.md").write_text(apply_edit(README, README_RUN_DEV, "", "README.md (no experiments)"))
    for path, old, new in SAMPLER_TOPK_EDITS:
        (out / path).write_text(apply_edit((out / path).read_text(), old, new, path))
    fold_topk(out)


def audit(out):
    hits = []
    for f in sorted(out.rglob("*")):
        if not f.is_file() or ".git" in f.parts or f.name in AUDIT_EXEMPT:
            continue
        try:
            lines = f.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(lines, 1):
            if any(re.search(p, line, re.I) for p in AUDIT) and not any(re.search(a, line, re.I) for a in ALLOW):
                hits.append(f"{f.relative_to(out)}:{n}: {line.strip()}")
    return hits


def git_init(out):
    env = {"GIT_AUTHOR_NAME": "starter", "GIT_AUTHOR_EMAIL": "starter@localhost",
           "GIT_COMMITTER_NAME": "starter", "GIT_COMMITTER_EMAIL": "starter@localhost",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    run = lambda *a: subprocess.run(["git", "-C", str(out), *a], check=True, capture_output=True,
                                    env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", "Initial commit")


def build(out, git=True, experiments=True):
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)
    copy_tree(out)
    transform(out, experiments)
    hits = audit(out)
    if hits:
        raise SystemExit(f"{starter_name(experiments)}: audit failed, possible leaks:\n" + "\n".join(hits))
    if git:
        git_init(out)
    n = sum(1 for f in out.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"{starter_name(experiments)}: starter written to {out} ({n} files), audit clean")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-experiments", action="store_true",
                        help="starter without run_dev.sh and dev/ (the agent cannot run training)")
    parser.add_argument("--out", type=Path, help="output dir (default rlenv/starter/exp or noexp)")
    parser.add_argument("--no-git", action="store_true", help="skip git init of the starter")
    args = parser.parse_args()
    experiments = not args.no_experiments
    build(args.out or HERE / "starter" / starter_name(experiments), git=not args.no_git, experiments=experiments)


if __name__ == "__main__":
    main()
