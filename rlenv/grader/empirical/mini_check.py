"""CPU guard-rail: does a submitted loss train where plain REINFORCE fails?

Runs rlenv/mini (a tiny transformer sorting digits, sampler with fixed
relative weight noise) with the submitted losses.py on held-out seeds, and
normalizes the final evaluation reward between REINFORCE (0) and score
centering (1), both from the reference losses.py and cached in
baselines.json. This signal separates "removes the drift" from "does not";
it does NOT separate score centering from TIS/MIS (see rlenv/mini/RESULTS.md),
which the theory check and the judge do.

    python mini_check.py path/to/losses.py [key=value ...]   # JSON, exit 0 if passed
    python mini_check.py --calibrate                          # rewrite baselines.json

Accepted key=value: sampler.vocab_logprobs (null | full | k) and rl.is.*
(the agent's `## Configuration`); anything else is ignored.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINES = HERE / "baselines.json"
sys.path.insert(0, str(ROOT / "rlenv/mini"))
from mini import SETTING  # noqa: E402
SEEDS = (1, 2, 3)          # the agent's dev runs use seed 0
PASS = 0.5                 # normalized score needed to pass


def parse_config(args):
    sys.path.insert(0, str(ROOT / "rlenv/mini"))
    import mini
    return mini.parse_config(args)


def _run(job):
    sys.path.insert(0, str(ROOT / "rlenv/mini"))
    import mini
    method, seed, losses_path, vocab, correction = job
    log = mini.train(method, seed=seed, losses_path=Path(losses_path), log_every=SETTING["steps"],
                     vocab_logprobs=vocab, correction=correction, **SETTING)
    return log[-1]["eval_reward"]


def run_many(jobs):
    with mp.get_context("spawn").Pool(min(len(jobs), mp.cpu_count())) as pool:
        return pool.map(_run, jobs)


def calibrate():
    ref = str(ROOT / "losses.py")
    out = {}
    for method in ("pg", "sc"):
        out[method] = run_many([(method, s, ref, "full", None) for s in SEEDS])
    out["setting"], out["seeds"] = SETTING, list(SEEDS)
    BASELINES.write_text(json.dumps(out, indent=2) + "\n")
    return out


def evaluate(losses_path, config=()):
    base = json.loads(BASELINES.read_text())
    assert base["setting"] == SETTING and base["seeds"] == list(SEEDS), "baselines.json is stale: run --calibrate"
    vocab, correction = parse_config(config)
    try:
        finals = run_many([("submission", s, str(losses_path), vocab, correction) for s in SEEDS])
    except Exception as e:  # a loss that crashes or does not trace fails
        return {"passed": False, "error": f"{type(e).__name__}: {e}"[:500]}
    mean = lambda xs: sum(xs) / len(xs)
    pg, sc = mean(base["pg"]), mean(base["sc"])
    score = (mean(finals) - pg) / (sc - pg)
    return {"passed": score >= PASS, "normalized": score, "final_eval_reward": finals,
            "reinforce": pg, "score_centering": sc}


if __name__ == "__main__":
    if sys.argv[1:] == ["--calibrate"]:
        print(json.dumps(calibrate(), indent=2))
    else:
        result = evaluate(sys.argv[1], sys.argv[2:])
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)
