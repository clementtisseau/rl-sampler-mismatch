"""Summarize a training run from its logs.jsonl: a table of the main
metrics every few steps, a text plot of the training reward, and a verdict
on whether the reward collapsed."""
import json
import sys

COLUMNS = ("reward", "eval_reward", "kl_samp_train", "grad_norm", "pol_entropy")
BLOCKS = " ▁▂▃▄▅▆▇█"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sparkline(values, width=60):
    if not values:
        return ""
    step = max(1, len(values) // width)
    pts = [sum(values[i:i + step]) / len(values[i:i + step]) for i in range(0, len(values), step)]
    lo, hi = min(pts), max(pts)
    span = hi - lo or 1.0
    return "".join(BLOCKS[round((v - lo) / span * (len(BLOCKS) - 1))] for v in pts) + f"  [{lo:.3f} .. {hi:.3f}]"


def fmt(value):
    return f"{value:>12.4g}" if isinstance(value, (int, float)) else f"{'':>12}"


def summarize(rows, every=10):
    print(f"{'step':>6}" + "".join(f"{c:>12}" for c in COLUMNS))
    for row in rows:
        if row["step"] % every == 0 or "eval_reward" in row or row is rows[-1]:
            print(f"{row['step']:>6}" + "".join(fmt(row.get(c)) for c in COLUMNS))
    rewards = [r["reward"] for r in rows if "reward" in r]
    print("\nreward " + sparkline(rewards))
    evals = [(r["step"], r["eval_reward"]) for r in rows if "eval_reward" in r]
    peak = max(rewards, default=0.0)
    final = sum(rewards[-10:]) / max(len(rewards[-10:]), 1)
    print(f"\nsteps run: {len(rows)}   peak reward: {peak:.3f}   mean of last 10: {final:.3f}")
    if evals:
        print("eval reward: " + ", ".join(f"step {s}: {v:.3f}" for s, v in evals))
    collapsed = peak > 0 and final < 0.5 * peak
    print("verdict: COLLAPSED (last 10 below half the peak)" if collapsed else "verdict: no collapse")
    return collapsed


if __name__ == "__main__":
    rows = load(sys.argv[1])
    if not rows:
        sys.exit("no metrics logged")
    summarize(rows)
