"""run_dev.sh: builds the right command, later overrides win, and the
summary flags collapse."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ROOT / "rlenv/starter_extra"
spec = importlib.util.spec_from_file_location("summarize", EXTRA / "dev/summarize.py")
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def dry_run(*args, backend="gpu"):
    env = {**os.environ, "DRY_RUN": "1", "DEV_STEPS": "40", "DEV_BACKEND": backend}
    out = subprocess.run(["bash", str(EXTRA / "run_dev.sh"), *args], env=env, check=True,
                         capture_output=True, text=True).stdout
    return out.split("+ ", 1)[1].split()


def test_dry_run_command():
    cmd = dry_run("rl.is.level=token")
    assert cmd[:2] == ["python", "train_rl.py"]
    assert "sampler.weight_noise_scale=0.05" in cmd and "stop.steps=40" in cmd
    assert cmd[-1] == "rl.is.level=token"


def test_dry_run_cpu_command():
    cmd = dry_run("rl.is.level=token", backend="cpu")
    assert cmd[:2] == ["python", "dev/mini.py"]
    assert "rl.is.level=token" in cmd and "--steps=40" in cmd


def test_later_overrides_win():
    omegaconf = pytest.importorskip("omegaconf")
    cmd = dry_run("sampler.weight_noise_scale=0.0")
    base = omegaconf.OmegaConf.create({"sampler": {"weight_noise_scale": 1.0}})
    sys_argv, sys.argv = sys.argv, ["x", *cmd[2:]]
    try:
        cli = omegaconf.OmegaConf.from_cli()
    finally:
        sys.argv = sys_argv
    assert omegaconf.OmegaConf.merge(base, cli).sampler.weight_noise_scale == 0.0


def rows(rewards):
    return [{"step": i, "reward": r, **({"eval_reward": r} if i % 25 == 0 else {})} for i, r in enumerate(rewards)]


def test_summary_flags_collapse(capsys):
    assert summarize.summarize(rows([0.1 * i for i in range(10)] + [0.05] * 10))
    assert "COLLAPSED" in capsys.readouterr().out


def test_summary_stable(tmp_path, capsys):
    log = tmp_path / "logs.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows([0.02 * i for i in range(30)])))
    assert not summarize.summarize(summarize.load(log))
    out = capsys.readouterr().out
    assert "no collapse" in out and "eval reward: step 0" in out
