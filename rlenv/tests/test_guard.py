"""The CPU guard-rail fails the uncorrected starter and a zero loss, and
passes score centering and TIS. Slow (~7 min): set RUN_SLOW=1."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1")
spec = importlib.util.spec_from_file_location("mini_check", ROOT / "rlenv/grader/empirical/mini_check.py")
mini_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini_check)


@pytest.fixture(scope="module")
def subs(tmp_path_factory):
    d = tmp_path_factory.mktemp("subs")
    subprocess.run([sys.executable, str(ROOT / "rlenv/sanitize.py"), "--variant", "topk-hint",
                    "--out", str(d / "starter"), "--no-git"], check=True)
    sc = (ROOT / "losses.py").read_text().replace("score_center=False", "score_center=True")
    sc = sc.replace('"center_ids"', '"sampler_topk_ids"').replace('"center_logprobs"', '"sampler_topk_logprobs"')
    (d / "sc.py").write_text(sc)
    (d / "zero.py").write_text("def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):\n"
                               "    return 0.0 * head(hidden).sum(), {}\n")
    return d


@slow
@pytest.mark.parametrize("name, path, config, passed", [
    ("starter", "starter/losses.py", (), False),
    ("starter+tis", "starter/losses.py", ("rl.is.high=2.0", "rl.is.outside=clamp"), True),
    ("sc-full", "sc.py", ("sampler.vocab_logprobs=full",), True),
    ("zero", "zero.py", (), False),
])
def test_guard(subs, name, path, config, passed):
    assert mini_check.evaluate(subs / path, config)["passed"] == passed


def test_parse_config():
    assert mini_check.parse_config(["sampler.vocab_logprobs=8", "rl.is.high=2.0"]) == (8, {"high": 2.0, "level": "token"})
    assert mini_check.parse_config(["rl.is.level=null", "model.source=x"]) == (None, None)
