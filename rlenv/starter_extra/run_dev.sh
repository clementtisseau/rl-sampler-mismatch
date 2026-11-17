#!/usr/bin/env bash
# Short RL run on the visible mismatch setting, then a summary of the curves.
#
#   ./run_dev.sh [key=value ...]      extra overrides, e.g. rl.is.level=token
#
# Visible setting: Qwen3-0.6B on Countdown, sampler weights perturbed by
# relative Gaussian noise of scale 0.05. Evaluation uses other settings.
# Environment: DEV_STEPS (default 150), DEV_SEED (default 0),
# WEIGHTS_DIR (model cache), DRY_RUN=1 (print the command and exit),
# DEV_BACKEND (gpu | cpu | auto: cpu runs dev/mini.py when no GPU is found).
set -euo pipefail
cd "$(dirname "$0")"

steps=${DEV_STEPS:-150}
seed=${DEV_SEED:-0}
run=dev-$(date +%Y%m%d-%H%M%S)
cmd=(python train_rl.py
     model.source=Qwen/Qwen3-0.6B "model.weights_dir=${WEIGHTS_DIR:-$HOME/.cache/postax/weights}"
     env.id=countdown env.prompt_format=chat
     rl.num_prompts=64 rl.seq_len=512 eval.num_prompts=64 eval.every_steps=25
     opt.optimizer=sgd opt.lr=0.01 "stop.steps=$steps"
     sampler.weight_noise_scale=0.05 sampler.staleness=0
     "run.seed=$seed" log.wandb_mode=disabled log.project=dev "log.run_name=$run"
     "$@")

backend=${DEV_BACKEND:-auto}
if [[ $backend == auto ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then backend=gpu; else backend=cpu; fi
fi
if [[ $backend == cpu ]]; then
    # No GPU: same mismatch and loss interface on a tiny model (dev/mini.py).
    cmd=(python dev/mini.py "--seed=$seed" "--out=outputs/dev/$run/logs.jsonl" "$@")
    [[ -n "${DEV_STEPS:-}" ]] && cmd+=("--steps=$steps")
fi

echo "+ ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi

mkdir -p outputs/dev
out="outputs/dev/$run.stdout.log"
echo "training on $backend, full output in $out"
status=0
"${cmd[@]}" > "$out" 2>&1 || status=$?
if [[ $status -ne 0 ]]; then
    echo "training failed (exit $status); last lines of $out:"
    tail -n 30 "$out"
    exit "$status"
fi
python dev/summarize.py "outputs/dev/$run/logs.jsonl"
