# Images

- `base` : image JAX CUDA, `requirements.txt`, poids Qwen3-0.6B et dataset
  Countdown pré-téléchargés (l'agent n'a pas internet).
- `agent` : `base` + `starter/` + `run_dev.sh`. Pas de `grader/`.
- `grader` : `base` + code de référence + `rlenv/grader/`.
