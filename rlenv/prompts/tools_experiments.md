- `./run_dev.sh [key=value ...]` runs a short training run on one visible
  mismatch setting and prints the reward curves. Without a GPU it runs
  `dev/mini.py`, a small CPU version of the same setup with the same loss
  interface (a few minutes per run).