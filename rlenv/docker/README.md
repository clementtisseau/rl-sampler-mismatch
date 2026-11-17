# Images

- `Dockerfile.agent-cpu` : bac à sable de l'agent sur CPU. Contient le starter
  (`ARG VARIANT=no-hint|topk-hint`) et JAX CPU ; `run_dev.sh` y lance
  `dev/mini.py`. Pas de réseau (`network_mode: none`), 4 CPU, 8 Go.
  `compose.<variante>.yaml` est le fichier de sandbox donné à Inspect.
- `Dockerfile.grader` : correcteur sur CPU (test théorique et garde-fou
  empirique). Il exécute le `losses.py` de l'agent, donc il tourne à part,
  sans réseau, avec le fichier monté en lecture seule. Le code de référence
  n'y est visible que du correcteur.
- *(à venir)* une image agent GPU pour la validation sur Qwen3-0.6B.

Construction manuelle (la tâche Inspect le fait toute seule) :

```bash
python rlenv/sanitize.py
docker build -t sc-rlenv-grader -f rlenv/docker/Dockerfile.grader .
```
