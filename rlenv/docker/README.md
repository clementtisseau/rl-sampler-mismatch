# Images

- `Dockerfile.agent-cpu` : bac à sable de l'agent sur CPU. Contient un starter
  (`ARG STARTER=no-hint|topk-hint|no-hint-noexp|topk-hint-noexp`) et JAX CPU ;
  avec expérimentation, `run_dev.sh` y lance `dev/mini.py`. La tâche écrit le
  fichier de sandbox Inspect dans `generated/compose.<starter>.yaml` : pas de
  réseau, 4 CPU, 8 Go.
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
