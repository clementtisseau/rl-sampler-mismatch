# Environnement de redécouverte : score centering

Cet environnement teste si un agent peut trouver la contribution de
*Score Centering Stabilizes Off-policy RL* (Marek & Ryabinin, arXiv:2609.20807)
à partir du problème connu (instabilité du RL sous training-inference mismatch)
et d'indices plus ou moins détaillés.

Le code original du dépôt (racine) reste intact : c'est la **référence**.
Tout ce qui concerne l'environnement vit dans `rlenv/`.

## Organisation

| Chemin | Rôle |
|---|---|
| `sanitize.py` | Génère `starter/` depuis la racine : retire le score centering et les indices qui le trahissent. Déterministe, rejouable. |
| `starter/` | **Généré, non versionné.** Le dépôt que l'agent reçoit dans son conteneur. |
| `prompts/L0..L4.md` | Les prompts, du problème seul (L0) à la méthode complète (L4). |
| `grader/` | Le correcteur : test théorique, juge LLM, évaluation empirique. |
| `docker/` | Images `agent` (starter + GPU, sans internet) et `grader`. |
| `task.py` | *(à venir)* Tâche Inspect AI, paramétrée par `hint_level`. |

## Flux

```
racine (référence) --sanitize.py--> starter/ --> conteneur agent
                                                    |
                                       losses.py + REPORT.md
                                                    v
                                     conteneur grader (3 signaux)
```

## Livrable de l'agent

- `losses.py` modifié (signature de `loss_fn_rl` figée hors options).
- `REPORT.md` : dérivation et justification de la modification.

## Score

1. **Test théorique** (CPU, secondes) : le gradient attendu de la loss sous
   l'échantillonneur ne contient plus de dérive, et la correction est nulle
   en on-policy.
2. **Juge LLM** sur `REPORT.md` + diff, avec l'article et la référence.
3. **Empirique** (GPU, évaluation finale seulement) : sweeps 0.6B Countdown
   sur réglages et seeds cachés, normalisé entre la meilleure baseline (0)
   et le score centering (1).

Récompense de RL = 1 + 2. Évaluation finale = 1 + 2 + 3.
