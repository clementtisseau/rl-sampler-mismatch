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

## Générer le starter

```bash
python rlenv/sanitize.py          # écrit rlenv/starter/ (git init neuf)
python -m pytest rlenv/tests      # starter == référence sans score centering
```

`sanitize.py` copie la racine sans `README.md`, `assets/`, `sweeps/` ni
`tests/`, retire le score centering de `losses.py` et `train_rl.py`, réécrit
les docstrings qui le mentionnent, renomme `center_ids` / `center_logprobs`
en `sampler_topk_ids` / `sampler_topk_logprobs`, écrit un README neutre, puis
audite le résultat (grep de mots révélateurs). Chaque modification doit
correspondre exactement au code de référence (empreintes SHA-256) : si le
code original change, le script échoue au lieu de laisser passer la méthode.

### Décisions ouvertes

- **Top-k de l'échantillonneur conservé** : la distribution top-k est
  toujours enregistrée (`sampler.vocab_logprobs: 128`) et présente dans le
  batch, documentée comme une donnée standard. Sans elle, la méthode est
  infaisable ; avec, c'est un léger indice.
- **`log_mass`** reste dans `losses.py` (utilisée par `is_weight`) avec une
  docstring neutre. Sa forme générale `log(q * w(p/q))` est un indice faible.
- **`LICENSE`** garde le nom de l'auteur (obligation de la licence MIT).
- **Contamination** : le README original indique que le score centering est
  déjà intégré à verl, TRL, slime, prime-rl, etc. Tout modèle dont les données
  vont au-delà de septembre 2026 est probablement contaminé.

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
