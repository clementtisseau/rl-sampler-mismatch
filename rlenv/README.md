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
| `starter/` | **Généré, non versionné.** Le dépôt que l'agent reçoit dans son conteneur : `exp/` (avec outils d'entraînement) ou `noexp/` (sans). |
| `starter_extra/` | Ajouté à chaque starter : `run_dev.sh` (run court sur le réglage visible, bruit 0,05 sur les poids, 150 pas) et `dev/summarize.py` (tableau, courbe texte, verdict d'effondrement). |
| `prompts/` | Les prompts : `common.md` (tâche, livrables, outils, évaluation) + niveaux cumulatifs, assemblés par `compose.py`. `L0` : rien de plus ; `L1` : ce qu'on savait avant l'article ; `L2a`, `L2b` : une direction d'analyse ; `L3` : la décomposition et où trouver la distribution de l'échantillonneur ; `L4` : la méthode. Un fichier `tools_*.md` par mode d'expérimentation. |
| `grader/` | Le correcteur : test théorique, juge LLM, évaluation empirique. |
| `docker/` | Images `agent` (starter + GPU, sans internet) et `grader`. |
| `task.py` | Tâche Inspect AI : un échantillon par niveau, agent `react` (bash, python, éditeur) dans le bac à sable Docker, trois scorers (`theory`, `guard`, `judge`). |

## Lancer une évaluation (CPU)

Il faut Docker et `pip install -r rlenv/requirements-host.txt`. La tâche
génère les starters et construit les images au premier lancement.

```bash
inspect eval rlenv/task.py --model <modèle agent> \
    -T levels=L3,L4 -T experiments=true -T hours=2 -T tokens=5000000 -T epochs=3
```

- `levels` : parmi `L0,L1,L2a,L2b,L3,L4` (défaut `L2a,L2b,L3,L4`) ;
  `experiments` : `true` (défaut) ou `false`, voir « Expériences prévues » ;
  `epochs` : tentatives par niveau (pass@k).
- Un run s'arrête quand l'agent appelle `submit()` ou atteint une limite :
  `hours` (temps de travail, défaut 2 ; l'attente de l'API n'est pas comptée,
  le temps réel est plafonné à 3 fois), `tokens` (total du run, défaut
  5 000 000 : la borne de coût), `messages` (défaut 500, contre les boucles).
  `compaction` (défaut `true`) compresse l'historique près de la limite de
  contexte (compaction native du fournisseur, sinon résumé).
- `judges` : un ou plusieurs modèles séparés par des virgules (rien = pas de juge).
- Le code de l'agent ne tourne que dans des conteneurs sans réseau : son bac à
  sable, et l'image du correcteur (`losses.py` monté en lecture seule).
- Réseau qui intercepte TLS : `SC_RLENV_PIPCA=<bundle CA>` pour construire
  les images.

## Expériences prévues

- **Avec ou sans expérimentation** (`-T experiments=true|false`). Comparer, à
  niveau d'indice égal, un agent qui peut lancer des entraînements
  (`run_dev.sh`, `dev/mini.py`) et un agent qui ne le peut pas (starter
  `noexp` sans ces outils, prompt qui le dit). Question : voir le
  problème en pratique (par exemple la KL entre les deux modèles qui explose
  sous REINFORCE) aide-t-il à trouver la solution attendue ? Mesure : taux de
  `rediscovered` / `discovered` du juge et réussite du test théorique, en
  pass@k par niveau. Le reste du prompt est identique dans les deux cas.

## Générer le starter

```bash
python rlenv/sanitize.py                    # écrit rlenv/starter/exp/ (git init neuf)
python rlenv/sanitize.py --no-experiments   # écrit rlenv/starter/noexp/
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

- **Distribution de l'échantillonneur dans le starter** : l'échantillonneur
  rend toujours ses 20 meilleurs logprobs par position, comme le `logprobs=20`
  d'un moteur d'inférence, et ils arrivent toujours dans le batch de la loss
  (`sampler_topk_ids` / `sampler_topk_logprobs`). Aucun réglage ne leur est
  dédié, le code du top-k est un détail privé de l'échantillonneur (fusionné
  dans `models/sampling.py`) et rien ne les documente pour la loss :
  `losses.py` ne transmet que le logprob des tokens tirés. Seules les clés
  `rl.is.*` sont acceptées en configuration. Le prompt dit ce que contient le
  batch à partir de **L3**, avec la formule de s̄ qui en a besoin.
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
