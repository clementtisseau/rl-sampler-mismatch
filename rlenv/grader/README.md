# Correcteur

Tourne dans un conteneur que l'agent ne voit pas. Il copie le `losses.py` de
l'agent dans un arbre propre, puis calcule trois signaux :

- `theory/check.py` : test théorique, exact et sur CPU (~10 s). Politique
  jouet (vocabulaire 5, longueur 3, logits indexés par préfixe), toutes les
  complétions énumérées. Trois contrôles, dans deux scénarios de données
  (distribution complète de l'échantillonneur, ou top-k avec une queue
  proportionnelle à p) ; il faut tout réussir dans au moins un scénario :
  - `no_drift` : avec un avantage constant, la mise à jour attendue sous q est nulle ;
  - `no_drift_prefix` : avec un avantage de moyenne nulle sous q mais qui
    dépend des d premiers tokens, la mise à jour de chaque position après d
    est nulle. Rejette une baseline globale exacte (GRPO poussé à la limite),
    qui annule la dérive moyenne mais pas celle de chaque préfixe ;
  - `on_policy` : si q = p, la mise à jour attendue est celle de REINFORCE ;
  - `bounded` : sous fort décalage, aucun gradient individuel ne dépasse
    10 fois le pire de REINFORCE (exclut l'IS non tronqué).
  `alignment` (cosinus avec le vrai gradient on-policy) est rapporté sans seuil.
  **`passed_topk`**, rapporté à part : réussite du scénario top-k seul, le cas
  d'un vrai moteur d'inférence. Une loss qui tronque la somme aux k premiers
  tokens sans modéliser la queue y échoue, même en on-policy : c'est le cas
  du premier run Haiku en L3.
  **Scénario TIS**, rapporté à part (`passed_with_tis`, ne change pas
  `passed`) : les mêmes contrôles avec la loss appelée avec une correction
  TIS. Une loss qui centre le score sous q alors que le token porte un poids
  w(p/q) y échoue (il faut centrer sous q·w) : c'est le cas du premier run
  Haiku en L4 (dérive 0,20, autant que REINFORCE).
  Usage : `python rlenv/grader/theory/check.py chemin/losses.py` (JSON, code
  de sortie 0 si réussi). Validé par `rlenv/tests/test_theory.py`.
- `judge/` : juge LLM (aucun appel de modèle ici).
  - `reference.md` : explication de référence de la contribution.
  - `rubric.md` : grille cachée, 6 critères notés 0/1/2 et pondérés (la
    décomposition par préfixe et la correction du score pèsent 3 chacune),
    7 quasi-solutions (N1 : décomposition par séquence → baseline façon GRPO,
    N2 : espérance sous p, etc.), cas « autre contribution valide » → relecture
    humaine, règles anti-manipulation.
  - `prompt.md` + `schema.json` : prompt du juge et format JSON de sortie.
  - `judge.py` : remplit le prompt, valide un verdict, calcule le score pondéré
    et les drapeaux `rediscovered` / `discovered` (ce dernier exclut ce qui
    était déjà donné dans le prompt de l'agent), agrège plusieurs juges
    (médiane, majorité).
  Le juge ne voit pas le résultat du test théorique : les signaux restent
  indépendants.
- `empirical/mini_check.py` : garde-fou empirique sur CPU (~1 min 30 avec
  3 cœurs). Lance le prototype `rlenv/mini` (tri de chiffres, bruit relatif
  0,3 sur les poids du modèle de génération, avantages ±1, 600 pas) avec la
  loss soumise sur 3 seeds cachées, puis normalise la récompense finale entre
  REINFORCE (0) et le score centering (1), mis en cache dans
  `baselines.json` (`--calibrate` les recalcule). Réussi si ≥ 0,5. Applique
  la section `## Configuration` de l'agent (`sampler.vocab_logprobs`,
  `rl.is.*`). Ce signal distingue « corrige la dérive » de « ne corrige
  rien », **pas** le score centering de TIS/MIS : voir
  `rlenv/mini/RESULTS.md`.
  Validation (moyenne des 3 seeds → score normalisé) : starter sans
  correction 0,67 → 0,00 ❌ ; starter + TIS 0,79 → 0,79 ✅ ; score centering
  (distribution complète) 0,83 → 1,00 ✅ ; score centering top-4 (sur 12
  tokens) 0,77 → 0,66 ✅ ; loss nulle 0,52 → −0,89 ❌. Rejouer :
  `RUN_SLOW=1 python -m pytest rlenv/tests/test_guard.py` (~7 min).
- Validation sur Qwen3-0.6B (GPU, plus tard) : sweeps dérivés de
  `sweeps/06b_*.yaml`, réglages et seeds cachés.
