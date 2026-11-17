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
- `empirical/` : sweeps cachés (dérivés de `sweeps/06b_*.yaml`) et résultats
  des baselines mis en cache.
