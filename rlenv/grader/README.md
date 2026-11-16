# Correcteur

Tourne dans un conteneur que l'agent ne voit pas. Il copie le `losses.py` de
l'agent dans un arbre propre, puis calcule trois signaux :

- `theory/check.py` : test théorique, exact et sur CPU (~10 s). Politique
  jouet (vocabulaire 5, longueur 3, logits indexés par préfixe), toutes les
  complétions énumérées. Trois contrôles, dans deux scénarios de données
  (distribution complète de l'échantillonneur, ou top-k avec une queue
  proportionnelle à p) ; il faut tout réussir dans au moins un scénario :
  - `no_drift` : avec un avantage constant, la mise à jour attendue sous q est nulle ;
  - `on_policy` : si q = p, la mise à jour attendue est celle de REINFORCE ;
  - `bounded` : sous fort décalage, aucun gradient individuel ne dépasse
    10 fois le pire de REINFORCE (exclut l'IS non tronqué).
  `alignment` (cosinus avec le vrai gradient on-policy) est rapporté sans seuil.
  Usage : `python rlenv/grader/theory/check.py chemin/losses.py` (JSON, code
  de sortie 0 si réussi). Validé par `rlenv/tests/test_theory.py`.
- `judge/` : grille cachée et prompt du juge LLM, explication de référence.
- `empirical/` : sweeps cachés (dérivés de `sweeps/06b_*.yaml`) et résultats
  des baselines mis en cache.
