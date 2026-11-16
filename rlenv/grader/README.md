# Correcteur

Tourne dans un conteneur que l'agent ne voit pas. Il copie le `losses.py` de
l'agent dans un arbre propre, puis calcule trois signaux :

- `theory/` : test d'absence de dérive par énumération exacte (généralise
  `tests/test_loss.py` de la référence).
- `judge/` : grille cachée et prompt du juge LLM, explication de référence.
- `empirical/` : sweeps cachés (dérivés de `sweeps/06b_*.yaml`) et résultats
  des baselines mis en cache.
