`haiku_L4_losses.py` : le `losses.py` rendu par Haiku au premier run L4
(`rlenv/runs/20261118-102641-exp-L4/`). Correct sans correction IS, mais il
centre le score sous q même quand le token porte un poids w(p/q) : il sert de
cas de régression pour le scénario TIS du test théorique.

`haiku_L3_losses.py` : le `losses.py` du premier run L3
(`rlenv/runs/20261118-112015-exp-L3/`). Correct avec la distribution complète,
mais il tronque la somme au top-k sans modéliser la queue : cas de régression
pour `passed_topk`.
