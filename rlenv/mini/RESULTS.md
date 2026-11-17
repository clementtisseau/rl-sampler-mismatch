# Prototype CPU : premiers résultats

Tâche : trier 5 chiffres (vocabulaire 12, 12 tokens). Petit transformer
(2 couches, d = 64), échauffement supervisé de 15 pas (récompense ≈ 0,56),
puis RL par SGD (lr 0,2), 16 prompts × 8 complétions par pas. Le modèle de
génération = poids du modèle entraîné + bruit gaussien relatif fixe (comme
`sampler.weight_noise_scale`). Loss : `losses.loss_fn_rl` de la référence.
Un run de 600 pas prend environ 1 à 2 min sur un cœur CPU.

Récompense d'évaluation finale (modèle propre, 512 prompts fixes).

## Avantages ±1 (`plus_minus`), bruit 0,3, 600 pas, 2 seeds

| Méthode | seed 0 | seed 1 |
|---|---|---|
| REINFORCE | 0,66 | 0,66 |
| TIS | 0,81 | 0,79 |
| MIS | 0,83 | 0,81 |
| SC | 0,83 | 0,83 |
| SC + TIS | 0,86 | 0,85 |

## Avantages centrés par groupe, 600 pas, seed 0

| Bruit | REINFORCE | TIS | SC |
|---|---|---|---|
| 0,5 | 0,57 (pic 0,65, KL → 9) | 0,70 | 0,70 |
| 0,7 | 0,23 (effondrement, KL → 23) | 0,64 | 0,49 |
| 1,0 | 0,22 | — | 0,37 |

## Avantages centrés par groupe, 3 000 pas, seed 0

| Bruit | REINFORCE | TIS | SC | SC + TIS |
|---|---|---|---|---|
| 0,2 | 0,93 | 0,98 | 0,96 | 0,97 |
| 0,3 | 0,90 | 0,94 | 0,94 | 0,94 |

## Lecture

- La dérive est bien reproduite : REINFORCE stagne ou s'effondre, avec une
  KL entre les deux modèles qui explose, alors que les corrections tiennent.
- Le score centering bat REINFORCE nettement, mais **pas encore TIS/MIS** :
  les écarts SC vs TIS (≤ 0,04) sont de l'ordre du bruit entre seeds, et
  sous fort décalage (KL de 1 à 2 nats par token) TIS fait mieux que SC.
- Le régime de l'article (petit décalage, grand vocabulaire avec une longue
  queue de tokens peu probables) n'est pas encore reproduit.

## Grand vocabulaire (option 1)

Ajout de tokens « distracteurs » qui n'apparaissent jamais dans une bonne
réponse (`MINI_EXTRA_TOKENS`), pour imiter la longue queue d'un vocabulaire
de LLM. Avantages centrés par groupe, 600 pas, seed 0.

Vocabulaire 512, réponses de 5 tokens, échauffement 30 pas (récompense 0,73,
masse de la queue sous q ≈ 4 %, qui tombe à 0,4 % pendant le RL) :

| Bruit | REINFORCE | TIS | SC |
|---|---|---|---|
| 0,1 | 0,91 | 0,92 | 0,92 |
| 0,3 | 0,82 | 0,88 | 0,86 |

Vocabulaire 512, réponses de 10 tokens (`MINI_PROMPT=10`), échauffement 40 pas :

| Bruit | REINFORCE | TIS | SC |
|---|---|---|---|
| 0,2 | 0,71 | 0,71 | 0,71 |
| 0,4 | 0,51 (en baisse) | 0,63 | 0,65 |

Lecture : le grand vocabulaire ne crée pas de séparation entre SC et TIS.
Le RL écrase vite la queue, ce qui retire aux tokens rares leur poids dans
la dérive. REINFORCE reste la seule méthode clairement pénalisée.
