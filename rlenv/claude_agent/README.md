# Lancer l'environnement avec un sous-agent Claude Code

Alternative à `inspect eval` quand on n'a pas de clé API : le modèle est un
sous-agent de la session Claude Code (abonnement), et son harness est celui
des sous-agents de Claude Code.

## Comment c'est isolé

- `hook.py` est un hook `PreToolUse` (déclaré dans `.claude/settings.json`).
  Pendant un run actif (`state/active.json`), chaque appel d'outil fait par un
  **sous-agent** (l'entrée du hook contient `agent_id`) est intercepté :
  - Bash : la commande est réécrite en
    `docker exec -w /workspace <conteneur> bash -lc '<commande>'`.
    Tout s'exécute donc dans le conteneur (pas de réseau, seulement le
    starter), quels que soient les `&&`, `;` ou `$(...)` de la commande ;
  - `SubagentHandback` (le rapport final) : autorisé ;
  - tout autre outil (Read, Grep, Write, web, Agent…) : refusé.
  Chaque appel est journalisé dans `rlenv/runs/<run>/agent_log.jsonl`.
- Les appels de la session principale ne sont jamais touchés.
- Testé : un sous-agent Haiku qui essaie de lire le dépôt, d'accéder au réseau
  et d'utiliser Read/Grep n'obtient que l'intérieur du conteneur
  (`rlenv/tests/test_claude_agent.py` couvre les décisions du hook).

## Déroulé d'un run

1. `python3 rlenv/claude_agent/run.py start --level L3 [--no-experiments] [--hours 2]`
   construit le starter et les images, démarre le conteneur (sans réseau,
   4 CPU, 8 Go), active le run et écrit `rlenv/runs/<run>/agent_prompt.md`
   (préambule du harness + prompt du niveau).
2. La session lance un sous-agent `general-purpose`, modèle `haiku`, avec ce
   prompt (ou l'agent `researcher` de `.claude/agents/`, chargé seulement par
   une session démarrée après sa création).
3. `python3 rlenv/claude_agent/run.py finish` récupère `losses.py` et
   `REPORT.md`, lance le test théorique et le garde-fou dans l'image du
   correcteur, écrit `results.json`, supprime le conteneur et désactive le run.
   Il copie aussi la transcription complète du sous-agent (messages, appels
   d'outils et leurs sorties) dans `rlenv/runs/<run>/transcripts/`, avec le
   modèle demandé (`<id>.meta.json`) et le modèle servi (`<id>.prefix.json`).
   Les blocs de raisonnement interne y sont vides : Claude Code ne les
   conserve pas.
   `abort` arrête un run sans le noter.

## Limites

- **Fuite du nom du dépôt.** Le contexte du sous-agent contient le dossier de
  travail de la session et le chemin de son scratchpad, qui contiennent le nom
  du dépôt. Tant que ce nom est `score-centering`, il révèle la méthode. Pour
  de vrais runs, il faut une session démarrée sur un dépôt au nom neutre.
- Pas de limite de temps ni de tokens imposée : la durée du prompt est une
  consigne. Le temps passé et le nombre de commandes sont enregistrés.
- Pas de juge LLM automatique (il passerait par l'API).
- Un seul run à la fois.
