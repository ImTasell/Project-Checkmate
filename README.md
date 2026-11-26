# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- 773-value board encoding: 12 piece planes, side to move, and castling rights.
- Two skill-conditioning values: normalized mover Elo and a flag indicating whether that rating is known.
- Nine behavioral style values accumulated from each player's earlier moves across imported games.
- Persistent name-derived player identities, Elo history, move counters, and raw style statistics.
- Duplicate-game detection so re-importing a PGN does not inflate profiles or duplicate samples.
- Incremental JSONL appending so new PGN imports extend the existing training dataset.
- Unsupervised, standardized k-means clustering that discovers recurring player behavior groups from accumulated profiles.
- Confidence/sample count is used to filter profiles with too little history, not as a behavioral clustering dimension.
- Out-of-order dated games are rejected to prevent later-game behavior from leaking into earlier move predictions.
- 784 model inputs total: 773 board, 2 skill, and 9 behavior values.
- 4,096-class from/to move policy.
- TensorFlow.js policy model and training loop.
- Compact JSONL dataset output and portable model artifact saving.
- Unit tests and GitHub Actions build/test workflow.

## Setup

Requires Node.js 20 or newer.

```sh
npm install
npm run build
npm test
```

## Build a human-game dataset

Download PGN games from a source whose data license and terms permit your intended use. Keep large raw PGNs outside Git (the `data/raw/` directory is ignored).

```sh
npm run pgn:dataset -- ./data/raw/human-games.pgn
```

The dataset command appends new samples to `data/processed/human-moves.jsonl` and updates `data/processed/player-profiles.json`. Optional second and third arguments select custom dataset and profile paths:

```sh
npm run pgn:dataset -- ./data/raw/another-batch.pgn ./data/processed/human-moves.jsonl ./data/processed/player-profiles.json
```

Keep using the same dataset and profile paths when importing additional PGNs. The profile database stores name-derived player IDs, cumulative raw behavior counters, game/move counts, latest known Elo, last-seen dates, and processed game hashes. Re-importing the same games does not append duplicate samples.

To rebuild from scratch, remove both the JSONL dataset and its matching profile database before re-importing. Do not reset only one: the profile database is what tracks which games have already been added.

Player identity currently comes from normalized PGN `White`/`Black` names. This lets repeated usernames match across files, but two different people with the same display name can collide. Platform-specific account IDs are the preferred future identity source. Anonymous PGN placeholders are not tracked.

Each JSONL sample includes the mover's and opponent's stable player IDs when available, plus a nine-value style fingerprint snapshotted before that exact move. The fingerprint includes prior moves from earlier imported games and earlier moves in the current game. Games with valid dates are sorted oldest-first within each imported file. A dated game older than the stored history of either named player is rejected rather than conditioning on future behavior. For files without complete dates, source order is preserved, so import those files oldest-first.

The style vector order is:

1. Capture rate.
2. Check rate.
3. Forward pawn-advance rate.
4. Average centrality of prior non-pawn piece destinations.
5. Early queen activity rate (first ten moves by that player).
6. Whether the player has castled.
7. Non-castling king-move rate.
8. Non-pawn piece activity rate.
9. Confidence, increasing with the number of prior moves observed (saturates at twelve).

These are measurable behavioral signals, not manually assigned labels such as "aggressive" or "positional". They are imperfect proxies and should not be interpreted as a complete chess personality.

## Discover player archetypes

After importing a substantial set of games, cluster the persistent player profiles:

```sh
npm run players:cluster
```

This writes `data/processed/player-archetypes.json`. Optional arguments select custom paths, a fixed number of clusters (or `auto`), and the minimum observed moves required for each player:

```sh
npm run players:cluster -- ./data/processed/player-profiles.json ./data/processed/player-archetypes.json auto 24
npm run players:cluster -- ./data/processed/player-profiles.json ./data/processed/player-archetypes.json 4 40
```

The clustering pipeline standardizes eight behavioral measurements and uses deterministic k-means. Automatic cluster-count selection compares candidate counts using a silhouette score. The confidence feature is deliberately excluded from clustering; it is only used indirectly through the minimum-history filter. The resulting labels are short descriptions of the cluster's strongest deviations from the overall player population, not official chess personality categories. Inspect cluster sizes and centroid values before using these archetypes in model training. A small or behaviorally uniform dataset may correctly produce no clusters.

## Train the first policy

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy
```

The trained TensorFlow.js model is saved as `model.json` and `weights.bin`. Training conditions each move prediction on the rating and observed behavior of the player who made that move. The inference helper `rankLegalMoves()` accepts the same Elo and style vector so a caller can provide a desired or live profile.

**Model compatibility:** the model input width is 784. Retrain models created by earlier commits; existing JSONL samples remain readable.

Training currently uses move-from/move-to labels; promotion type is retained in the dataset's UCI move but is not yet a separate model output.

## Direction

1. Validate archetype stability and move-imitation quality on held-out players and games.
2. Condition policy training on discovered archetypes and compare results against continuous style vectors.
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add opponent adaptation and outcome-aware learning only after the human imitation baseline is measurable.
