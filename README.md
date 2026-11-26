# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- 773-value board encoding: 12 piece planes, side to move, and castling rights.
- Two skill-conditioning values: normalized mover Elo and a flag indicating whether that rating is known.
- Nine behavioral style values accumulated from each player's earlier moves across imported games.
- Persistent name-derived player identities, Elo history, move counters, and raw style statistics.
- Duplicate-game detection so re-importing a PGN does not inflate profiles or duplicate samples.
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
npm run pgn:dataset -- ./data/raw/human-games.pgn ./data/processed/human-moves.jsonl
```

The optional third argument selects the persistent player-profile database. By default it is `data/processed/player-profiles.json`. Keep using the same file when importing additional PGNs; it stores name-derived player IDs, cumulative raw behavior counters, game/move counts, latest known Elo, last-seen dates, and processed game hashes.

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

These are measurable behavioral signals, not manually assigned labels such as "aggressive" or "positional". They are imperfect proxies and should not be interpreted as a complete chess personality. Clustering real player fingerprints into readable style labels can be evaluated once we have a sufficiently large dataset.

Position, skill, and style features are generated or compactly stored rather than writing expanded board tensors into JSONL. Older records without player IDs, per-move ratings, or style fields remain supported; missing features receive neutral/unknown encodings.

## Train the first policy

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy
```

The trained TensorFlow.js model is saved as `model.json` and `weights.bin`. Training conditions each move prediction on the rating and observed behavior of the player who made that move. The inference helper `rankLegalMoves()` accepts the same Elo and style vector so a caller can provide a desired or live profile.

**Model compatibility:** the model input width is 784. Retrain models created by earlier commits; existing JSONL datasets can be reused and regenerated to include persistent player history.

Training currently uses move-from/move-to labels; promotion type is retained in the dataset's UCI move but is not yet a separate model output.

## Direction

1. Validate move imitation on real PGNs with held-out-game evaluation.
2. Cluster accumulated player fingerprints into data-derived style archetypes and evaluate how well they match held-out games.
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add opponent adaptation and outcome-aware learning only after the human imitation baseline is measurable.
