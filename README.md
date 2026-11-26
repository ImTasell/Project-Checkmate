# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- 773-value board encoding: 12 piece planes, side to move, and castling rights.
- Two skill-conditioning values: normalized mover Elo and a flag indicating whether that rating is known.
- Nine behavioral style values derived incrementally from each player's earlier moves in the current game.
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

Each JSONL record stores a compact FEN, chosen human UCI move, move class, game result, ply, available player Elo values, and a nine-value behavioral fingerprint. Style features are calculated from that player's earlier moves in the same game and snapshotted before the current move is observed. This avoids leaking the target move or future decisions into its own input. The profile resets at the start of each game; persistent cross-game player identities and style clustering are later steps.

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

Position, skill, and style features are generated or compactly stored rather than writing expanded board tensors into JSONL. Older records without per-move ratings or style fields remain supported; those missing features receive neutral/unknown encodings.

## Train the first policy

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy
```

The trained TensorFlow.js model is saved as `model.json` and `weights.bin`. Training conditions each move prediction on the rating and observed behavior of the player who made that move. The inference helper `rankLegalMoves()` accepts the same Elo and style vector so a caller can provide a desired or live profile.

**Model compatibility:** this update changes the model input width from 775 to 784. Retrain models created by earlier commits; existing JSONL datasets can be reused and regenerated to include style history.

Training currently uses move-from/move-to labels; promotion type is retained in the dataset's UCI move but is not yet a separate model output.

## Direction

1. Validate move imitation on real PGNs with held-out-game evaluation.
2. Build persistent player profiles across multiple games and cluster observed fingerprints into data-derived style archetypes.
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add opponent adaptation and outcome-aware learning only after the human imitation baseline is measurable.
