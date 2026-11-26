# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- 773-value board encoding: 12 piece planes, side to move, and castling rights.
- Two skill-conditioning values: normalized mover Elo and a flag indicating whether that rating is known (775 model inputs total).
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

Each JSONL record stores a compact FEN, chosen human UCI move, move class, game result, ply, and available White/Black Elo values. It also records the mover's Elo and opponent's Elo for that specific ply. Position and skill features are generated during training instead of storing 775 floats per move, keeping datasets substantially smaller. Older records without per-move ratings remain supported; the mover's rating is resolved from the FEN side to move and the White/Black Elo fields.

## Train the first policy

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy
```

The trained TensorFlow.js model is saved as `model.json` and `weights.bin`. Training conditions each move prediction on the rating of the player who made that move, rather than always feeding White's rating or using a game-level average. Unknown ratings receive a neutral numeric value with a separate missing-rating flag. Training currently uses move-from/move-to labels; promotion type is retained in the dataset's UCI move but is not yet a separate model output.

## Direction

1. Validate dataset parsing and training on real PGNs.
2. Add explicit player-style conditioning and held-out-game evaluation (splitting by game, not individual positions).
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add opponent adaptation and outcome-aware learning only after the human imitation baseline is measurable.
