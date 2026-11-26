# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and player Elo.
- 773-value position encoding: 12 piece planes, side to move, and castling rights.
- 4,096-class from/to move policy.
- TensorFlow.js policy model and training loop.
- JSONL dataset output and portable model artifact saving.
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

Each JSONL record contains the position FEN, encoded features, chosen human UCI move, move class, result, ply, and available player Elo values.

## Train the first policy

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy
```

The trained TensorFlow.js model is saved as `model.json` and `weights.bin`. Training currently uses move-from/move-to labels; promotion type is retained in the dataset's UCI move but is not yet a separate model output.

## Direction

1. Validate dataset parsing and training on real PGNs.
2. Add held-out-game evaluation and rating/style-aware splits.
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add outcome-based learning and self-play after the human imitation baseline is measurable.
