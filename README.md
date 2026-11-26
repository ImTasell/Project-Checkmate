# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- 773-value board encoding: 12 piece planes, side to move, and castling rights.
- Two skill-conditioning values: normalized mover Elo and a flag indicating whether that rating is known.
- Nine behavioral style values accumulated from each player's earlier moves across imported games.
- Persistent name-derived player identities, Elo history, move counters, and raw style statistics.
- Duplicate-game detection and incremental JSONL appending for reliable multi-file imports.
- Unsupervised standardized k-means clustering that discovers recurring player behavior groups from accumulated profiles.
- A fixed-width six-slot archetype encoding, assigned from each move's pre-move style snapshot.
- A player-disjoint evaluation command that compares continuous-style and archetype-conditioned policies on held-out humans.
- 791 model inputs total: 773 board, 2 skill, 9 continuous behavior, and 7 archetype values.
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

The dataset command appends new samples to `data/processed/human-moves.jsonl` and updates `data/processed/player-profiles.json`. Keep using those same files for new batches so duplicate detection and persistent style histories stay aligned.

## Discover player archetypes

After importing enough player history:

```sh
npm run players:cluster
```

This writes `data/processed/player-archetypes.json`, including the feature normalization parameters and standardized cluster centroids needed to classify new style snapshots. The confidence feature is excluded from the distance calculation and only gates profiles with too little prior behavior.

## Train with archetype conditioning

```sh
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy ./data/processed/player-archetypes.json
```

The network input is now 791 features: 773 board, 2 Elo, 9 continuous style, and 7 archetype features (six one-hot slots plus a known-profile flag). A sample's archetype is assigned from its own pre-move style snapshot, never from the player's final profile label. Unknown or short histories receive an all-zero archetype vector. The exact archetype transform is saved in `conditioning.json` with the model; load that metadata for inference and pass the corresponding encoder to `rankLegalMoves()`.

Older JSONL data remains readable. Models from the previous 784-input architecture must be retrained.

## Compare continuous style vs archetype conditioning

Run this after importing enough games for at least six training players and one held-out player:

```sh
npm run players:evaluate-archetypes
```

The evaluation uses a deterministic player-disjoint split: archetypes are discovered only from training players, and the held-out players are classified by their pre-move behavior snapshots against those training centroids. Both policies start from identical initial weights and train on the same training moves. The report compares held-out top-1 and top-5 move accuracy. Unknown-player samples are excluded from this comparison. A small dataset may fail the minimum-data checks; do not interpret tiny-cohort results as evidence that an archetype helps or hurts.

## Model limitations

The current policy has 4,096 from/to classes. Promotion type is retained in UCI records but is not a separate output class yet. The current evaluation measures imitation accuracy, not playing strength or human-likeness by itself; legal move masking, calibration, and human-likeness studies remain future work.

## Direction

1. Collect a sufficiently broad, license-compliant corpus and run player-disjoint evaluation.
2. Check cluster stability and compare imitation quality across multiple player splits and random seeds.
3. Add legal-move masking and promotion-specific move classes at inference.
4. Add opponent adaptation and outcome-aware learning only after the imitation baseline is measurable.
