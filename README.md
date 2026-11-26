# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, game results, and per-move mover/opponent Elo metadata.
- Persistent name-derived player identities and cross-game behavior histories.
- Unsupervised standardized k-means clustering for data-derived player archetypes.
- Move-policy conditioning on continuous style and learned archetype profiles.
- PGN decision context: pre-move clock pressure and the player's history of explicitly annotated moves.
- Current-move elapsed time, remaining clock, and explicit NAG annotations are retained as target metadata, not leaked into the move-policy inputs.
- 795 model inputs total: 773 board, 2 skill, 9 continuous behavior, 7 archetype, and 4 decision-context values.
- 4,096-class from/to move policy.
- TensorFlow.js training, portable model artifacts, dataset importer, and automated evaluation tools.

## Setup

Requires Node.js 20 or newer.

```sh
npm install
npm run build
npm test
```

## Build a human-game dataset

Import PGNs from sources whose license and terms permit your intended use. Large raw PGNs stay outside Git:

```sh
npm run pgn:dataset -- ./data/raw/human-games.pgn
```

The importer appends samples to `data/processed/human-moves.jsonl` and updates persistent profiles in `data/processed/player-profiles.json`. Re-importing games already in that profile database does not duplicate samples.

### Decision context

When PGNs provide a `TimeControl` header and per-move `[%clk ...]` comments, Checkmate can derive a pre-move time-pressure estimate from the clock reported after the player's previous move. When comments provide `[%emt ...]`, elapsed thinking time is stored as metadata for the current move.

The importer also captures explicit PGN annotation glyphs (`!`, `?`, `!!`, `??`, `!?`, `?!`, and numeric NAGs). The current move's annotation and elapsed time are target metadata only. The model receives only the player's earlier annotated-move history and pre-move clock context.

Decision feature order:
1. Estimated time pressure (0-1), only when a base time and prior clock reading are available.
2. Time-known flag.
3. Prior explicitly annotated mistake rate (from annotations on earlier moves by this player).
4. Prior annotation coverage (annotated earlier moves divided by earlier moves).

Unknown clock data remains unknown instead of being fabricated. Most ordinary PGN archives omit clock comments and annotations; those records remain valid and receive neutral/unknown decision features. Explicit annotations are human/editor judgments, not ground-truth engine evaluations. They should not be interpreted as reliable labels for every move.

## Discover and use player archetypes

```sh
npm run players:cluster
npm run train -- ./data/processed/human-moves.jsonl ./models/human-policy ./data/processed/player-archetypes.json
```

Archetype transformations are saved in `conditioning.json` alongside the model and must be reused at inference. Policy input width is now 795; earlier 784-input models must be retrained. Existing JSONL data stays readable; rebuild the dataset to populate the new decision-context fields.

## Evaluate archetype conditioning

```sh
npm run players:evaluate-archetypes
```

This compares continuous-style and archetype-conditioned policies on held-out players. Results measure move imitation accuracy, not playing strength or human-likeness by themselves.

## Direction

1. Collect licensed human PGNs with reliable time annotations and explicit move annotations.
2. Evaluate move imitation by skill band, clock state, and annotation coverage.
3. Add calibrated decision-quality modeling only after enough trustworthy annotated data is available.
4. Add legal-move masking and promotion-specific move classes.
