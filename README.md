# Project Checkmate

A TypeScript chess project focused on learning move preferences from real human games. The first milestone is supervised human-move imitation; Stockfish is not the primary teacher.

## Current foundation

- PGN importer for multi-game files, move-by-move FENs, UCI moves, results, Elo, explicit NAG annotations, and clock comments.
- Persistent player identities, accumulated behavioral histories, and data-derived player archetypes.
- Move-policy conditioning on board position, player skill, continuous style, archetype, and pre-move decision context.
- A separate decision-quality classifier that learns from explicitly annotated human moves.
- Unannotated moves are excluded from decision-quality training rather than being treated as good decisions.
- Decision-quality inference evaluates a position plus the move actually played and returns probabilities for seven annotation categories.
- Legal-move analysis scores every candidate with that separate classifier and exposes positive, interesting, and critical annotation probabilities as advisory signals.
- Human-like move selection samples from the learned policy with a configurable temperature and an optional, separately weighted annotation signal.
- Model training, player-disjoint evaluation, compact JSONL datasets, and portable TensorFlow.js artifacts.

## Setup

Requires Node.js 20 or newer.

```sh
npm install
npm run build
npm test
```

## Build a human-game dataset

Import PGNs from sources whose license and terms permit your intended use:

```sh
npm run pgn:dataset -- ./data/raw/human-games.pgn
```

The importer appends samples to `data/processed/human-moves.jsonl` and updates persistent player profiles. Re-importing known games does not duplicate training samples.

### Decision context and annotations

When PGNs provide a `TimeControl` header and `[%clk ...]` comments, the importer derives a pre-move clock-pressure estimate. `[%emt ...]` is stored as metadata for the target move. Explicit suffix annotations and numeric NAGs are also retained.

The move policy receives only pre-move clock and earlier annotated-move history. Current-move annotations and elapsed time are target metadata, not policy inputs.

Annotations are human/editor judgments, not engine-verified truth. They are often sparse and subjective.

## Discover and train player archetypes

```sh
npm run players:cluster
npm run train
```

The policy uses 795 inputs and its archetype transform is stored alongside the model.

## Train the decision-quality classifier

```sh
npm run train:quality
```

Optional arguments are:

```sh
npm run train:quality -- ./data/processed/human-moves.jsonl ./models/decision-quality ./data/processed/player-archetypes.json 10
```

This trains a separate model using only explicitly annotated moves. Its inputs contain the board position, the move actually played (from/to and promotion encodings), player skill/style/archetype, pre-move decision context, and 29 move-specific tactical descriptors. Those descriptors cover captures, en passant, castling, promotions, checks, mate/stalemate, piece identities, attacked/defended destination geometry, and immediate legal capture replies. They are reconstructed from the sample FEN and UCI move, so existing JSONL datasets do not need extra stored feature fields. The annotation on the target move is used only as the training label. Supported labels are good, brilliant, interesting, mistake, dubious, blunder, and worst; generic NAGs without a quality meaning are excluded.

The model predicts how an annotated move was judged by a human/editor. This is weak supervision, not an objective engine evaluation or proof that a move is objectively bad. Attack-map counts are geometric signals and may include pinned pieces; only the immediate-capture-reply features use the opponent's legal move list. None of these descriptors is an exchange evaluation or substitute for search.

## Analyze every legal move

After training the decision-quality model, analyze a FEN:

```sh
npm run analyze:quality -- "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Optional arguments are the model directory and current player's Elo:

```sh
npm run analyze:quality -- "<FEN>" ./models/decision-quality 1600
```

The analyzer scores every legal candidate and reports the probability of a positive annotation (good/brilliant), an interesting annotation, and a critical annotation (mistake/dubious/blunder/worst). It does not combine those probabilities with policy scores or change move selection. Treat the output as advisory human-annotation likelihood, not objective chess strength.

## Select a human-like move

The policy model can now be loaded and used to sample a legal move:

```sh
npm run choose:move -- \"<FEN>\"
```

Optional positional arguments are policy directory, quality-model directory (or `-`), Elo (or `-`), temperature, quality weight, and seed:

```sh
npm run choose:move -- \"<FEN>\" ./models/human-policy ./models/decision-quality 1600 1.2 0.15 42
```

Temperature controls sampling diversity. Quality weight defaults to zero; enable it only after evaluating the annotation model on a real, player-disjoint annotated corpus. The sampler reports the original normalized policy probability and final selection probability separately. A seed makes the sampled choice reproducible. Promotion candidates currently share the policy's from/to score because promotion is not yet a separate policy output class.

## Evaluate decision-quality generalization

```sh
npm run evaluate:quality
```

This uses a deterministic player-disjoint split. Archetypes, when available, are learned from training players only. The report contains accuracy, macro-F1, per-class precision/recall/F1, class counts, and a confusion matrix. A small or highly sparse annotated corpus may fail minimum-data checks; collect more explicitly annotated games rather than silently treating unannotated moves as correct.

## Limitations and direction

The move policy still uses 4,096 from/to classes, so promotion types are not separate policy outputs yet. Decision annotations are not objective quality labels, and time-pressure signals are missing for games without clock comments. Future evaluation should include stronger player-disjoint and multi-source datasets before claims about human-like play.
