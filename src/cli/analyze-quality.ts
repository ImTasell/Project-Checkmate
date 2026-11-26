import { loadDecisionQualityModel } from "../model/decision-quality-storage.js";
import { rankLegalMovesByDecisionQuality } from "../model/decision-quality.js";

const [fenArg, modelArg = "models/decision-quality", eloArg] = process.argv.slice(2);
if (!fenArg) {
  throw new Error(
    "Usage: npm run analyze:quality -- \"<FEN>\" [model-directory] [player-Elo]",
  );
}

const playerElo = eloArg === undefined ? undefined : Number(eloArg);
if (playerElo !== undefined && (!Number.isFinite(playerElo) || playerElo <= 0)) {
  throw new Error("Player Elo must be a positive number.");
}

const loaded = await loadDecisionQualityModel(modelArg);
try {
  const scores = rankLegalMovesByDecisionQuality(
    loaded.model,
    {
      fen: fenArg,
      ...(playerElo !== undefined ? { playerElo } : {}),
    },
    loaded.archetypeModel,
  );

  if (scores.length === 0) {
    console.log("No legal moves: the position is checkmate or stalemate.");
  } else {
    console.log("Move | positive annotation | interesting | critical annotation | top label");
    for (const score of scores) {
      const percent = (value: number) => (value * 100).toFixed(1).padStart(5) + "%";
      console.log(
        score.move + " | "
        + percent(score.positiveAnnotationProbability) + " | "
        + percent(score.interestingAnnotationProbability) + " | "
        + percent(score.criticalAnnotationProbability) + " | "
        + score.predictedLabel + " (" + percent(score.confidence) + ")",
      );
    }
  }

  console.log("");
  console.log("Advisory only: these are learned PGN-annotation probabilities, not engine evaluations.");
} finally {
  loaded.model.dispose();
}
