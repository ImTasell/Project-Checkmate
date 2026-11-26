import { loadArchetypeEncoder } from "../data/archetypes.js";
import { loadDecisionQualityModel } from "../model/decision-quality-storage.js";
import { loadPolicyModel } from "../model/storage.js";
import { chooseHumanLikeMove } from "../model/selection.js";

function parseOptionalPositiveNumber(value: string | undefined, label: string): number | undefined {
  if (value === undefined || value === "-") return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(label + " must be a positive number.");
  }
  return parsed;
}

function parseSeed(value: string | undefined): number | undefined {
  if (value === undefined || value === "-") return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > 0xffffffff) {
    throw new Error("Seed must be an integer between 0 and 4294967295.");
  }
  return parsed >>> 0;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const [
  fenArg,
  policyDirectory = "models/human-policy",
  qualityDirectory,
  eloArg,
  temperatureArg = "1.15",
  qualityWeightArg = "0",
  seedArg,
] = process.argv.slice(2);

if (!fenArg) {
  throw new Error(
    "Usage: npm run choose:move -- \"<FEN>\" [policy-directory] [quality-directory|-] [player-Elo|-] [temperature] [quality-weight] [seed|-]",
  );
}

const playerElo = parseOptionalPositiveNumber(eloArg, "Player Elo");
const temperature = Number(temperatureArg);
const qualityWeight = Number(qualityWeightArg);
if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 20) {
  throw new Error("Temperature must be greater than 0 and at most 20.");
}
if (!Number.isFinite(qualityWeight) || qualityWeight < 0 || qualityWeight > 2) {
  throw new Error("Quality weight must be between 0 and 2.");
}
if (qualityWeight > 0 && (!qualityDirectory || qualityDirectory === "-")) {
  throw new Error("A quality-model directory is required when quality weight is greater than zero.");
}

const seed = parseSeed(seedArg);
const policy = await loadPolicyModel(policyDirectory);
const quality = qualityDirectory && qualityDirectory !== "-"
  ? await loadDecisionQualityModel(qualityDirectory)
  : undefined;

try {
  const choice = chooseHumanLikeMove(
    policy.model,
    {
      fen: fenArg,
      ...(playerElo !== undefined ? { playerElo } : {}),
    },
    {
      policyArchetypeModel: policy.archetypeModel,
      ...(quality ? {
        qualityModel: quality.model,
        qualityArchetypeModel: quality.archetypeModel,
      } : {}),
      temperature,
      qualityWeight,
      ...(seed !== undefined ? { random: seededRandom(seed) } : {}),
    },
  );

  if (!choice) {
    console.log("No legal moves: the position is checkmate or stalemate.");
  } else {
    console.log("Selected move: " + choice.move);
    console.log("Ranked candidates:");
    for (const item of choice.rankedMoves.slice(0, 12)) {
      const percent = (value: number) => (value * 100).toFixed(1).padStart(5) + "%";
      const qualityText = item.positiveAnnotationProbability === undefined
        ? "quality model unavailable"
        : "positive=" + percent(item.positiveAnnotationProbability)
          + " critical=" + percent(item.criticalAnnotationProbability ?? 0);
      console.log(
        item.move + " policy=" + percent(item.policyProbability)
        + " selection=" + percent(item.selectionProbability)
        + " " + qualityText,
      );
    }
  }

  console.log("Temperature: " + temperature);
  console.log("Quality weight: " + qualityWeight);
  if (seed !== undefined) console.log("Seed: " + seed);
  console.log("Quality adjustment is advisory annotation likelihood, not an engine evaluation.");
} finally {
  policy.model.dispose();
  quality?.model.dispose();
}
