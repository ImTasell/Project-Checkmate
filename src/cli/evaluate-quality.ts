import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { clusterPlayerProfiles } from "../data/clustering.js";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";
import { readSamplesJsonl } from "../data/jsonl.js";
import { loadPlayerProfileDatabase } from "../data/players.js";
import type { HumanMoveSample } from "../data/pgn.js";
import {
  createDecisionQualityModel,
  DECISION_QUALITY_LABELS,
  encodeDecisionQualitySample,
  encodeDecisionQualityTarget,
  evaluateDecisionQuality,
  getDecisionQualityLabel,
  DECISION_QUALITY_INPUT_FEATURES,
  trainDecisionQualityModel,
} from "../model/decision-quality.js";

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

async function predictBatch(
  model: tf.LayersModel,
  samples: HumanMoveSample[],
  archetypeModel?: ArchetypeEncoderModel,
): Promise<number[]> {
  const predictions: number[] = [];
  const batchSize = 64;
  for (let start = 0; start < samples.length; start += batchSize) {
    const batch = samples.slice(start, start + batchSize);
    const inputs = tf.tensor2d(
      batch.map((sample) => Array.from(encodeDecisionQualitySample(sample, archetypeModel))),
      [batch.length, DECISION_QUALITY_INPUT_FEATURES],
    );
    const output = model.predict(inputs) as tf.Tensor;
    const values = output.dataSync();
    batch.forEach((_, row) => {
      let bestIndex = 0;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (let label = 0; label < DECISION_QUALITY_LABELS.length; label += 1) {
        const score = values[row * DECISION_QUALITY_LABELS.length + label] ?? 0;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = label;
        }
      }
      predictions.push(bestIndex);
    });
    inputs.dispose();
    output.dispose();
  }
  return predictions;
}

const [
  datasetArg = "data/processed/human-moves.jsonl",
  profilesArg = "data/processed/player-profiles.json",
  reportArg = "data/processed/decision-quality-evaluation.json",
  epochsArg = "10",
] = process.argv.slice(2);
const epochs = Number(epochsArg);
if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
  throw new Error("Epoch count must be an integer between 1 and 50.");
}

const samples: HumanMoveSample[] = [];
for await (const sample of readSamplesJsonl(resolve(datasetArg))) samples.push(sample);
const profiles = await loadPlayerProfileDatabase(resolve(profilesArg));
const labeled = samples.filter((sample) => {
  const playerId = sample.playerId;
  if (!playerId) return false;
  const profile = profiles.players[playerId];
  return profile !== undefined
    && profile.moves >= 24
    && getDecisionQualityLabel(sample) !== undefined;
});
const playerIds = [...new Set(labeled.map((sample) => sample.playerId as string))]
  .sort((a, b) => stableHash(a) - stableHash(b) || a.localeCompare(b));
const testPlayerCount = Math.max(1, Math.floor(playerIds.length * 0.2));
const testPlayerIds = new Set(playerIds.slice(0, testPlayerCount));
const trainPlayerIds = new Set(playerIds.slice(testPlayerCount));
const trainingSamples = labeled.filter((sample) => sample.playerId && trainPlayerIds.has(sample.playerId));
const testSamples = labeled.filter((sample) => sample.playerId && testPlayerIds.has(sample.playerId));

if (trainPlayerIds.size < 6 || testPlayerIds.size < 1 || trainingSamples.length < 20 || testSamples.length < 10) {
  throw new Error(
    `Not enough player-disjoint annotated data: ${trainPlayerIds.size} train players, ${testPlayerIds.size} test players, ${trainingSamples.length} train labeled moves, ${testSamples.length} test labeled moves. Need at least 6 train players, 1 test player, 20 train labels and 10 test labels.`,
  );
}

const trainLabels = trainingSamples.map(encodeDecisionQualityTarget).filter((label): label is number => label !== undefined);
if (new Set(trainLabels).size < 2) {
  throw new Error("Training-player annotations contain fewer than two decision-quality classes.");
}
const testLabels = testSamples.map(encodeDecisionQualityTarget).filter((label): label is number => label !== undefined);

const trainingProfiles = {
  ...profiles,
  players: Object.fromEntries(
    Object.entries(profiles.players).filter(([id]) => trainPlayerIds.has(id)),
  ),
};
const clustering = clusterPlayerProfiles(trainingProfiles, { minMoves: 24, minPlayers: 6 });
const archetypeModel: ArchetypeEncoderModel | undefined = clustering.archetypes.length >= 2
  ? {
      featureNames: clustering.featureNames,
      featureMeans: clustering.featureMeans,
      featureDeviations: clustering.featureDeviations,
      archetypes: clustering.archetypes.map(({ id, label, standardizedCentroid }) => ({
        id,
        label,
        standardizedCentroid,
      })),
    }
  : undefined;

await tf.ready();
const model = createDecisionQualityModel();
await trainDecisionQualityModel(model, trainingSamples, {
  epochs,
  batchSize: 32,
  validationSplit: 0.1,
  ...(archetypeModel ? { archetypeModel } : {}),
  logProgress: false,
});
const predictions = await predictBatch(model, testSamples, archetypeModel);
const metrics = evaluateDecisionQuality(testLabels, predictions);
model.dispose();

const classCounts = Object.fromEntries(DECISION_QUALITY_LABELS.map((label, index) => [
  label,
  {
    train: trainLabels.filter((value) => value === index).length,
    test: testLabels.filter((value) => value === index).length,
  },
]));
const report = {
  generatedAt: new Date().toISOString(),
  split: "deterministic player-disjoint 80/20 hash split",
  trainPlayers: trainPlayerIds.size,
  testPlayers: testPlayerIds.size,
  trainAnnotatedMoves: trainingSamples.length,
  testAnnotatedMoves: testSamples.length,
  unannotatedMovesExcluded: samples.length - labeled.length,
  archetypesUsed: archetypeModel?.archetypes.length ?? 0,
  classCounts,
  metrics,
};

const outputPath = resolve(reportArg);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Held-out annotated-move accuracy: ${(metrics.accuracy * 100).toFixed(2)}%; macro-F1: ${metrics.macroF1.toFixed(3)}`);
for (const item of metrics.classes.filter((entry) => entry.support > 0)) {
  console.log(`- ${item.label}: support=${item.support}, precision=${item.precision.toFixed(3)}, recall=${item.recall.toFixed(3)}, F1=${item.f1.toFixed(3)}`);
}
console.log(`Saved player-disjoint decision-quality report to ${outputPath}`);
