import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { clusterPlayerProfiles } from "../data/clustering.js";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";
import { encodeTrainingSample, MODEL_INPUT_FEATURES } from "../model/conditioning.js";
import { readSamplesJsonl } from "../data/jsonl.js";
import { loadPlayerProfileDatabase } from "../data/players.js";
import type { HumanMoveSample } from "../data/pgn.js";
import { createPolicyModel, MOVE_CLASSES, trainPolicy } from "../model/policy.js";

interface PolicyMetrics {
  top1Accuracy: number;
  top5Accuracy: number;
  samples: number;
}

function stableBucket(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % 5;
}

async function evaluatePolicy(
  model: tf.LayersModel,
  samples: HumanMoveSample[],
  archetypeModel?: ArchetypeEncoderModel,
): Promise<PolicyMetrics> {
  let correctTop1 = 0;
  let correctTop5 = 0;
  const batchSize = 64;

  for (let start = 0; start < samples.length; start += batchSize) {
    const batch = samples.slice(start, start + batchSize);
    const inputs = tf.tensor2d(
      batch.map((sample) => Array.from(encodeTrainingSample(sample, archetypeModel))),
      [batch.length, MODEL_INPUT_FEATURES],
    );
    const prediction = model.predict(inputs) as tf.Tensor;
    const values = prediction.dataSync();

    batch.forEach((sample, row) => {
      const offset = row * MOVE_CLASSES;
      const target = sample.moveIndex;
      let bestIndex = -1;
      let bestScore = Number.NEGATIVE_INFINITY;
      const top: Array<{ index: number; score: number }> = [];

      for (let move = 0; move < MOVE_CLASSES; move += 1) {
        const score = values[offset + move] ?? 0;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = move;
        }
        if (top.length < 5 || score > (top[4]?.score ?? Number.NEGATIVE_INFINITY)) {
          top.push({ index: move, score });
          top.sort((a, b) => b.score - a.score);
          if (top.length > 5) top.pop();
        }
      }
      if (bestIndex === target) correctTop1 += 1;
      if (top.some((item) => item.index === target)) correctTop5 += 1;
    });

    inputs.dispose();
    prediction.dispose();
  }

  return {
    top1Accuracy: correctTop1 / samples.length,
    top5Accuracy: correctTop5 / samples.length,
    samples: samples.length,
  };
}

const [
  datasetArg = "data/processed/human-moves.jsonl",
  profilesArg = "data/processed/player-profiles.json",
  reportArg = "data/processed/archetype-evaluation.json",
  epochsArg = "5",
] = process.argv.slice(2);

const epochs = Number(epochsArg);
if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
  throw new Error("Epoch count must be an integer between 1 and 50.");
}

const samples: HumanMoveSample[] = [];
for await (const sample of readSamplesJsonl(resolve(datasetArg))) samples.push(sample);
const profiles = await loadPlayerProfileDatabase(resolve(profilesArg));
const playerIds = Object.keys(profiles.players).sort();
const testPlayerIds = new Set(playerIds.filter((id) => stableBucket(id) === 0));
const trainPlayerIds = new Set(playerIds.filter((id) => !testPlayerIds.has(id)));

const trainingSamples = samples.filter((sample) => sample.playerId && trainPlayerIds.has(sample.playerId));
const testSamples = samples.filter((sample) => sample.playerId && testPlayerIds.has(sample.playerId));
const excludedUnknownPlayerSamples = samples.length - trainingSamples.length - testSamples.length;

if (trainPlayerIds.size < 6 || testPlayerIds.size < 1 || trainingSamples.length < 20 || testSamples.length < 10) {
  throw new Error(
    `Not enough player-disjoint data: ${trainPlayerIds.size} train players, ${testPlayerIds.size} test players, ${trainingSamples.length} train moves, ${testSamples.length} test moves. Need at least 6 train players, 1 test player, 20 train moves and 10 test moves.`,
  );
}

const trainingProfiles = {
  ...profiles,
  players: Object.fromEntries(
    Object.entries(profiles.players).filter(([id]) => trainPlayerIds.has(id)),
  ),
};
const clustering = clusterPlayerProfiles(trainingProfiles, { minMoves: 24, minPlayers: 6 });
if (clustering.archetypes.length < 2) {
  throw new Error(clustering.reason ?? "Training players did not produce at least two usable archetypes.");
}
const archetypeModel: ArchetypeEncoderModel = {
  featureNames: clustering.featureNames,
  featureMeans: clustering.featureMeans,
  featureDeviations: clustering.featureDeviations,
  archetypes: clustering.archetypes.map(({ id, label, standardizedCentroid }) => ({
    id,
    label,
    standardizedCentroid,
  })),
};

await tf.ready();
const continuousModel = createPolicyModel();
const archetypeModelNetwork = createPolicyModel();
const initialWeights = continuousModel.getWeights().map((weight) => weight.clone());
archetypeModelNetwork.setWeights(initialWeights);
initialWeights.forEach((weight) => weight.dispose());

console.log(`Training/evaluating continuous-style baseline with ${trainingSamples.length} moves...`);
await trainPolicy(continuousModel, trainingSamples, {
  epochs,
  batchSize: 64,
  validationSplit: 0.1,
  logProgress: false,
});
console.log(`Training/evaluating archetype-conditioned policy with ${clustering.archetypes.length} archetypes...`);
await trainPolicy(archetypeModelNetwork, trainingSamples, {
  epochs,
  batchSize: 64,
  validationSplit: 0.1,
  archetypeModel,
  logProgress: false,
});

const continuous = await evaluatePolicy(continuousModel, testSamples);
const archetypeConditioned = await evaluatePolicy(archetypeModelNetwork, testSamples, archetypeModel);
continuousModel.dispose();
archetypeModelNetwork.dispose();

const report = {
  generatedAt: new Date().toISOString(),
  split: "deterministic player-disjoint 80/20 hash split",
  trainPlayers: trainPlayerIds.size,
  testPlayers: testPlayerIds.size,
  trainMoves: trainingSamples.length,
  testMoves: testSamples.length,
  excludedUnknownPlayerSamples,
  clustering: {
    algorithm: clustering.algorithm,
    archetypeCount: clustering.selectedClusters,
    silhouetteScore: clustering.silhouetteScore,
    archetypes: clustering.archetypes.map(({ id, label, size, centroid }) => ({ id, label, size, centroid })),
  },
  models: {
    continuousStyle: continuous,
    archetypeConditioned,
    delta: {
      top1Accuracy: archetypeConditioned.top1Accuracy - continuous.top1Accuracy,
      top5Accuracy: archetypeConditioned.top5Accuracy - continuous.top5Accuracy,
    },
  },
};

const outputPath = resolve(reportArg);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Held-out top-1 accuracy: continuous=${(continuous.top1Accuracy * 100).toFixed(2)}%, archetype=${(archetypeConditioned.top1Accuracy * 100).toFixed(2)}%`);
console.log(`Held-out top-5 accuracy: continuous=${(continuous.top5Accuracy * 100).toFixed(2)}%, archetype=${(archetypeConditioned.top5Accuracy * 100).toFixed(2)}%`);
console.log(`Saved player-disjoint evaluation report to ${outputPath}`);
