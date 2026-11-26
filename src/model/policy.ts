import * as tf from "@tensorflow/tfjs";
import type { HumanMoveSample } from "../data/pgn.js";
import { encodeArchetypeFeatures, type ArchetypeEncoderModel } from "../data/archetypes.js";
import {
  CONTINUOUS_MODEL_INPUT_FEATURES,
  encodeDecisionFeatures,
  encodeSkillFeatures,
  encodeStyleFeatures,
  encodeTrainingSample,
  MODEL_INPUT_FEATURES,
  SKILL_FEATURES,
} from "./conditioning.js";
import { POSITION_FEATURES } from "../chess/position.js";

export const MOVE_CLASSES = 64 * 64;

export function createPolicyModel(): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [MODEL_INPUT_FEATURES],
    units: 128,
    activation: "relu",
    kernelInitializer: "heNormal",
  }));
  model.add(tf.layers.dropout({ rate: 0.15 }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: MOVE_CLASSES, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export interface TrainingOptions {
  epochs?: number;
  batchSize?: number;
  validationSplit?: number;
  archetypeModel?: ArchetypeEncoderModel;
  logProgress?: boolean;
}

export async function trainPolicy(
  model: tf.Sequential,
  samples: HumanMoveSample[],
  options: TrainingOptions = {},
): Promise<tf.History> {
  if (samples.length < 2) throw new Error("Need at least two human move samples to train.");
  const xs = tf.tensor2d(
    samples.map((sample) => Array.from(encodeTrainingSample(sample, options.archetypeModel))),
    [samples.length, MODEL_INPUT_FEATURES],
  );
  const ys = tf.tensor1d(samples.map((sample) => sample.moveIndex), "int32");

  try {
    return await model.fit(xs, ys, {
      epochs: options.epochs ?? 5,
      batchSize: options.batchSize ?? 64,
      validationSplit: options.validationSplit ?? 0.1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (options.logProgress === false) return;
          const loss = logs?.loss?.toFixed(4) ?? "n/a";
          const accuracy = logs?.acc?.toFixed(4) ?? logs?.accuracy?.toFixed(4) ?? "n/a";
          console.log(`epoch ${epoch + 1}: loss=${loss} accuracy=${accuracy}`);
        },
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
  }
}

export function rankLegalMoves(
  model: tf.LayersModel,
  fenFeatures: Float32Array,
  legalMoves: Array<{ from: string; to: string; promotion?: string }>,
  playerElo?: number,
  styleFeatures?: readonly number[],
  archetypeModel?: ArchetypeEncoderModel,
  decisionFeatures?: readonly number[],
): Array<{ move: string; score: number }> {
  if (!legalMoves.length) return [];
  if (fenFeatures.length !== POSITION_FEATURES) {
    throw new Error(`Expected ${POSITION_FEATURES} position features, received ${fenFeatures.length}.`);
  }

  const inputFeatures = new Float32Array(MODEL_INPUT_FEATURES);
  inputFeatures.set(fenFeatures, 0);
  inputFeatures.set(encodeSkillFeatures(playerElo), POSITION_FEATURES);
  inputFeatures.set(encodeStyleFeatures(styleFeatures), POSITION_FEATURES + SKILL_FEATURES);
  const archetypeFeatures = encodeArchetypeFeatures(styleFeatures, archetypeModel);
  inputFeatures.set(archetypeFeatures, CONTINUOUS_MODEL_INPUT_FEATURES);
  inputFeatures.set(encodeDecisionFeatures(decisionFeatures), CONTINUOUS_MODEL_INPUT_FEATURES + archetypeFeatures.length);
  const input = tf.tensor2d([Array.from(inputFeatures)], [1, MODEL_INPUT_FEATURES]);
  const prediction = model.predict(input) as tf.Tensor;
  const scores = prediction.dataSync();
  input.dispose();
  prediction.dispose();

  return legalMoves.map((move) => {
    const squareIndex = (square: string): number =>
      (Number(square[1]) - 1) * 8 + square.charCodeAt(0) - 97;
    const index = squareIndex(move.from) * 64 + squareIndex(move.to);
    return { move: `${move.from}${move.to}${move.promotion ?? ""}`, score: scores[index] ?? 0 };
  }).sort((a, b) => b.score - a.score);
}
