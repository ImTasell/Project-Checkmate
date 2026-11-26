import * as tf from "@tensorflow/tfjs";
import type { HumanMoveSample } from "../data/pgn.js";
import { classifyMoveAnnotation, type MoveAnnotation } from "../data/decision.js";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";
import { encodeMoveTactics, TACTICAL_FEATURES } from "../chess/tactics.js";
import { encodeModelInput, getMoverElo, MODEL_INPUT_FEATURES } from "./conditioning.js";

export const DECISION_QUALITY_LABELS = [
  "good",
  "brilliant",
  "interesting",
  "mistake",
  "dubious",
  "blunder",
  "worst",
] as const;

export type DecisionQualityLabel = typeof DECISION_QUALITY_LABELS[number];
export const MOVE_CONTEXT_FEATURES = 64 + 64 + 4;
export const DECISION_QUALITY_INPUT_FEATURES =
  MODEL_INPUT_FEATURES + MOVE_CONTEXT_FEATURES + TACTICAL_FEATURES;

const PROMOTION_INDEX: Record<string, number> = { q: 0, r: 1, b: 2, n: 3 };

export interface DecisionQualityOptions {
  epochs?: number;
  batchSize?: number;
  validationSplit?: number;
  archetypeModel?: ArchetypeEncoderModel;
  logProgress?: boolean;
}

export interface ClassMetrics {
  label: DecisionQualityLabel;
  support: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface DecisionQualityMetrics {
  samples: number;
  accuracy: number;
  macroF1: number;
  classes: ClassMetrics[];
  confusionMatrix: number[][];
}

export function getDecisionQualityLabel(
  sample: Pick<HumanMoveSample, "moveAnnotation" | "annotationNags">,
): DecisionQualityLabel | undefined {
  const annotation: MoveAnnotation | undefined = sample.moveAnnotation
    ?? (sample.annotationNags ? classifyMoveAnnotation(sample.annotationNags) : undefined);
  if (!annotation || annotation === "other") return undefined;
  return DECISION_QUALITY_LABELS.includes(annotation as DecisionQualityLabel)
    ? annotation as DecisionQualityLabel
    : undefined;
}

export function encodeMoveContext(moveUci: string): Float32Array {
  const match = moveUci.toLowerCase().match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/);
  if (!match?.[1] || !match[2]) {
    throw new Error("Invalid UCI move for decision-quality input: " + moveUci);
  }

  const squareIndex = (square: string): number =>
    (Number(square[1]) - 1) * 8 + square.charCodeAt(0) - 97;
  const features = new Float32Array(MOVE_CONTEXT_FEATURES);
  features[squareIndex(match[1])] = 1;
  features[64 + squareIndex(match[2])] = 1;
  const promotion = match[3];
  if (promotion) features[128 + (PROMOTION_INDEX[promotion] ?? 0)] = 1;
  return features;
}

/**
 * The played move is an input because this auxiliary model judges an observed
 * human decision. Annotation labels are strictly targets and never enter here.
 * Tactical context is reconstructed from the pre-move FEN and UCI move, so
 * existing JSONL datasets remain usable without storing derived feature arrays.
 */
export function encodeDecisionQualitySample(
  sample: HumanMoveSample,
  archetypeModel?: ArchetypeEncoderModel,
): Float32Array {
  const base = encodeModelInput(
    sample.fen,
    getMoverElo(sample),
    sample.styleFeatures,
    archetypeModel,
    sample.decisionFeatures,
  );
  const move = encodeMoveContext(sample.moveUci);
  const tactics = encodeMoveTactics(sample.fen, sample.moveUci);
  const features = new Float32Array(DECISION_QUALITY_INPUT_FEATURES);
  features.set(base, 0);
  features.set(move, MODEL_INPUT_FEATURES);
  features.set(tactics, MODEL_INPUT_FEATURES + MOVE_CONTEXT_FEATURES);
  return features;
}

export function createDecisionQualityModel(): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [DECISION_QUALITY_INPUT_FEATURES],
    units: 128,
    activation: "relu",
    kernelInitializer: "heNormal",
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: DECISION_QUALITY_LABELS.length, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export function encodeDecisionQualityTarget(sample: HumanMoveSample): number | undefined {
  const label = getDecisionQualityLabel(sample);
  return label === undefined ? undefined : DECISION_QUALITY_LABELS.indexOf(label);
}

export async function trainDecisionQualityModel(
  model: tf.Sequential,
  samples: HumanMoveSample[],
  options: DecisionQualityOptions = {},
): Promise<tf.History> {
  const labeled = samples
    .map((sample) => ({ sample, target: encodeDecisionQualityTarget(sample) }))
    .filter((item): item is { sample: HumanMoveSample; target: number } => item.target !== undefined);

  if (labeled.length < 10) {
    throw new Error("Need at least 10 explicitly annotated human moves; found " + labeled.length + ".");
  }
  const presentClasses = new Set(labeled.map((item) => item.target));
  if (presentClasses.size < 2) {
    throw new Error("Need at least two distinct annotation classes to train a decision-quality model.");
  }

  const xs = tf.tensor2d(
    labeled.map(({ sample }) => Array.from(encodeDecisionQualitySample(sample, options.archetypeModel))),
    [labeled.length, DECISION_QUALITY_INPUT_FEATURES],
  );
  const ys = tf.tensor1d(labeled.map(({ target }) => target), "int32");
  try {
    return await model.fit(xs, ys, {
      epochs: options.epochs ?? 10,
      batchSize: options.batchSize ?? 32,
      validationSplit: options.validationSplit ?? 0.1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (options.logProgress === false) return;
          const loss = logs?.loss?.toFixed(4) ?? "n/a";
          const accuracy = logs?.acc?.toFixed(4) ?? logs?.accuracy?.toFixed(4) ?? "n/a";
          console.log("quality epoch " + (epoch + 1) + ": loss=" + loss + " accuracy=" + accuracy);
        },
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
  }
}

export function evaluateDecisionQuality(
  expected: readonly number[],
  predicted: readonly number[],
): DecisionQualityMetrics {
  if (expected.length !== predicted.length || expected.length === 0) {
    throw new Error("Decision-quality metrics require non-empty expected and predicted arrays of equal length.");
  }
  const classCount = DECISION_QUALITY_LABELS.length;
  const confusionMatrix = Array.from({ length: classCount }, () => Array(classCount).fill(0) as number[]);
  let correct = 0;
  expected.forEach((target, index) => {
    const guess = predicted[index];
    if (!Number.isInteger(target) || target < 0 || target >= classCount
      || guess === undefined || !Number.isInteger(guess) || guess < 0 || guess >= classCount) {
      throw new Error("Decision-quality metrics received an invalid class index.");
    }
    confusionMatrix[target]![guess] = (confusionMatrix[target]![guess] ?? 0) + 1;
    if (target === guess) correct += 1;
  });

  const classes = DECISION_QUALITY_LABELS.map((label, index): ClassMetrics => {
    const truePositive = confusionMatrix[index]?.[index] ?? 0;
    const support = confusionMatrix[index]?.reduce((sum, value) => sum + value, 0) ?? 0;
    const predictedCount = confusionMatrix.reduce((sum, row) => sum + (row[index] ?? 0), 0);
    const precision = predictedCount ? truePositive / predictedCount : 0;
    const recall = support ? truePositive / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { label, support, precision, recall, f1 };
  });
  const supported = classes.filter((item) => item.support > 0);
  return {
    samples: expected.length,
    accuracy: correct / expected.length,
    macroF1: supported.reduce((sum, item) => sum + item.f1, 0) / Math.max(1, supported.length),
    classes,
    confusionMatrix,
  };
}

export function predictDecisionQuality(
  model: tf.LayersModel,
  sample: HumanMoveSample,
  archetypeModel?: ArchetypeEncoderModel,
): Array<{ label: DecisionQualityLabel; probability: number }> {
  const features = encodeDecisionQualitySample(sample, archetypeModel);
  const input = tf.tensor2d([Array.from(features)], [1, DECISION_QUALITY_INPUT_FEATURES]);
  const prediction = model.predict(input) as tf.Tensor;
  const values = prediction.dataSync();
  input.dispose();
  prediction.dispose();
  return DECISION_QUALITY_LABELS
    .map((label, index) => ({ label, probability: values[index] ?? 0 }))
    .sort((a, b) => b.probability - a.probability);
}
