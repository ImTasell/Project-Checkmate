import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import type { LayersModel } from "@tensorflow/tfjs";
import {
  validateArchetypeEncoder,
  type ArchetypeEncoderModel,
} from "../data/archetypes.js";
import {
  DECISION_QUALITY_LABELS,
  DECISION_QUALITY_INPUT_FEATURES,
  MOVE_CONTEXT_FEATURES,
} from "./decision-quality.js";
import { TACTICAL_FEATURES, TACTICAL_FEATURE_NAMES } from "../chess/tactics.js";
import { MODEL_INPUT_FEATURES } from "./conditioning.js";

interface StoredLayersModel {
  modelTopology?: unknown;
  weightsManifest?: Array<{
    paths?: string[];
    weights?: tf.io.WeightsManifestEntry[];
  }>;
}

interface QualityModelMetadata {
  version?: number;
  inputFeatures?: number;
  policyContextFeatures?: number;
  moveContextFeatures?: number;
  tacticalContextFeatures?: number;
  labels?: string[];
  tacticalFeatures?: string[];
  archetypeModel?: unknown;
}

export interface LoadedDecisionQualityModel {
  model: LayersModel;
  archetypeModel?: ArchetypeEncoderModel;
}

export async function saveDecisionQualityModel(
  model: LayersModel,
  outputDirectory: string,
  archetypeModel?: ArchetypeEncoderModel,
): Promise<void> {
  const directory = resolve(outputDirectory);
  await mkdir(directory, { recursive: true });
  await model.save({
    save: async (artifacts) => {
      if (!artifacts.modelTopology || !artifacts.weightSpecs || !artifacts.weightData) {
        throw new Error("Decision-quality model save did not produce topology and weights.");
      }
      const weights = artifacts.weightData instanceof ArrayBuffer
        ? Buffer.from(artifacts.weightData)
        : Buffer.concat(artifacts.weightData.map((part) => Buffer.from(part)));
      const modelJson = {
        format: "layers-model",
        generatedBy: "Project Checkmate",
        convertedBy: null,
        modelTopology: artifacts.modelTopology,
        weightsManifest: [{ paths: ["weights.bin"], weights: artifacts.weightSpecs }],
      };
      await writeFile(resolve(directory, "model.json"), JSON.stringify(modelJson, null, 2));
      await writeFile(resolve(directory, "weights.bin"), weights);
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON" as const,
          modelTopologyBytes: Buffer.byteLength(JSON.stringify(artifacts.modelTopology)),
          weightSpecsBytes: Buffer.byteLength(JSON.stringify(artifacts.weightSpecs)),
          weightDataBytes: weights.byteLength,
        },
      };
    },
  });
  const metadata = {
    version: 2,
    inputFeatures: DECISION_QUALITY_INPUT_FEATURES,
    policyContextFeatures: MODEL_INPUT_FEATURES,
    moveContextFeatures: MOVE_CONTEXT_FEATURES,
    tacticalContextFeatures: TACTICAL_FEATURES,
    labels: DECISION_QUALITY_LABELS,
    moveFeatures: ["from-square-one-hot-64", "to-square-one-hot-64", "promotion-one-hot-4"],
    tacticalFeatures: TACTICAL_FEATURE_NAMES,
    targetSource: "explicit PGN NAG annotations only",
    archetypeModel: archetypeModel ?? null,
  };
  await writeFile(resolve(directory, "quality-conditioning.json"), JSON.stringify(metadata, null, 2) + "\n", "utf8");
}

export async function loadDecisionQualityModel(
  modelDirectory: string,
): Promise<LoadedDecisionQualityModel> {
  const directory = resolve(modelDirectory);
  const [modelJsonText, weights, metadataText] = await Promise.all([
    readFile(resolve(directory, "model.json"), "utf8"),
    readFile(resolve(directory, "weights.bin")),
    readFile(resolve(directory, "quality-conditioning.json"), "utf8"),
  ]);

  let storedModel: StoredLayersModel;
  let metadata: QualityModelMetadata;
  try {
    storedModel = JSON.parse(modelJsonText) as StoredLayersModel;
    metadata = JSON.parse(metadataText) as QualityModelMetadata;
  } catch {
    throw new Error("Decision-quality model files contain invalid JSON.");
  }

  if (metadata.version !== 2
    || metadata.inputFeatures !== DECISION_QUALITY_INPUT_FEATURES
    || metadata.policyContextFeatures !== MODEL_INPUT_FEATURES
    || metadata.moveContextFeatures !== MOVE_CONTEXT_FEATURES
    || metadata.tacticalContextFeatures !== TACTICAL_FEATURES
    || JSON.stringify(metadata.labels) !== JSON.stringify(DECISION_QUALITY_LABELS)
    || JSON.stringify(metadata.tacticalFeatures) !== JSON.stringify(TACTICAL_FEATURE_NAMES)) {
    throw new Error(
      "Decision-quality model schema is incompatible. Retrain it with npm run train:quality.",
    );
  }

  const manifest = storedModel.weightsManifest?.[0];
  if (!storedModel.modelTopology || !manifest?.weights
    || !manifest.paths || manifest.paths.length !== 1 || manifest.paths[0] !== "weights.bin") {
    throw new Error("Decision-quality model is missing a valid topology or weights manifest.");
  }

  const copiedWeights = new Uint8Array(weights.byteLength);
  copiedWeights.set(weights);
  const model = await tf.loadLayersModel({
    load: async () => ({
      modelTopology: storedModel.modelTopology as {},
      weightSpecs: manifest.weights ?? [],
      weightData: copiedWeights.buffer,
      format: "layers-model",
      generatedBy: "Project Checkmate",
    }),
  });

  const inputFeatures = model.inputs[0]?.shape?.[1];
  const outputFeatures = model.outputs[0]?.shape?.[1];
  if (inputFeatures !== DECISION_QUALITY_INPUT_FEATURES
    || outputFeatures !== DECISION_QUALITY_LABELS.length) {
    model.dispose();
    throw new Error("Loaded decision-quality model tensor shapes do not match its metadata.");
  }

  const archetypeModel = metadata.archetypeModel
    ? validateArchetypeEncoder(metadata.archetypeModel)
    : undefined;
  return archetypeModel ? { model, archetypeModel } : { model };
}
