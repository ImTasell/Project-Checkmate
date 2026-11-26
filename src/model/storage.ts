import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import type { LayersModel } from "@tensorflow/tfjs";
import {
  validateArchetypeEncoder,
  type ArchetypeEncoderModel,
} from "../data/archetypes.js";
import { DECISION_FEATURE_NAMES } from "../data/decision.js";
import { MODEL_INPUT_FEATURES } from "./conditioning.js";
import { MOVE_CLASSES } from "./policy.js";

interface StoredLayersModel {
  modelTopology?: unknown;
  weightsManifest?: Array<{
    paths?: string[];
    weights?: tf.io.WeightsManifestEntry[];
  }>;
}

interface PolicyModelMetadata {
  version?: number;
  modelInputFeatures?: number;
  archetypeConditioning?: string;
  archetypeModel?: unknown;
  decisionFeatureNames?: string[];
}

export interface LoadedPolicyModel {
  model: LayersModel;
  archetypeModel?: ArchetypeEncoderModel;
}

export async function savePolicyModel(
  model: LayersModel,
  outputDirectory: string,
  archetypeModel?: ArchetypeEncoderModel,
): Promise<void> {
  const directory = resolve(outputDirectory);
  await mkdir(directory, { recursive: true });

  await model.save({
    save: async (artifacts) => {
      if (!artifacts.modelTopology || !artifacts.weightSpecs || !artifacts.weightData) {
        throw new Error("Model save did not produce topology and weights.");
      }
      const weights = artifacts.weightData instanceof ArrayBuffer
        ? Buffer.from(artifacts.weightData)
        : Buffer.concat(artifacts.weightData.map((part) => Buffer.from(part)));
      const topologyJson = JSON.stringify(artifacts.modelTopology);
      const specsJson = JSON.stringify(artifacts.weightSpecs);
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
          modelTopologyBytes: Buffer.byteLength(topologyJson),
          weightSpecsBytes: Buffer.byteLength(specsJson),
          weightDataBytes: weights.byteLength,
        },
      };
    },
  });

  const metadata = {
    version: 1,
    modelInputFeatures: MODEL_INPUT_FEATURES,
    archetypeConditioning: archetypeModel ? "enabled" : "disabled",
    archetypeModel: archetypeModel ?? null,
    decisionFeatureNames: DECISION_FEATURE_NAMES,
  };
  await writeFile(resolve(directory, "conditioning.json"), JSON.stringify(metadata, null, 2) + "\n", "utf8");
}

export async function loadPolicyModel(modelDirectory: string): Promise<LoadedPolicyModel> {
  const directory = resolve(modelDirectory);
  const [modelJsonText, weights, metadataText] = await Promise.all([
    readFile(resolve(directory, "model.json"), "utf8"),
    readFile(resolve(directory, "weights.bin")),
    readFile(resolve(directory, "conditioning.json"), "utf8"),
  ]);

  let storedModel: StoredLayersModel;
  let metadata: PolicyModelMetadata;
  try {
    storedModel = JSON.parse(modelJsonText) as StoredLayersModel;
    metadata = JSON.parse(metadataText) as PolicyModelMetadata;
  } catch {
    throw new Error("Policy model files contain invalid JSON.");
  }

  if (metadata.version !== 1
    || metadata.modelInputFeatures !== MODEL_INPUT_FEATURES
    || JSON.stringify(metadata.decisionFeatureNames) !== JSON.stringify(DECISION_FEATURE_NAMES)
    || (metadata.archetypeConditioning !== "enabled" && metadata.archetypeConditioning !== "disabled")
    || ((metadata.archetypeConditioning === "enabled") !== Boolean(metadata.archetypeModel))) {
    throw new Error("Policy model schema is incompatible. Retrain it with npm run train.");
  }

  const manifest = storedModel.weightsManifest?.[0];
  if (!storedModel.modelTopology || !manifest?.weights
    || !manifest.paths || manifest.paths.length !== 1 || manifest.paths[0] !== "weights.bin") {
    throw new Error("Policy model is missing a valid topology or weights manifest.");
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

  if (model.inputs[0]?.shape?.[1] !== MODEL_INPUT_FEATURES
    || model.outputs[0]?.shape?.[1] !== MOVE_CLASSES) {
    model.dispose();
    throw new Error("Loaded policy model tensor shapes do not match its metadata.");
  }

  const archetypeModel = metadata.archetypeModel
    ? validateArchetypeEncoder(metadata.archetypeModel)
    : undefined;
  return archetypeModel ? { model, archetypeModel } : { model };
}
