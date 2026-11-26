import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { LayersModel } from "@tensorflow/tfjs";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";
import { DECISION_QUALITY_LABELS, DECISION_QUALITY_INPUT_FEATURES } from "./decision-quality.js";
import { MODEL_INPUT_FEATURES } from "./conditioning.js";

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
  await writeFile(resolve(directory, "quality-conditioning.json"), `${JSON.stringify({
    version: 1,
    inputFeatures: DECISION_QUALITY_INPUT_FEATURES,
    policyContextFeatures: MODEL_INPUT_FEATURES,
    labels: DECISION_QUALITY_LABELS,
    moveFeatures: ["from-square-one-hot-64", "to-square-one-hot-64", "promotion-one-hot-4"],
    targetSource: "explicit PGN NAG annotations only",
    archetypeModel: archetypeModel ?? null,
  }, null, 2)}\n`, "utf8");
}
