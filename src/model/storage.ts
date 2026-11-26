import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { LayersModel } from "@tensorflow/tfjs";
import { MODEL_INPUT_FEATURES } from "./conditioning.js";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";

/** Save a TFJS LayersModel and the exact feature transform needed at inference. */
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
          modelTopologyType: "JSON",
          modelTopologyBytes: Buffer.byteLength(topologyJson),
          weightSpecsBytes: Buffer.byteLength(specsJson),
          weightDataBytes: weights.byteLength,
        },
      };
    },
  });

  await writeFile(resolve(directory, "conditioning.json"), `${JSON.stringify({
    version: 1,
    modelInputFeatures: MODEL_INPUT_FEATURES,
    archetypeConditioning: archetypeModel ? "enabled" : "disabled",
    archetypeModel: archetypeModel ?? null,
  }, null, 2)}\n`, "utf8");
}
