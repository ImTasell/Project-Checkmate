import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { afterEach, describe, expect, it } from "vitest";
import { createDecisionQualityModel, DECISION_QUALITY_INPUT_FEATURES } from "../src/model/decision-quality.js";
import {
  loadDecisionQualityModel,
  saveDecisionQualityModel,
} from "../src/model/decision-quality-storage.js";

let directory = "";

afterEach(async () => {
  if (directory) {
    await rm(directory, { recursive: true, force: true });
    directory = "";
  }
});

describe("decision-quality model artifacts", () => {
  it("saves and reloads the current feature schema and TensorFlow weights", async () => {
    await tf.ready();
    directory = await mkdtemp(join(tmpdir(), "checkmate-quality-model-"));
    const original = createDecisionQualityModel();
    await saveDecisionQualityModel(original, directory);

    const metadata = JSON.parse(await readFile(join(directory, "quality-conditioning.json"), "utf8"));
    expect(metadata.inputFeatures).toBe(DECISION_QUALITY_INPUT_FEATURES);
    expect(metadata.version).toBe(2);

    const loaded = await loadDecisionQualityModel(directory);
    expect(loaded.model.inputs[0]?.shape?.[1]).toBe(DECISION_QUALITY_INPUT_FEATURES);
    expect(loaded.model.outputs[0]?.shape?.[1]).toBe(7);

    loaded.model.dispose();
    original.dispose();
  });
});
