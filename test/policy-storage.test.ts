import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { afterEach, describe, expect, it } from "vitest";
import { createPolicyModel } from "../src/model/policy.js";
import { loadPolicyModel, savePolicyModel } from "../src/model/storage.js";
import { MODEL_INPUT_FEATURES } from "../src/model/conditioning.js";
import { MOVE_CLASSES } from "../src/model/policy.js";

let directory = "";

afterEach(async () => {
  if (directory) {
    await rm(directory, { recursive: true, force: true });
    directory = "";
  }
});

describe("policy model artifacts", () => {
  it("saves and reloads policy weights and validates the current conditioning schema", async () => {
    await tf.ready();
    directory = await mkdtemp(join(tmpdir(), "checkmate-policy-model-"));
    const original = createPolicyModel();
    await savePolicyModel(original, directory);

    const metadata = JSON.parse(await readFile(join(directory, "conditioning.json"), "utf8"));
    expect(metadata.modelInputFeatures).toBe(MODEL_INPUT_FEATURES);
    expect(metadata.version).toBe(1);

    const loaded = await loadPolicyModel(directory);
    expect(loaded.model.inputs[0]?.shape?.[1]).toBe(MODEL_INPUT_FEATURES);
    expect(loaded.model.outputs[0]?.shape?.[1]).toBe(MOVE_CLASSES);

    loaded.model.dispose();
    original.dispose();
  });
});
