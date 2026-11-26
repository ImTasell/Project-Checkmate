import { describe, expect, it } from "vitest";
import { BEHAVIOR_FEATURE_NAMES } from "../src/data/clustering.js";
import {
  ARCHETYPE_FEATURES,
  encodeArchetypeFeatures,
  findNearestArchetype,
  validateArchetypeEncoder,
  type ArchetypeEncoderModel,
} from "../src/data/archetypes.js";

const model: ArchetypeEncoderModel = {
  featureNames: [...BEHAVIOR_FEATURE_NAMES],
  featureMeans: new Array(8).fill(0.5),
  featureDeviations: new Array(8).fill(0.25),
  archetypes: [
    { id: "style-a", label: "capture-oriented", standardizedCentroid: [1, 1, 1, 1, 1, 1, 1, 1] },
    { id: "style-b", label: "positional", standardizedCentroid: [-1, -1, -1, -1, -1, -1, -1, -1] },
  ],
};

describe("archetype conditioning", () => {
  it("assigns an online style snapshot to the nearest learned centroid", () => {
    const nearest = findNearestArchetype([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5], model);
    expect(nearest?.id).toBe("style-a");
    expect(nearest?.index).toBe(0);
  });

  it("encodes a fixed-width one-hot vector and a known-profile flag", () => {
    const encoded = encodeArchetypeFeatures([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5], model);
    expect(encoded).toHaveLength(ARCHETYPE_FEATURES);
    expect(Array.from(encoded)).toEqual([1, 0, 0, 0, 0, 0, 1]);
  });

  it("leaves archetype conditioning unknown until enough prior behavior exists", () => {
    const encoded = encodeArchetypeFeatures([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.1], model);
    expect(Array.from(encoded)).toEqual(new Array(ARCHETYPE_FEATURES).fill(0));
    expect(Array.from(encodeArchetypeFeatures(undefined, model))).toEqual(new Array(ARCHETYPE_FEATURES).fill(0));
  });

  it("rejects incompatible or malformed cluster metadata", () => {
    expect(validateArchetypeEncoder(model)).toEqual(model);
    expect(() => validateArchetypeEncoder({ ...model, featureNames: ["wrong"] })).toThrow();
    expect(() => validateArchetypeEncoder({ ...model, archetypes: [] })).toThrow();
  });
});
