import { readFile } from "node:fs/promises";
import { BEHAVIOR_FEATURE_NAMES } from "./clustering.js";
import { STYLE_FEATURES } from "./style.js";

export const MAX_STYLE_ARCHETYPES = 6;
export const ARCHETYPE_FEATURES = MAX_STYLE_ARCHETYPES + 1;
export const MIN_ARCHETYPE_CONFIDENCE = 0.25;

export interface ArchetypeEncoderModel {
  featureNames: string[];
  featureMeans: number[];
  featureDeviations: number[];
  archetypes: Array<{
    id: string;
    label: string;
    standardizedCentroid: number[];
  }>;
}

export interface NearestArchetype {
  id: string;
  index: number;
  distance: number;
}

function squaredDistance(a: readonly number[], b: readonly number[]): number {
  return a.reduce((sum, value, index) => {
    const delta = value - (b[index] ?? 0);
    return sum + delta * delta;
  }, 0);
}

export function validateArchetypeEncoder(value: unknown): ArchetypeEncoderModel {
  if (typeof value !== "object" || value === null) throw new Error("Invalid archetype model: expected an object.");
  const candidate = value as Partial<ArchetypeEncoderModel>;
  if (!Array.isArray(candidate.featureNames)
    || candidate.featureNames.length !== BEHAVIOR_FEATURE_NAMES.length
    || !candidate.featureNames.every((name, index) => name === BEHAVIOR_FEATURE_NAMES[index])
    || !Array.isArray(candidate.featureMeans)
    || candidate.featureMeans.length !== BEHAVIOR_FEATURE_NAMES.length
    || !candidate.featureMeans.every((number) => typeof number === "number" && Number.isFinite(number))
    || !Array.isArray(candidate.featureDeviations)
    || candidate.featureDeviations.length !== BEHAVIOR_FEATURE_NAMES.length
    || !candidate.featureDeviations.every((number) => typeof number === "number" && Number.isFinite(number) && number > 0)
    || !Array.isArray(candidate.archetypes)
    || candidate.archetypes.length < 2
    || candidate.archetypes.length > MAX_STYLE_ARCHETYPES) {
    throw new Error("Invalid archetype model: feature schema or cluster count is incompatible.");
  }

  for (const archetype of candidate.archetypes) {
    if (!archetype || typeof archetype.id !== "string" || typeof archetype.label !== "string"
      || !Array.isArray(archetype.standardizedCentroid)
      || archetype.standardizedCentroid.length !== BEHAVIOR_FEATURE_NAMES.length
      || !archetype.standardizedCentroid.every((number) => typeof number === "number" && Number.isFinite(number))) {
      throw new Error("Invalid archetype model: malformed cluster centroid.");
    }
  }

  return {
    featureNames: [...candidate.featureNames],
    featureMeans: [...candidate.featureMeans],
    featureDeviations: [...candidate.featureDeviations],
    archetypes: candidate.archetypes.map((archetype) => ({
      id: archetype.id,
      label: archetype.label,
      standardizedCentroid: [...archetype.standardizedCentroid],
    })),
  };
}

export async function loadArchetypeEncoder(path: string): Promise<ArchetypeEncoderModel | undefined> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error(`Invalid JSON in archetype file: ${path}`);
  }
  return validateArchetypeEncoder(parsed);
}

export function findNearestArchetype(
  styleFeatures: readonly number[] | undefined,
  model: ArchetypeEncoderModel | undefined,
): NearestArchetype | undefined {
  if (!model || !styleFeatures || styleFeatures.length !== STYLE_FEATURES) return undefined;
  const confidence = styleFeatures[8] ?? 0;
  if (!Number.isFinite(confidence) || confidence < MIN_ARCHETYPE_CONFIDENCE) return undefined;

  const standardized = BEHAVIOR_FEATURE_NAMES.map((_, index) => {
    const value = styleFeatures[index];
    if (value === undefined || !Number.isFinite(value)) return 0;
    return (Math.max(0, Math.min(1, value)) - (model.featureMeans[index] ?? 0))
      / (model.featureDeviations[index] ?? 1);
  });

  let nearest: NearestArchetype | undefined;
  model.archetypes.forEach((archetype, index) => {
    const distance = Math.sqrt(squaredDistance(standardized, archetype.standardizedCentroid));
    if (!nearest || distance < nearest.distance) nearest = { id: archetype.id, index, distance };
  });
  return nearest;
}

/** Fixed-width one-hot archetype vector plus a known-profile bit. */
export function encodeArchetypeFeatures(
  styleFeatures: readonly number[] | undefined,
  model: ArchetypeEncoderModel | undefined,
): Float32Array {
  const encoded = new Float32Array(ARCHETYPE_FEATURES);
  const nearest = findNearestArchetype(styleFeatures, model);
  if (!nearest || nearest.index >= MAX_STYLE_ARCHETYPES) return encoded;
  encoded[nearest.index] = 1;
  encoded[MAX_STYLE_ARCHETYPES] = 1;
  return encoded;
}
