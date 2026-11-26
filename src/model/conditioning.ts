import { encodePosition, POSITION_FEATURES } from "../chess/position.js";
import type { HumanMoveSample } from "../data/pgn.js";
import { ARCHETYPE_FEATURES, encodeArchetypeFeatures, type ArchetypeEncoderModel } from "../data/archetypes.js";
import { STYLE_FEATURES } from "../data/style.js";

export const SKILL_FEATURES = 2;
export const CONTINUOUS_MODEL_INPUT_FEATURES = POSITION_FEATURES + SKILL_FEATURES + STYLE_FEATURES;
/** Six fixed archetype slots plus a flag that distinguishes unknown from known profiles. */
export const MODEL_INPUT_FEATURES = CONTINUOUS_MODEL_INPUT_FEATURES + ARCHETYPE_FEATURES;

const MIN_RATING = 400;
const MAX_RATING = 2800;

export function encodeSkillFeatures(playerElo?: number): Float32Array {
  if (playerElo === undefined || !Number.isFinite(playerElo) || playerElo <= 0) {
    return new Float32Array([0.5, 0]);
  }
  const clamped = Math.min(MAX_RATING, Math.max(MIN_RATING, playerElo));
  return new Float32Array([(clamped - MIN_RATING) / (MAX_RATING - MIN_RATING), 1]);
}

export function encodeStyleFeatures(styleFeatures?: readonly number[]): Float32Array {
  if (styleFeatures === undefined) return new Float32Array(STYLE_FEATURES);
  if (styleFeatures.length !== STYLE_FEATURES) {
    throw new Error(`Expected ${STYLE_FEATURES} style features, received ${styleFeatures.length}.`);
  }

  const encoded = new Float32Array(STYLE_FEATURES);
  for (let index = 0; index < STYLE_FEATURES; index += 1) {
    const value = styleFeatures[index];
    if (value === undefined || !Number.isFinite(value)) {
      throw new Error(`Style feature ${index} must be a finite number.`);
    }
    encoded[index] = Math.max(0, Math.min(1, value));
  }
  return encoded;
}

export function getMoverElo(sample: Pick<HumanMoveSample, "fen" | "playerElo" | "whiteElo" | "blackElo">): number | undefined {
  if (sample.playerElo !== undefined) return sample.playerElo;
  const activeColor = sample.fen.split(/\s+/)[1];
  return activeColor === "b" ? sample.blackElo : sample.whiteElo;
}

export function encodeModelInput(
  fen: string,
  playerElo?: number,
  styleFeatures?: readonly number[],
  archetypeModel?: ArchetypeEncoderModel,
): Float32Array {
  const position = encodePosition(fen);
  const skill = encodeSkillFeatures(playerElo);
  const style = encodeStyleFeatures(styleFeatures);
  const archetype = encodeArchetypeFeatures(styleFeatures, archetypeModel);
  const features = new Float32Array(MODEL_INPUT_FEATURES);
  features.set(position, 0);
  features.set(skill, POSITION_FEATURES);
  features.set(style, POSITION_FEATURES + SKILL_FEATURES);
  features.set(archetype, CONTINUOUS_MODEL_INPUT_FEATURES);
  return features;
}

export function encodeTrainingSample(
  sample: HumanMoveSample,
  archetypeModel?: ArchetypeEncoderModel,
): Float32Array {
  return encodeModelInput(sample.fen, getMoverElo(sample), sample.styleFeatures, archetypeModel);
}
