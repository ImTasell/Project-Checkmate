import { encodePosition, POSITION_FEATURES } from "../chess/position.js";
import type { HumanMoveSample } from "../data/pgn.js";

/** Two extra inputs: normalized mover Elo and whether that rating is known. */
export const SKILL_FEATURES = 2;
export const MODEL_INPUT_FEATURES = POSITION_FEATURES + SKILL_FEATURES;

const MIN_RATING = 400;
const MAX_RATING = 2800;

/**
 * Encodes a player's Elo into a stable [0, 1] range.
 * Unknown ratings use a neutral value plus a zero-valued known flag, so the
 * model can distinguish missing metadata from a genuinely average-rated player.
 */
export function encodeSkillFeatures(playerElo?: number): Float32Array {
  if (playerElo === undefined || !Number.isFinite(playerElo) || playerElo <= 0) {
    return new Float32Array([0.5, 0]);
  }

  const clamped = Math.min(MAX_RATING, Math.max(MIN_RATING, playerElo));
  return new Float32Array([
    (clamped - MIN_RATING) / (MAX_RATING - MIN_RATING),
    1,
  ]);
}

/** Resolve the mover's rating, including support for older whiteElo/blackElo datasets. */
export function getMoverElo(sample: Pick<HumanMoveSample, "fen" | "playerElo" | "whiteElo" | "blackElo">): number | undefined {
  if (sample.playerElo !== undefined) return sample.playerElo;
  const activeColor = sample.fen.split(/\s+/)[1];
  return activeColor === "b" ? sample.blackElo : sample.whiteElo;
}

export function encodeModelInput(fen: string, playerElo?: number): Float32Array {
  const position = encodePosition(fen);
  const skill = encodeSkillFeatures(playerElo);
  const features = new Float32Array(MODEL_INPUT_FEATURES);
  features.set(position, 0);
  features.set(skill, POSITION_FEATURES);
  return features;
}

export function encodeTrainingSample(sample: HumanMoveSample): Float32Array {
  return encodeModelInput(sample.fen, getMoverElo(sample));
}
