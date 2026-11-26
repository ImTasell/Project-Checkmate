import { describe, expect, it } from "vitest";
import { POSITION_FEATURES } from "../src/chess/position.js";
import { STYLE_FEATURES } from "../src/data/style.js";
import {
  encodeModelInput,
  encodeSkillFeatures,
  encodeStyleFeatures,
  getMoverElo,
  MODEL_INPUT_FEATURES,
  SKILL_FEATURES,
} from "../src/model/conditioning.js";

const whiteToMove = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const blackToMove = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";

describe("human skill conditioning", () => {
  it("normalizes Elo and marks known ratings", () => {
    expect(Array.from(encodeSkillFeatures(400))).toEqual([0, 1]);
    expect(Array.from(encodeSkillFeatures(1600))).toEqual([0.5, 1]);
    expect(Array.from(encodeSkillFeatures(2800))).toEqual([1, 1]);
  });

  it("clamps out-of-range ratings and distinguishes unknown ratings", () => {
    expect(Array.from(encodeSkillFeatures(100))).toEqual([0, 1]);
    expect(Array.from(encodeSkillFeatures(3500))).toEqual([1, 1]);
    expect(Array.from(encodeSkillFeatures(undefined))).toEqual([0.5, 0]);
  });

  it("resolves the mover Elo from the side to move for legacy samples", () => {
    expect(getMoverElo({ fen: whiteToMove, whiteElo: 1200, blackElo: 1800 })).toBe(1200);
    expect(getMoverElo({ fen: blackToMove, whiteElo: 1200, blackElo: 1800 })).toBe(1800);
  });

  it("appends skill and style features to the existing position encoding", () => {
    const style = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const features = encodeModelInput(whiteToMove, 1600, style);
    expect(features).toHaveLength(MODEL_INPUT_FEATURES);
    expect(MODEL_INPUT_FEATURES).toBe(POSITION_FEATURES + SKILL_FEATURES + STYLE_FEATURES);
    expect(features[POSITION_FEATURES]).toBe(0.5);
    expect(features[POSITION_FEATURES + 1]).toBe(1);
    expect(Array.from(features.slice(POSITION_FEATURES + SKILL_FEATURES))).toEqual(style);
  });

  it("provides neutral style features for legacy samples and validates profile shape", () => {
    expect(Array.from(encodeStyleFeatures())).toEqual(new Array(STYLE_FEATURES).fill(0));
    expect(() => encodeStyleFeatures([0.1])).toThrow();
    expect(() => encodeStyleFeatures([0, 0, 0, 0, 0, 0, 0, 0, Number.NaN])).toThrow();
  });
});
