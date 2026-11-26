import { describe, expect, it } from "vitest";
import { encodeMove, encodePosition, POSITION_FEATURES } from "../src/chess/position.js";

describe("position encoding", () => {
  it("encodes the starting position into 773 features", () => {
    const features = encodePosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(features).toHaveLength(POSITION_FEATURES);
    expect(features[768]).toBe(1);
    expect(features[769]).toBe(1);
    expect(features[770]).toBe(1);
    expect(features[771]).toBe(1);
    expect(features[772]).toBe(1);
    expect(features.reduce((sum, value) => sum + value, 0)).toBe(32 + 5);
  });

  it("uses a1=0 and h8=4095 for move indexing", () => {
    expect(encodeMove("a1a1")).toBe(0);
    expect(encodeMove("h8h8")).toBe(4095);
    expect(encodeMove("e7e8q")).toBe(52 * 64 + 60);
  });

  it("rejects malformed UCI moves", () => {
    expect(() => encodeMove("e2e9")).toThrow();
    expect(() => encodeMove("e2e4xx")).toThrow();
  });
});
