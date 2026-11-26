import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import {
  encodeMoveTactics,
  TACTICAL_FEATURE_NAMES,
  TACTICAL_FEATURES,
} from "../src/chess/tactics.js";

describe("candidate-move tactical context", () => {
  it("exports a stable 29-feature order and encodes the moving piece", () => {
    const features = encodeMoveTactics(new Chess().fen(), "e2e4");
    expect(TACTICAL_FEATURES).toBe(29);
    expect(TACTICAL_FEATURE_NAMES).toHaveLength(TACTICAL_FEATURES);
    expect(features).toHaveLength(TACTICAL_FEATURES);
    expect(features[0]).toBe(0);
    expect(features[7]).toBe(1);
    expect(features[13]).toBe(0);
  });

  it("identifies captures and the captured piece type", () => {
    const fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1";
    const features = encodeMoveTactics(fen, "e4d5");
    expect(features[0]).toBe(1);
    expect(features[13]).toBe(1);
    expect(features[14]).toBe(0);
  });

  it("distinguishes en passant, castling, and promotion", () => {
    const enPassant = encodeMoveTactics(
      "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
      "e5d6",
    );
    expect(enPassant[0]).toBe(1);
    expect(enPassant[1]).toBe(1);
    expect(enPassant[13]).toBe(1);

    const castling = encodeMoveTactics(
      "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
      "e1g1",
    );
    expect(castling[2]).toBe(1);

    const promotion = encodeMoveTactics(
      "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
      "a7a8q",
    );
    expect(promotion[3]).toBe(1);
    expect(promotion[7]).toBe(1);
  });

  it("recognizes checkmate after the candidate move", () => {
    const features = encodeMoveTactics(
      "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
      "g6g7",
    );
    expect(features[4]).toBe(1);
    expect(features[5]).toBe(1);
  });

  it("detects an immediate legal capture of the moved piece", () => {
    const features = encodeMoveTactics(
      "4k3/8/8/4p3/8/8/6N1/4K3 w - - 0 1",
      "g2f4",
    );
    expect(features[19]).toBe(1);
    expect(features[20]).toBe(0);
    expect(features[23]).toBe(1);
    expect(features[24]).toBe(0.25);
  });

  it("rejects malformed and illegal moves", () => {
    const fen = new Chess().fen();
    expect(() => encodeMoveTactics(fen, "not-a-move")).toThrow();
    expect(() => encodeMoveTactics(fen, "e2e5")).toThrow();
  });
});
