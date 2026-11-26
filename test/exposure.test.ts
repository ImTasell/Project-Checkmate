import { describe, expect, it } from "vitest";
import { hasUnansweredCaptureExposure } from "../src/chess/exposure.js";

describe("one-ply unanswered capture exposure", () => {
  it("flags a legal capture that has no immediate legal recapture", () => {
    const fen = "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1";
    expect(hasUnansweredCaptureExposure(fen, "e4")).toBe(true);
  });

  it("does not flag a capture that can be immediately recaptured", () => {
    const fen = "4k3/8/8/8/4P3/3b4/8/K3R3 b - - 0 1";
    expect(hasUnansweredCaptureExposure(fen, "e4")).toBe(false);
  });

  it("does not flag an empty destination square", () => {
    const fen = "4k3/8/8/8/8/3b4/8/K3R3 b - - 0 1";
    expect(hasUnansweredCaptureExposure(fen, "e4")).toBe(false);
  });
});
