import { describe, expect, it } from "vitest";
import { importPgn } from "../src/data/pgn.js";

describe("compact human move samples", () => {
  it("stores FEN and move labels without expanded feature arrays", () => {
    const pgn = `[Event "Compact sample"]
[Result "1-0"]

1. e4 e5 1-0
`;
    const { samples } = importPgn(pgn);
    expect(samples.length).toBe(2);
    expect(samples[0]?.fen).toContain(" w ");
    expect(samples[0]?.moveUci).toBe("e2e4");
    expect(samples[0]?.moveIndex).toBe(796);
    expect("features" in (samples[0] ?? {})).toBe(false);
    expect(samples[0]?.styleFeatures).toHaveLength(9);
  });
});
