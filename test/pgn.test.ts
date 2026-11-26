import { describe, expect, it } from "vitest";
import { importPgn } from "../src/data/pgn.js";

const samplePgn = `[Event "Unit Test"]
[Site "Local"]
[Date "2026.01.01"]
[Round "1"]
[White "Human A"]
[Black "Human B"]
[WhiteElo "1200"]
[BlackElo "1300"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Second Game"]
[Result "1/2-1/2"]

1. d4 d5 1/2-1/2
`;

describe("PGN import", () => {
  it("extracts positions, UCI moves, ratings and results from multiple games", () => {
    const { samples, stats } = importPgn(samplePgn);
    expect(stats).toEqual({ games: 2, positions: 6, skippedGames: 0 });
    expect(samples[0]?.moveUci).toBe("e2e4");
    expect(samples[0]?.whiteElo).toBe(1200);
    expect(samples[0]?.blackElo).toBe(1300);
    expect(samples[0]?.playerElo).toBe(1200);
    expect(samples[0]?.opponentElo).toBe(1300);
    expect(samples[0]?.result).toBe("1-0");
    expect(samples[1]?.moveUci).toBe("e7e5");
    expect(samples[1]?.playerElo).toBe(1300);
    expect(samples[1]?.opponentElo).toBe(1200);
    expect(samples[4]?.moveUci).toBe("d2d4");
  });
});
