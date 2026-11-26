import { describe, expect, it } from "vitest";
import { importPgn } from "../src/data/pgn.js";
import { createPlayerProfileDatabase } from "../src/data/players.js";

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
    expect(stats.games).toBe(2);
    expect(stats.positions).toBe(6);
    expect(stats.skippedGames).toBe(0);
    expect(stats.duplicateGames).toBe(0);
    expect(stats.outOfOrderGames).toBe(0);
    expect(samples[0]?.moveUci).toBe("e2e4");
    expect(samples[0]?.whiteElo).toBe(1200);
    expect(samples[0]?.blackElo).toBe(1300);
    expect(samples[0]?.playerElo).toBe(1200);
    expect(samples[0]?.opponentElo).toBe(1300);
    expect(samples[0]?.playerId).toBe("pgn-name:human a");
    expect(samples[0]?.opponentId).toBe("pgn-name:human b");
    expect(samples[0]?.result).toBe("1-0");
    expect(samples[1]?.moveUci).toBe("e7e5");
    expect(samples[1]?.playerElo).toBe(1300);
    expect(samples[1]?.opponentElo).toBe(1200);
    expect(samples[4]?.moveUci).toBe("d2d4");
  });

  it("carries each named player's style into later games without leaking the current move", () => {
    const database = createPlayerProfileDatabase();
    const firstGame = `[Event "First"]
[Date "2026.01.01"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "1500"]
[BlackElo "1400"]
[Result "*"]

1. e4 e5 2. d4 *
`;
    const secondGame = `[Event "Second"]
[Date "2026.01.02"]
[White "  ALICE  "]
[Black "Charlie"]
[WhiteElo "1510"]
[Result "*"]

1. Nf3 Nf6 *
`;

    const first = importPgn(firstGame, database);
    expect(first.samples[0]?.styleFeatures?.[8]).toBe(0);
    expect(database.players["pgn-name:alice"]?.games).toBe(1);
    expect(database.players["pgn-name:alice"]?.moves).toBe(2);

    const second = importPgn(secondGame, database);
    expect(second.samples[0]?.playerId).toBe("pgn-name:alice");
    expect(second.samples[0]?.styleFeatures?.[2]).toBe(1);
    expect(second.samples[0]?.styleFeatures?.[8]).toBeCloseTo(2 / 12);
    expect(database.players["pgn-name:alice"]?.games).toBe(2);
    expect(database.players["pgn-name:alice"]?.moves).toBe(3);
    expect(second.samples[0]?.gameDate).toBe("2026-01-02");
  });

  it("ignores duplicate imports and refuses dated games older than a player's stored history", () => {
    const database = createPlayerProfileDatabase();
    const recent = `[Event "Recent"]
[Date "2026.01.03"]
[White "Alice"]
[Black "Bob"]
[Result "*"]

1. e4 e5 *
`;
    const older = `[Event "Older"]
[Date "2026.01.02"]
[White "Alice"]
[Black "Carol"]
[Result "*"]

1. d4 d5 *
`;

    const first = importPgn(recent, database);
    expect(first.samples).toHaveLength(2);
    const duplicate = importPgn(recent, database);
    expect(duplicate.samples).toHaveLength(0);
    expect(duplicate.stats.duplicateGames).toBe(1);
    expect(database.players["pgn-name:alice"]?.games).toBe(1);

    const outOfOrder = importPgn(older, database);
    expect(outOfOrder.samples).toHaveLength(0);
    expect(outOfOrder.stats.outOfOrderGames).toBe(1);
    expect(database.players["pgn-name:alice"]?.games).toBe(1);
  });

  it("imports clock annotations and pre-move decision context without target leakage", () => {
    const pgn = `[Event "Decision annotations"]
[White "Alice"]
[Black "Bob"]
[TimeControl "300+2"]
[Result "*"]

1. e4? {[%clk 0:04:50] [%emt 0:00:10]} e5 {[%clk 0:04:55] [%emt 0:00:05]}
2. Nf3 $4 {[%clk 0:04:30] [%emt 0:00:20]} Nc6
3. Bb5 *
`;
    const { samples, stats } = importPgn(pgn);
    expect(stats.annotatedMoves).toBe(2);
    expect(stats.clockAnnotatedMoves).toBe(3);

    expect(samples[0]?.decisionFeatures).toEqual([0, 1, 0, 0]);
    expect(samples[0]?.moveAnnotation).toBe("mistake");
    expect(samples[0]?.annotationNags).toEqual([2]);
    expect(samples[0]?.elapsedSeconds).toBe(10);
    expect(samples[0]?.clockRemainingSeconds).toBe(290);

    // White's second move begins with the clock reported after White's first move.
    expect(samples[2]?.decisionFeatures?.[0]).toBeCloseTo(1 - 290 / 300);
    // The earlier explicit ? annotation is available as history, while the
    // current $4 blunder marker must not be visible in this sample's inputs.
    expect(samples[2]?.decisionFeatures?.[2]).toBe(1);
    expect(samples[2]?.decisionFeatures?.[3]).toBe(1);
    expect(samples[2]?.moveAnnotation).toBe("blunder");

    // White's third move sees both prior White moves and their annotations.
    expect(samples[4]?.decisionFeatures?.[2]).toBe(1);
    expect(samples[4]?.decisionFeatures?.[3]).toBe(1);
  });

  it("records an unanswered exposure only when the opponent captures it on the next move", () => {
    const database = createPlayerProfileDatabase();
    const pgn = "[Event \"Exposure history\"]\n[White \"Alice\"]\n[Black \"Bob\"]\n[Result \"*\"]\n\n1. d4 d5 2. c4 dxc4 *\n";
    importPgn(pgn, database);
    const white = database.players["pgn-name:alice"]?.styleState;
    expect(white?.unansweredExposureOpportunities).toBe(1);
    expect(white?.acceptedExposureOpportunities).toBe(1);
    expect(white?.contextObservedMoves).toBe(2);
  });

});
