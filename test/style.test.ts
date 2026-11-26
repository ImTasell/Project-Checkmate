import { describe, expect, it } from "vitest";
import { importPgn } from "../src/data/pgn.js";
import { PlayerStyleAccumulator, STYLE_FEATURE_NAMES } from "../src/data/style.js";

describe("online player style fingerprints", () => {
  it("starts unknown and only records behavior after the move is observed", () => {
    const profile = new PlayerStyleAccumulator();
    expect(profile.snapshot()).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);

    profile.observe({
      piece: "p",
      from: "e2",
      to: "e4",
      flags: "b",
      san: "e4",
    }, "w");

    const snapshot = profile.snapshot();
    expect(snapshot[2]).toBe(1);
    expect(snapshot[8]).toBeCloseTo(1 / 12);
  });

  it("captures a player's prior-game behavior without leaking the target move", () => {
    const pgn = `[Event "Style history"]
[White "Player A"]
[Black "Player B"]
[Result "*"]

1. e4 d5 2. exd5 Qxd5 *
`;
    const { samples } = importPgn(pgn);

    // White's first move has no history; Black's first move has no history either.
    expect(samples[0]?.styleFeatures?.[8]).toBe(0);
    expect(samples[1]?.styleFeatures?.[8]).toBe(0);

    // Before White's second move, the profile contains only the earlier e-pawn push.
    expect(samples[2]?.styleFeatures?.[2]).toBe(1);
    expect(samples[2]?.styleFeatures?.[0]).toBe(0);

    // Before White's next move there is no additional white observation in this game.
    expect(samples[2]?.styleFeatures?.[8]).toBeCloseTo(1 / 12);
  });

  it("exports a stable, documented feature order", () => {
    expect(STYLE_FEATURE_NAMES).toEqual([
      "captureRate",
      "checkRate",
      "pawnAdvanceRate",
      "centralization",
      "earlyQueenActivity",
      "castlingTendency",
      "kingMoveRate",
      "pieceActivityRate",
      "confidence",
    ]);
  });

  it("tracks quiet moves and unanswered piece exposure separately", () => {
    const style = new PlayerStyleAccumulator();
    style.observe({
      piece: "n",
      from: "b1",
      to: "c3",
      flags: "n",
      san: "Nc3",
      offersUnansweredExposure: true,
    }, "w");
    style.observe({
      piece: "p",
      from: "e2",
      to: "e4",
      flags: "b",
      san: "e4",
    }, "w");
    style.observeAcceptedExposure();

    const state = style.serialize();
    expect(state.contextObservedMoves).toBe(2);
    expect(state.quietMoves).toBe(2);
    expect(state.unansweredExposureOpportunities).toBe(1);
    expect(state.acceptedExposureOpportunities).toBe(1);
  });

  it("preserves compatibility with older profiles that lack context counters", () => {
    const style = new PlayerStyleAccumulator({
      moves: 20,
      captures: 3,
      checks: 2,
      pawnAdvances: 7,
      centralityTotal: 4,
      centralityMoves: 5,
      earlyQueenMoves: 0,
      castled: true,
      kingMoves: 1,
      activePieceMoves: 5,
    });
    expect(style.serialize().moves).toBe(20);
    expect(style.serialize().contextObservedMoves).toBe(0);
    expect(style.snapshot()[0]).toBeCloseTo(3 / 20);
  });

});
