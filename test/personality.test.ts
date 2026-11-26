import { describe, expect, it } from "vitest";
import { buildPlayerPersonality, buildPlayerPersonalityReport } from "../src/data/personality.js";
import { createPlayerProfileDatabase, getOrCreatePlayerProfile } from "../src/data/players.js";
import { PlayerStyleAccumulator } from "../src/data/style.js";

function createProfile() {
  const database = createPlayerProfileDatabase();
  const profile = getOrCreatePlayerProfile(database, "pgn-name:alice", "Alice");
  const style = new PlayerStyleAccumulator();
  for (let index = 0; index < 30; index += 1) {
    style.observe({
      piece: index % 2 ? "n" : "p",
      from: index % 2 ? "b1" : "e2",
      to: index % 2 ? "c3" : "e4",
      flags: index % 3 === 0 ? "c" : "b",
      san: index % 3 === 0 ? "Nxe4+" : "e4",
      offersUnansweredExposure: index === 8,
    }, "w");
    if (index === 8) style.observeAcceptedExposure();
  }
  profile.games = 2;
  profile.moves = 30;
  profile.styleState = style.serialize();
  return profile;
}

describe("player tendency summaries", () => {
  it("derives observable rates from sufficient history", () => {
    const summary = buildPlayerPersonality(createProfile());
    expect(summary.eligible).toBe(true);
    expect(summary.tendencies.captureFrequency.value).toBeCloseTo(1 / 3);
    expect(summary.tendencies.forcingPlayFrequency.value).toBeCloseTo(1 / 3);
    expect(summary.tendencies.quietMoveFrequency.value).toBeCloseTo(2 / 3);
    expect(summary.tendencies.unansweredExposureFrequency.value).toBeCloseTo(1 / 30);
    expect(summary.tendencies.acceptedExposureFrequency.value).toBeCloseTo(1);
  });

  it("does not invent values for short or legacy histories", () => {
    const profile = createProfile();
    profile.moves = 10;
    profile.styleState.moves = 10;
    const short = buildPlayerPersonality(profile);
    expect(short.eligible).toBe(false);
    expect(short.tendencies.captureFrequency.value).toBeNull();

    const legacy = createProfile();
    legacy.styleState.contextObservedMoves = 0;
    legacy.styleState.quietMoves = 0;
    legacy.styleState.unansweredExposureOpportunities = 0;
    legacy.styleState.acceptedExposureOpportunities = 0;
    const oldSummary = buildPlayerPersonality(legacy);
    expect(oldSummary.tendencies.quietMoveFrequency.value).toBeNull();
    expect(oldSummary.tendencies.unansweredExposureFrequency.value).toBeNull();
  });

  it("creates a stable report with eligible and excluded counts", () => {
    const database = createPlayerProfileDatabase();
    const first = getOrCreatePlayerProfile(database, "pgn-name:alice", "Alice");
    const second = getOrCreatePlayerProfile(database, "pgn-name:bob", "Bob");
    first.games = 1;
    first.moves = 30;
    first.styleState = createProfile().styleState;
    second.games = 1;
    second.moves = 8;
    const report = buildPlayerPersonalityReport(database, {
      generatedAt: "2026-01-01T00:00:00.000Z",
    });
    expect(report.totalPlayers).toBe(2);
    expect(report.eligiblePlayers).toBe(1);
    expect(report.excludedPlayers).toBe(1);
    expect(report.profiles[0]?.id).toBe("pgn-name:alice");
    expect(report.generatedAt).toBe("2026-01-01T00:00:00.000Z");
  });
});
