import { describe, expect, it } from "vitest";
import { clusterPlayerProfiles } from "../src/data/clustering.js";
import { createPlayerProfileDatabase, type PlayerProfileRecord } from "../src/data/players.js";
import type { PlayerStyleState } from "../src/data/style.js";

function makeProfile(
  id: string,
  group: "tactical" | "positional",
  moves = 100,
): PlayerProfileRecord {
  const tactical = group === "tactical";
  const styleState: PlayerStyleState = {
    moves,
    captures: Math.round(moves * (tactical ? 0.72 : 0.12)),
    checks: Math.round(moves * (tactical ? 0.48 : 0.06)),
    pawnAdvances: Math.round(moves * (tactical ? 0.35 : 0.76)),
    centralityTotal: moves * (tactical ? 0.78 : 0.32),
    centralityMoves: moves,
    earlyQueenMoves: Math.round(moves * (tactical ? 0.22 : 0.03)),
    castled: !tactical,
    kingMoves: Math.round(moves * (tactical ? 0.16 : 0.02)),
    activePieceMoves: Math.round(moves * (tactical ? 0.74 : 0.26)),
  };
  return {
    id,
    displayName: id,
    games: 2,
    moves,
    styleState,
  };
}

describe("data-derived player style archetypes", () => {
  it("separates distinct behavior groups without using sample confidence as a style feature", () => {
    const database = createPlayerProfileDatabase();
    for (let index = 0; index < 8; index += 1) {
      database.players[`tactical-${index}`] = makeProfile(`tactical-${index}`, "tactical", 60 + index * 20);
      database.players[`positional-${index}`] = makeProfile(`positional-${index}`, "positional", 80 + index * 17);
    }

    const result = clusterPlayerProfiles(database, {
      minMoves: 24,
      minPlayers: 6,
      clusters: 2,
      seed: 12,
    });

    expect(result.reason).toBeUndefined();
    expect(result.eligiblePlayers).toBe(16);
    expect(result.selectedClusters).toBe(2);
    expect(result.archetypes.map((cluster) => cluster.size).sort((a, b) => a - b)).toEqual([8, 8]);

    const tacticalAssignments = new Set(Array.from({ length: 8 }, (_, index) =>
      result.assignments[`tactical-${index}`],
    ));
    const positionalAssignments = new Set(Array.from({ length: 8 }, (_, index) =>
      result.assignments[`positional-${index}`],
    ));
    expect(tacticalAssignments.size).toBe(1);
    expect(positionalAssignments.size).toBe(1);
    expect([...tacticalAssignments][0]).not.toBe([...positionalAssignments][0]);
  });

  it("filters low-history players and refuses to invent clusters for a small cohort", () => {
    const database = createPlayerProfileDatabase();
    database.players["experienced-a"] = makeProfile("experienced-a", "tactical", 40);
    database.players["experienced-b"] = makeProfile("experienced-b", "positional", 40);
    database.players["new-player"] = makeProfile("new-player", "tactical", 8);

    const result = clusterPlayerProfiles(database, { minMoves: 24, minPlayers: 6 });
    expect(result.eligiblePlayers).toBe(2);
    expect(result.excludedPlayers).toBe(1);
    expect(result.selectedClusters).toBe(0);
    expect(result.archetypes).toEqual([]);
    expect(result.reason).toContain("Need at least 6 players");
  });

  it("selects the number of groups deterministically when the cluster count is automatic", () => {
    const database = createPlayerProfileDatabase();
    for (let index = 0; index < 8; index += 1) {
      database.players[`tactical-${index}`] = makeProfile(`tactical-${index}`, "tactical");
      database.players[`positional-${index}`] = makeProfile(`positional-${index}`, "positional");
    }

    const first = clusterPlayerProfiles(database, { minMoves: 24, seed: 99 });
    const second = clusterPlayerProfiles(database, { minMoves: 24, seed: 99 });
    expect(first.selectedClusters).toBe(2);
    expect(first.assignments).toEqual(second.assignments);
    expect(first.silhouetteScore).toBeCloseTo(second.silhouetteScore ?? 0, 8);
  });
});
