import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createPlayerProfileDatabase,
  loadPlayerProfileDatabase,
  normalizePlayerId,
  savePlayerProfileDatabase,
} from "../src/data/players.js";
import { PlayerStyleAccumulator } from "../src/data/style.js";

let temporaryDirectory = "";

afterEach(async () => {
  if (temporaryDirectory) {
    await rm(temporaryDirectory, { recursive: true, force: true });
    temporaryDirectory = "";
  }
});

describe("persistent player profiles", () => {
  it("normalizes names and excludes PGN placeholders", () => {
    expect(normalizePlayerId("  Alice   Example ")).toBe("pgn-name:alice example");
    expect(normalizePlayerId("ALICE EXAMPLE")).toBe("pgn-name:alice example");
    expect(normalizePlayerId("?")).toBeUndefined();
    expect(normalizePlayerId("Anonymous")).toBeUndefined();
  });

  it("round-trips exact style counters through an atomic JSON database", async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), "checkmate-profiles-"));
    const path = join(temporaryDirectory, "nested", "profiles.json");
    const database = createPlayerProfileDatabase();
    const style = new PlayerStyleAccumulator();
    style.observe({ piece: "p", from: "e2", to: "e4", flags: "b", san: "e4" }, "w");
    style.observe({ piece: "p", from: "d7", to: "d5", flags: "b", san: "d5" }, "b");

    database.players["pgn-name:alice"] = {
      id: "pgn-name:alice",
      displayName: "Alice",
      games: 1,
      moves: 1,
      elo: 1500,
      lastSeen: "2026-01-01",
      styleState: style.serialize(),
    };
    database.processedGameIds.push("game-hash");

    await savePlayerProfileDatabase(path, database);
    const restored = await loadPlayerProfileDatabase(path);
    expect(restored).toEqual(database);
    expect(JSON.parse(await readFile(path, "utf8")).version).toBe(1);
  });

  it("starts an empty database when the profile file does not exist", async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), "checkmate-profiles-"));
    const restored = await loadPlayerProfileDatabase(join(temporaryDirectory, "missing.json"));
    expect(restored).toEqual(createPlayerProfileDatabase());
  });
});
