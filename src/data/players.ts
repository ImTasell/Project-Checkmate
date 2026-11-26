import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { PlayerStyleAccumulator, type PlayerStyleState } from "./style.js";

export interface PlayerProfileRecord {
  /** Name-derived stable key; external platform IDs can replace this later. */
  id: string;
  displayName: string;
  games: number;
  moves: number;
  elo?: number;
  lastSeen?: string;
  styleState: PlayerStyleState;
}

export interface PlayerProfileDatabase {
  version: 1;
  /** Hashes make repeated imports idempotent. */
  processedGameIds: string[];
  players: Record<string, PlayerProfileRecord>;
}

export function createPlayerProfileDatabase(): PlayerProfileDatabase {
  return { version: 1, processedGameIds: [], players: {} };
}

/**
 * PGN headers generally contain usernames rather than platform account IDs.
 * Normalize case/spacing so the same name maps consistently; placeholders are
 * deliberately excluded instead of merging every anonymous player together.
 */
export function normalizePlayerId(name: string | undefined): string | undefined {
  const displayName = name?.normalize("NFKC").trim().replace(/\s+/g, " ");
  if (!displayName) return undefined;
  const normalized = displayName.toLocaleLowerCase("en-US");
  if (["?", "-", "unknown", "anonymous", "none", "null"].includes(normalized)) return undefined;
  return `pgn-name:${normalized}`;
}

export function getOrCreatePlayerProfile(
  database: PlayerProfileDatabase,
  id: string,
  displayName: string,
): PlayerProfileRecord {
  const existing = database.players[id];
  if (existing) return existing;

  const styleState = new PlayerStyleAccumulator().serialize();
  const created: PlayerProfileRecord = {
    id,
    displayName: displayName.normalize("NFKC").trim().replace(/\s+/g, " "),
    games: 0,
    moves: 0,
    styleState,
  };
  database.players[id] = created;
  return created;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseDatabase(value: unknown, path: string): PlayerProfileDatabase {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.processedGameIds) || !isRecord(value.players)) {
    throw new Error(`Invalid player profile database: ${path}`);
  }

  const database = createPlayerProfileDatabase();
  for (const gameId of value.processedGameIds) {
    if (typeof gameId !== "string") throw new Error(`Invalid processed game ID in ${path}`);
    database.processedGameIds.push(gameId);
  }

  for (const [id, raw] of Object.entries(value.players)) {
    if (!isRecord(raw) || raw.id !== id || typeof raw.displayName !== "string"
      || typeof raw.games !== "number" || !Number.isInteger(raw.games) || raw.games < 0
      || typeof raw.moves !== "number" || !Number.isInteger(raw.moves) || raw.moves < 0
      || !isRecord(raw.styleState)) {
      throw new Error(`Invalid player profile '${id}' in ${path}`);
    }
    const styleState = new PlayerStyleAccumulator(raw.styleState as Partial<PlayerStyleState>).serialize();
    const record: PlayerProfileRecord = {
      id,
      displayName: raw.displayName,
      games: raw.games,
      moves: raw.moves,
      styleState,
    };
    if (typeof raw.elo === "number" && Number.isFinite(raw.elo) && raw.elo > 0) record.elo = raw.elo;
    if (typeof raw.lastSeen === "string") record.lastSeen = raw.lastSeen;
    database.players[id] = record;
  }
  return database;
}

export async function loadPlayerProfileDatabase(path: string): Promise<PlayerProfileDatabase> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT") {
      return createPlayerProfileDatabase();
    }
    throw error;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error(`Invalid JSON in player profile database: ${path}`);
  }
  return parseDatabase(parsed, path);
}

export async function savePlayerProfileDatabase(path: string, database: PlayerProfileDatabase): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.tmp`;
  const content = `${JSON.stringify(database, null, 2)}\n`;
  await writeFile(temporaryPath, content, "utf8");
  await rename(temporaryPath, path);
}
