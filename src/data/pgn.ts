import { createHash } from "node:crypto";
import { Chess } from "chess.js";
import { encodeMove } from "../chess/position.js";
import { PlayerStyleAccumulator } from "./style.js";
import {
  createPlayerProfileDatabase,
  getOrCreatePlayerProfile,
  normalizePlayerId,
  type PlayerProfileDatabase,
  type PlayerProfileRecord,
} from "./players.js";

export interface HumanMoveSample {
  fen: string;
  moveUci: string;
  moveIndex: number;
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  whiteElo?: number;
  blackElo?: number;
  /** Elo of the player who made this specific move. */
  playerElo?: number;
  /** Elo of the opponent faced by the player who made this move. */
  opponentElo?: number;
  /** Stable name-derived player identity, when the PGN has a real player name. */
  playerId?: string;
  opponentId?: string;
  /** Player's accumulated behavior before this move, including prior games. */
  styleFeatures?: number[];
  /** ISO calendar date from the PGN Date header, when available. */
  gameDate?: string;
  ply: number;
}

export interface PgnImportStats {
  games: number;
  positions: number;
  skippedGames: number;
  duplicateGames: number;
  outOfOrderGames: number;
}

function splitPgnGames(pgn: string): string[] {
  const normalized = pgn.replace(/^\uFEFF/, "").trim();
  if (!normalized) return [];
  return normalized.split(/(?=^\[Event\s+)/m).map((part) => part.trim()).filter(Boolean);
}

function parseElo(value: string | undefined): number | undefined {
  if (!value || value === "?") return undefined;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function parseGameDate(value: string | undefined): string | undefined {
  if (!value || !/^\d{4}\.\d{2}\.\d{2}$/.test(value)) return undefined;
  const [yearText, monthText, dayText] = value.split(".");
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
    return undefined;
  }
  return `${yearText}-${monthText}-${dayText}`;
}

function extractDateTag(gameText: string): string | undefined {
  const match = gameText.match(/^\[Date\s+"([^"]+)"\]/m);
  return parseGameDate(match?.[1]);
}

function orderGames(gameTexts: string[]): string[] {
  if (gameTexts.length < 2 || !gameTexts.every((game) => extractDateTag(game) !== undefined)) {
    return gameTexts;
  }
  return gameTexts
    .map((game, index) => ({ game, date: extractDateTag(game) as string, index }))
    .sort((a, b) => a.date.localeCompare(b.date) || a.index - b.index)
    .map(({ game }) => game);
}

function gameFingerprint(
  headers: Record<string, string>,
  whiteId: string | undefined,
  blackId: string | undefined,
  moves: Array<{ from: string; to: string; promotion?: string }>,
): string {
  const identity = [
    whiteId ?? headers.White ?? "",
    blackId ?? headers.Black ?? "",
    headers.Date ?? "",
    headers.Event ?? "",
    headers.Site ?? "",
    headers.Round ?? "",
    headers.Result ?? "",
    moves.map((move) => `${move.from}${move.to}${move.promotion ?? ""}`).join(" "),
  ].join("\u0000");
  return createHash("sha256").update(identity).digest("hex");
}

function updateProfile(
  record: PlayerProfileRecord | undefined,
  style: PlayerStyleAccumulator,
  elo: number | undefined,
  gameDate: string | undefined,
): void {
  if (!record) return;
  record.games += 1;
  record.styleState = style.serialize();
  record.moves = record.styleState.moves;
  if (elo !== undefined) record.elo = elo;
  if (gameDate !== undefined) record.lastSeen = gameDate;
}

export function importPgn(
  pgn: string,
  database: PlayerProfileDatabase = createPlayerProfileDatabase(),
): { samples: HumanMoveSample[]; stats: PgnImportStats; profiles: PlayerProfileDatabase } {
  const samples: HumanMoveSample[] = [];
  const stats: PgnImportStats = {
    games: 0,
    positions: 0,
    skippedGames: 0,
    duplicateGames: 0,
    outOfOrderGames: 0,
  };
  const processedGameIds = new Set(database.processedGameIds);

  for (const gameText of orderGames(splitPgnGames(pgn))) {
    const chess = new Chess();
    try {
      chess.loadPgn(gameText, { strict: false });
      const moves = chess.history({ verbose: true });
      const headers = chess.getHeaders();
      const result = headers.Result;
      const normalizedResult: HumanMoveSample["result"] =
        result === "1-0" || result === "0-1" || result === "1/2-1/2" ? result : "*";
      const whiteElo = parseElo(headers.WhiteElo);
      const blackElo = parseElo(headers.BlackElo);
      const whiteName = headers.White?.trim();
      const blackName = headers.Black?.trim();
      const whiteId = normalizePlayerId(whiteName);
      const blackId = normalizePlayerId(blackName);
      const gameDate = parseGameDate(headers.Date);

      stats.games += 1;
      const gameId = gameFingerprint(headers, whiteId, blackId, moves);
      if (processedGameIds.has(gameId)) {
        stats.duplicateGames += 1;
        continue;
      }

      const existingProfiles = [whiteId ? database.players[whiteId] : undefined, blackId ? database.players[blackId] : undefined];
      if (gameDate && existingProfiles.some((profile) => profile?.lastSeen && gameDate < profile.lastSeen)) {
        stats.skippedGames += 1;
        stats.outOfOrderGames += 1;
        continue;
      }

      const whiteRecord = whiteId && whiteName
        ? getOrCreatePlayerProfile(database, whiteId, whiteName)
        : undefined;
      const blackRecord = blackId && blackName
        ? getOrCreatePlayerProfile(database, blackId, blackName)
        : undefined;
      const whiteStyle = new PlayerStyleAccumulator(whiteRecord?.styleState);
      const blackStyle = new PlayerStyleAccumulator(blackRecord?.styleState);

      for (let ply = 0; ply < moves.length; ply += 1) {
        const move = moves[ply];
        if (!move) continue;
        const moveUci = `${move.from}${move.to}${move.promotion ?? ""}`;
        const isWhite = move.color === "w";
        const playerElo = isWhite ? whiteElo : blackElo;
        const opponentElo = isWhite ? blackElo : whiteElo;
        const playerStyle = isWhite ? whiteStyle : blackStyle;
        const playerId = isWhite ? whiteId : blackId;
        const opponentId = isWhite ? blackId : whiteId;
        const sample: HumanMoveSample = {
          fen: move.before,
          moveUci,
          moveIndex: encodeMove(moveUci),
          result: normalizedResult,
          ply: ply + 1,
          styleFeatures: playerStyle.snapshot(),
        };
        if (whiteElo !== undefined) sample.whiteElo = whiteElo;
        if (blackElo !== undefined) sample.blackElo = blackElo;
        if (playerElo !== undefined) sample.playerElo = playerElo;
        if (opponentElo !== undefined) sample.opponentElo = opponentElo;
        if (playerId !== undefined) sample.playerId = playerId;
        if (opponentId !== undefined) sample.opponentId = opponentId;
        if (gameDate !== undefined) sample.gameDate = gameDate;
        samples.push(sample);

        playerStyle.observe({
          piece: move.piece,
          from: move.from,
          to: move.to,
          flags: move.flags,
          san: move.san,
        }, move.color);
        stats.positions += 1;
      }

      updateProfile(whiteRecord, whiteStyle, whiteElo, gameDate);
      updateProfile(blackRecord, blackStyle, blackElo, gameDate);
      database.processedGameIds.push(gameId);
      processedGameIds.add(gameId);
    } catch {
      stats.skippedGames += 1;
    }
  }

  return { samples, stats, profiles: database };
}
