import { Chess } from "chess.js";
import { encodeMove } from "../chess/position.js";
import { PlayerStyleAccumulator } from "./style.js";

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
  /** Player's behavior fingerprint from their earlier moves in this game. */
  styleFeatures?: number[];
  ply: number;
}

export interface PgnImportStats {
  games: number;
  positions: number;
  skippedGames: number;
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

export function importPgn(pgn: string): { samples: HumanMoveSample[]; stats: PgnImportStats } {
  const samples: HumanMoveSample[] = [];
  const stats: PgnImportStats = { games: 0, positions: 0, skippedGames: 0 };

  for (const gameText of splitPgnGames(pgn)) {
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
      const whiteStyle = new PlayerStyleAccumulator();
      const blackStyle = new PlayerStyleAccumulator();

      stats.games += 1;
      for (let ply = 0; ply < moves.length; ply += 1) {
        const move = moves[ply];
        if (!move) continue;
        const moveUci = `${move.from}${move.to}${move.promotion ?? ""}`;
        const playerElo = move.color === "w" ? whiteElo : blackElo;
        const opponentElo = move.color === "w" ? blackElo : whiteElo;
        const playerStyle = move.color === "w" ? whiteStyle : blackStyle;
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
    } catch {
      stats.skippedGames += 1;
    }
  }

  return { samples, stats };
}
