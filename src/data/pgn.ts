import { Chess } from "chess.js";
import { encodeMove, encodePosition, POSITION_FEATURES } from "../chess/position.js";

export interface HumanMoveSample {
  fen: string;
  moveUci: string;
  moveIndex: number;
  features: number[];
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  whiteElo?: number;
  blackElo?: number;
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

  // A new game begins with its Event tag. PGN files commonly concatenate games.
  const chunks = normalized.split(/(?=^\[Event\s+)/m).map((part) => part.trim()).filter(Boolean);
  return chunks.length ? chunks : [normalized];
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

      stats.games += 1;
      for (let ply = 0; ply < moves.length; ply += 1) {
        const move = moves[ply];
        if (!move) continue;
        const fen = move.before;
        const moveUci = `${move.from}${move.to}${move.promotion ?? ""}`;
        const features = encodePosition(fen);
        if (features.length !== POSITION_FEATURES) {
          throw new Error("Position encoder returned an unexpected feature count.");
        }

        const sample: HumanMoveSample = {
          fen,
          moveUci,
          moveIndex: encodeMove(moveUci),
          features: Array.from(features),
          result: normalizedResult,
          ply: ply + 1,
        };
        if (whiteElo !== undefined) sample.whiteElo = whiteElo;
        if (blackElo !== undefined) sample.blackElo = blackElo;
        samples.push(sample);
        stats.positions += 1;
      }
    } catch {
      stats.skippedGames += 1;
    }
  }

  return { samples, stats };
}
