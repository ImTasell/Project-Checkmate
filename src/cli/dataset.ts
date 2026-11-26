import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { importPgn } from "../data/pgn.js";
import { writeSamplesJsonl } from "../data/jsonl.js";

const [inputArg, outputArg = "data/processed/human-moves.jsonl"] = process.argv.slice(2);
if (!inputArg) {
  console.error("Usage: npm run pgn:dataset -- <input.pgn> [output.jsonl]");
  process.exitCode = 2;
} else {
  const input = resolve(inputArg);
  const output = resolve(outputArg);
  const pgn = await readFile(input, "utf8");
  const { samples, stats } = importPgn(pgn);
  const written = await writeSamplesJsonl(output, samples);
  console.log(`Imported ${stats.games} games; ${stats.positions} positions; skipped ${stats.skippedGames} games.`);
  console.log(`Wrote ${written} move samples to ${output}`);
}
