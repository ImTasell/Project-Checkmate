import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { importPgn } from "../data/pgn.js";
import { appendSamplesJsonl } from "../data/jsonl.js";
import { loadPlayerProfileDatabase, savePlayerProfileDatabase } from "../data/players.js";

const [
  inputArg,
  outputArg = "data/processed/human-moves.jsonl",
  profilesArg = "data/processed/player-profiles.json",
] = process.argv.slice(2);

if (!inputArg) {
  console.error("Usage: npm run pgn:dataset -- <input.pgn> [output.jsonl] [profiles.json]");
  process.exitCode = 2;
} else {
  const input = resolve(inputArg);
  const output = resolve(outputArg);
  const profilesPath = resolve(profilesArg);
  const pgn = await readFile(input, "utf8");
  const profiles = await loadPlayerProfileDatabase(profilesPath);
  const { samples, stats } = importPgn(pgn, profiles);
  const written = await appendSamplesJsonl(output, samples);
  await savePlayerProfileDatabase(profilesPath, profiles);

  console.log(
    `Parsed ${stats.games} games; appended ${written} positions; ${stats.skippedGames} skipped; ${stats.duplicateGames} duplicates ignored; ${stats.outOfOrderGames} out-of-order games rejected.`,
  );
  console.log(`Saved ${Object.keys(profiles.players).length} persistent player profiles to ${profilesPath}`);
  console.log(`Updated training dataset at ${output}`);
}
