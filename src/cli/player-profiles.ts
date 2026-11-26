import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { loadPlayerProfileDatabase } from "../data/players.js";
import { buildPlayerPersonalityReport } from "../data/personality.js";

const [
  profilesArg = "data/processed/player-profiles.json",
  outputArg = "data/processed/player-personalities.json",
  minimumMovesArg = "24",
] = process.argv.slice(2);

const minimumMoves = Number(minimumMovesArg);
if (!Number.isInteger(minimumMoves) || minimumMoves < 1) {
  throw new Error("Minimum history must be a positive integer.");
}

const profilesPath = resolve(profilesArg);
const outputPath = resolve(outputArg);
const database = await loadPlayerProfileDatabase(profilesPath);
const report = buildPlayerPersonalityReport(database, { minimumMoves });

await mkdir(dirname(outputPath), { recursive: true });
const temporaryPath = outputPath + "." + process.pid + ".tmp";
await writeFile(temporaryPath, JSON.stringify(report, null, 2) + "\n", "utf8");
await rename(temporaryPath, outputPath);

console.log("Player tendencies: " + report.eligiblePlayers + " eligible, "
  + report.excludedPlayers + " below the " + minimumMoves + "-move minimum.");
console.log("Saved descriptive player profiles to " + outputPath);

for (const profile of report.profiles.filter((item) => item.eligible).slice(0, 10)) {
  const asPercent = (value: number | null) => value === null ? "unknown" : (value * 100).toFixed(1) + "%";
  const traits = profile.tendencies;
  console.log(profile.displayName + " (" + profile.moves + " moves)");
  console.log("  forcing=" + asPercent(traits.forcingPlayFrequency.value)
    + ", quiet=" + asPercent(traits.quietMoveFrequency.value)
    + ", pawn expansion=" + asPercent(traits.pawnExpansionFrequency.value)
    + ", centralization=" + asPercent(traits.centralization.value));
  console.log("  exposure=" + asPercent(traits.unansweredExposureFrequency.value)
    + ", accepted exposure=" + asPercent(traits.acceptedExposureFrequency.value)
    + ", castled=" + (traits.castledAtLeastOnce === null ? "unknown" : String(traits.castledAtLeastOnce)));
}
