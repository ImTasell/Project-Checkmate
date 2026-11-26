import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { clusterPlayerProfiles } from "../data/clustering.js";
import { loadPlayerProfileDatabase } from "../data/players.js";

const [
  profilesArg = "data/processed/player-profiles.json",
  outputArg = "data/processed/player-archetypes.json",
  clustersArg = "auto",
  minMovesArg = "24",
] = process.argv.slice(2);

const profilesPath = resolve(profilesArg);
const outputPath = resolve(outputArg);
const parsedClusters = clustersArg === "auto" ? undefined : Number(clustersArg);
const minMoves = Number(minMovesArg);

if (clustersArg !== "auto" && (!Number.isInteger(parsedClusters) || (parsedClusters ?? 0) < 2)) {
  throw new Error("Cluster count must be 'auto' or an integer of at least 2.");
}
if (!Number.isInteger(minMoves) || minMoves < 1) {
  throw new Error("Minimum player moves must be a positive integer.");
}

const profiles = await loadPlayerProfileDatabase(profilesPath);
const result = clusterPlayerProfiles(profiles, {
  minMoves,
  ...(parsedClusters === undefined ? {} : { clusters: parsedClusters }),
});

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify({
  generatedAt: new Date().toISOString(),
  ...result,
}, null, 2)}\n`, "utf8");

console.log(`Analyzed ${result.eligiblePlayers} eligible players; excluded ${result.excludedPlayers} low-history profiles.`);
if (result.reason) {
  console.log(result.reason);
} else {
  console.log(`Discovered ${result.selectedClusters} behavioral archetypes (silhouette ${result.silhouetteScore?.toFixed(3) ?? "n/a"}):`);
  for (const archetype of result.archetypes) {
    console.log(`- ${archetype.id}: ${archetype.label} (${archetype.size} players)`);
  }
}
console.log(`Saved player archetypes to ${outputPath}`);
