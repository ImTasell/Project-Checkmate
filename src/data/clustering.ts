import { PlayerStyleAccumulator, STYLE_FEATURE_NAMES } from "./style.js";
import type { PlayerProfileDatabase, PlayerProfileRecord } from "./players.js";

export const BEHAVIOR_FEATURE_NAMES = STYLE_FEATURE_NAMES.slice(0, 8);
export type BehaviorFeatureName = typeof BEHAVIOR_FEATURE_NAMES[number];
export type BehaviorCentroid = Record<BehaviorFeatureName, number>;

export interface StyleArchetype {
  id: string;
  label: string;
  size: number;
  centroid: BehaviorCentroid;
  memberIds: string[];
}

export interface PlayerClusteringOptions {
  /** Require enough observed moves before a player can influence clustering. */
  minMoves?: number;
  /** Do not publish archetypes from a tiny player population. */
  minPlayers?: number;
  /** Maximum number of clusters considered when selecting k automatically. */
  maxClusters?: number;
  /** Force a cluster count instead of selecting it with a silhouette score. */
  clusters?: number;
  seed?: number;
}

export interface PlayerClusteringResult {
  algorithm: "standardized-k-means";
  minMoves: number;
  eligiblePlayers: number;
  excludedPlayers: number;
  selectedClusters: number;
  silhouetteScore: number | null;
  reason?: string;
  archetypes: StyleArchetype[];
  assignments: Record<string, string>;
}

interface PlayerVector {
  id: string;
  displayName: string;
  raw: number[];
  normalized: number[];
}

interface Standardization {
  means: number[];
  deviations: number[];
}

interface KMeansResult {
  assignments: number[];
  centers: number[][];
}

function squaredDistance(a: readonly number[], b: readonly number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    const delta = (a[index] ?? 0) - (b[index] ?? 0);
    sum += delta * delta;
  }
  return sum;
}

function standardize(rows: PlayerVector[]): Standardization {
  const dimensions = BEHAVIOR_FEATURE_NAMES.length;
  const means = Array.from({ length: dimensions }, (_, dimension) =>
    rows.reduce((sum, row) => sum + (row.raw[dimension] ?? 0), 0) / rows.length,
  );
  const deviations = means.map((mean, dimension) => {
    const variance = rows.reduce((sum, row) => {
      const delta = (row.raw[dimension] ?? 0) - mean;
      return sum + delta * delta;
    }, 0) / rows.length;
    const deviation = Math.sqrt(variance);
    return deviation > 1e-9 ? deviation : 1;
  });

  for (const row of rows) {
    row.normalized = row.raw.map((value, index) => (value - (means[index] ?? 0)) / (deviations[index] ?? 1));
  }
  return { means, deviations };
}

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function initializeCenters(rows: PlayerVector[], k: number, random: () => number): number[][] {
  const centers: number[][] = [];
  const first = Math.floor(random() * rows.length);
  centers.push([...(rows[first]?.normalized ?? [])]);

  while (centers.length < k) {
    const distances = rows.map((row) =>
      Math.min(...centers.map((center) => squaredDistance(row.normalized, center))),
    );
    const total = distances.reduce((sum, distance) => sum + distance, 0);

    if (total <= 1e-12) {
      const used = new Set(centers.map((center) => rows.findIndex((row) =>
        row.normalized.every((value, index) => Math.abs(value - (center[index] ?? 0)) < 1e-9),
      )));
      const next = rows.findIndex((_, index) => !used.has(index));
      if (next < 0) break;
      centers.push([...(rows[next]?.normalized ?? [])]);
      continue;
    }

    let threshold = random() * total;
    let selected = distances.length - 1;
    for (let index = 0; index < distances.length; index += 1) {
      threshold -= distances[index] ?? 0;
      if (threshold <= 0) {
        selected = index;
        break;
      }
    }
    centers.push([...(rows[selected]?.normalized ?? [])]);
  }
  return centers;
}

function assignRows(rows: PlayerVector[], centers: number[][]): number[] {
  return rows.map((row) => {
    let bestCluster = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let cluster = 0; cluster < centers.length; cluster += 1) {
      const distance = squaredDistance(row.normalized, centers[cluster] ?? []);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestCluster = cluster;
      }
    }
    return bestCluster;
  });
}

function runKMeans(rows: PlayerVector[], k: number, seed: number): KMeansResult {
  const random = makeRandom(seed);
  const centers = initializeCenters(rows, k, random);
  const actualK = centers.length;
  let assignments = assignRows(rows, centers);

  for (let iteration = 0; iteration < 100; iteration += 1) {
    const sums = Array.from({ length: actualK }, () => Array(BEHAVIOR_FEATURE_NAMES.length).fill(0) as number[]);
    const counts = Array(actualK).fill(0) as number[];

    rows.forEach((row, rowIndex) => {
      const cluster = assignments[rowIndex] ?? 0;
      counts[cluster] = (counts[cluster] ?? 0) + 1;
      row.normalized.forEach((value, dimension) => {
        sums[cluster]![dimension] = (sums[cluster]![dimension] ?? 0) + value;
      });
    });

    let changed = false;
    for (let cluster = 0; cluster < actualK; cluster += 1) {
      const count = counts[cluster] ?? 0;
      if (count === 0) {
        const farthest = rows
          .map((row) => Math.min(...centers.map((center) => squaredDistance(row.normalized, center))))
          .reduce((best, distance, index, all) => distance > (all[best] ?? -1) ? index : best, 0);
        const replacement = rows[farthest]?.normalized;
        if (replacement) centers[cluster] = [...replacement];
        changed = true;
        continue;
      }
      const nextCenter = (sums[cluster] ?? []).map((sum) => sum / count);
      if (squaredDistance(nextCenter, centers[cluster] ?? []) > 1e-10) changed = true;
      centers[cluster] = nextCenter;
    }

    const nextAssignments = assignRows(rows, centers);
    if (!changed && nextAssignments.every((value, index) => value === assignments[index])) {
      assignments = nextAssignments;
      break;
    }
    assignments = nextAssignments;
  }

  return { assignments, centers };
}

function approximateSilhouette(rows: PlayerVector[], assignments: number[], k: number): number {
  if (rows.length < 3 || k < 2) return -1;

  // Deterministic evenly spaced sample keeps model selection bounded on large databases.
  const sampleSize = Math.min(rows.length, 300);
  const indices = Array.from({ length: sampleSize }, (_, index) =>
    Math.floor((index * rows.length) / sampleSize),
  );
  const scores: number[] = [];

  for (const index of indices) {
    const ownCluster = assignments[index] ?? 0;
    let ownTotal = 0;
    let ownCount = 0;
    const otherTotals = Array(k).fill(0) as number[];
    const otherCounts = Array(k).fill(0) as number[];

    for (const otherIndex of indices) {
      if (otherIndex === index) continue;
      const cluster = assignments[otherIndex] ?? 0;
      const distance = Math.sqrt(squaredDistance(
        rows[index]?.normalized ?? [],
        rows[otherIndex]?.normalized ?? [],
      ));
      if (cluster === ownCluster) {
        ownTotal += distance;
        ownCount += 1;
      } else {
        otherTotals[cluster] = (otherTotals[cluster] ?? 0) + distance;
        otherCounts[cluster] = (otherCounts[cluster] ?? 0) + 1;
      }
    }

    if (ownCount === 0) {
      scores.push(0);
      continue;
    }
    const a = ownTotal / ownCount;
    let b = Number.POSITIVE_INFINITY;
    for (let cluster = 0; cluster < k; cluster += 1) {
      if (cluster === ownCluster || (otherCounts[cluster] ?? 0) === 0) continue;
      b = Math.min(b, (otherTotals[cluster] ?? 0) / (otherCounts[cluster] ?? 1));
    }
    scores.push(Number.isFinite(b) ? (b - a) / Math.max(a, b, 1e-12) : 0);
  }
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

function uniqueBehaviorCount(rows: PlayerVector[]): number {
  const unique = new Set(rows.map((row) => row.normalized.map((value) => value.toFixed(7)).join(",")));
  return unique.size;
}

function selectClusterCount(
  rows: PlayerVector[],
  options: PlayerClusteringOptions,
): { k: number; score: number | null; fitted: KMeansResult } {
  const maxClusters = Math.min(
    Math.max(2, Math.trunc(options.maxClusters ?? 6)),
    Math.floor(Math.sqrt(rows.length)),
    uniqueBehaviorCount(rows),
    rows.length - 1,
  );

  if (options.clusters !== undefined) {
    if (!Number.isInteger(options.clusters) || options.clusters < 2 || options.clusters > maxClusters) {
      throw new Error(`Requested ${options.clusters} clusters, but this dataset supports 2-${maxClusters} distinct clusters.`);
    }
    const fitted = runKMeans(rows, options.clusters, options.seed ?? 42);
    return {
      k: fitted.centers.length,
      score: approximateSilhouette(rows, fitted.assignments, fitted.centers.length),
      fitted,
    };
  }

  if (maxClusters < 2) {
    throw new Error("Not enough distinct player behavior profiles to form at least two clusters.");
  }

  let best: { k: number; score: number; fitted: KMeansResult } | undefined;
  for (let k = 2; k <= maxClusters; k += 1) {
    const fitted = runKMeans(rows, k, (options.seed ?? 42) + k * 997);
    const score = approximateSilhouette(rows, fitted.assignments, fitted.centers.length);
    if (!best || score > best.score + 1e-9 || (Math.abs(score - best.score) <= 1e-9 && k < best.k)) {
      best = { k: fitted.centers.length, score, fitted };
    }
  }
  if (!best) throw new Error("Unable to select a cluster count.");
  return { ...best };
}

function describeCluster(center: number[], global: Standardization): string {
  const descriptors: Array<{ deviation: number; text: string }> = [
    { deviation: center[0] ?? 0, text: (center[0] ?? 0) >= 0 ? "capture-oriented" : "capture-selective" },
    { deviation: center[1] ?? 0, text: (center[1] ?? 0) >= 0 ? "check-seeking" : "check-conservative" },
    { deviation: center[2] ?? 0, text: (center[2] ?? 0) >= 0 ? "pawn-forward" : "pawn-reserving" },
    { deviation: center[3] ?? 0, text: (center[3] ?? 0) >= 0 ? "centralizing" : "edge-preferring" },
    { deviation: center[4] ?? 0, text: (center[4] ?? 0) >= 0 ? "early-queen-active" : "queen-conserving" },
    { deviation: center[5] ?? 0, text: (center[5] ?? 0) >= 0 ? "castle-prone" : "castle-reluctant" },
    { deviation: center[6] ?? 0, text: (center[6] ?? 0) >= 0 ? "king-mobile" : "king-stable" },
    { deviation: center[7] ?? 0, text: (center[7] ?? 0) >= 0 ? "piece-active" : "piece-conserving" },
  ];
  const selected = descriptors
    .map((item) => ({ ...item, deviation: Math.abs(item.deviation) }))
    .filter((item) => item.deviation >= 0.55)
    .sort((a, b) => b.deviation - a.deviation)
    .slice(0, 2);
  void global;
  return selected.length ? selected.map((item) => item.text).join(" · ") : "balanced / mixed tendencies";
}

function makeRawCentroid(rows: PlayerVector[], memberIndexes: number[]): BehaviorCentroid {
  const centroid = {} as BehaviorCentroid;
  BEHAVIOR_FEATURE_NAMES.forEach((name, dimension) => {
    centroid[name] = memberIndexes.reduce((sum, index) =>
      sum + (rows[index]?.raw[dimension] ?? 0), 0,
    ) / memberIndexes.length;
  });
  return centroid;
}

function vectorizeProfile(profile: PlayerProfileRecord): PlayerVector {
  const raw = new PlayerStyleAccumulator(profile.styleState).snapshot().slice(0, BEHAVIOR_FEATURE_NAMES.length);
  return { id: profile.id, displayName: profile.displayName, raw, normalized: [] };
}

/**
 * Groups sufficiently observed human players by standardized behavior.
 * Confidence is used only to filter low-history players; it is not a style
 * dimension, so the number of imported moves cannot itself define an archetype.
 */
export function clusterPlayerProfiles(
  database: PlayerProfileDatabase,
  options: PlayerClusteringOptions = {},
): PlayerClusteringResult {
  const minMoves = options.minMoves ?? 24;
  const minPlayers = options.minPlayers ?? 6;
  if (!Number.isInteger(minMoves) || minMoves < 1) throw new Error("minMoves must be a positive integer.");
  if (!Number.isInteger(minPlayers) || minPlayers < 3) throw new Error("minPlayers must be an integer of at least 3.");

  const allProfiles = Object.values(database.players);
  const rows = allProfiles
    .filter((profile) => profile.moves >= minMoves)
    .sort((a, b) => a.id.localeCompare(b.id))
    .map(vectorizeProfile);
  const base: PlayerClusteringResult = {
    algorithm: "standardized-k-means",
    minMoves,
    eligiblePlayers: rows.length,
    excludedPlayers: allProfiles.length - rows.length,
    selectedClusters: 0,
    silhouetteScore: null,
    archetypes: [],
    assignments: {},
  };

  if (rows.length < minPlayers) {
    base.reason = `Need at least ${minPlayers} players with ${minMoves}+ moves; found ${rows.length}.`;
    return base;
  }

  const global = standardize(rows);
  let selected: { k: number; score: number | null; fitted: KMeansResult };
  try {
    selected = selectClusterCount(rows, options);
  } catch (error) {
    base.reason = error instanceof Error ? error.message : "Unable to cluster player profiles.";
    return base;
  }

  const groups = Array.from({ length: selected.k }, () => [] as number[]);
  selected.fitted.assignments.forEach((cluster, index) => groups[cluster]?.push(index));

  const nonEmptyGroups = groups
    .map((members) => members)
    .filter((members) => members.length > 0)
    .sort((a, b) => {
      const aId = rows[a[0] ?? 0]?.id ?? "";
      const bId = rows[b[0] ?? 0]?.id ?? "";
      return aId.localeCompare(bId);
    });

  const archetypes = nonEmptyGroups.map((members, index): StyleArchetype => {
    const center = selected.fitted.centers[
      selected.fitted.assignments[members[0] ?? 0] ?? 0
    ] ?? [];
    const id = `style-${index + 1}`;
    const label = describeCluster(center, global);
    for (const memberIndex of members) {
      const playerId = rows[memberIndex]?.id;
      if (playerId) base.assignments[playerId] = id;
    }
    return {
      id,
      label,
      size: members.length,
      centroid: makeRawCentroid(rows, members),
      memberIds: members.map((memberIndex) => rows[memberIndex]?.id ?? ""),
    };
  });

  base.selectedClusters = archetypes.length;
  base.silhouetteScore = selected.score;
  base.archetypes = archetypes;
  return base;
}
