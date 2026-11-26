import { createReadStream, createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { createInterface } from "node:readline";
import type { HumanMoveSample } from "./pgn.js";
import { DECISION_FEATURES, type MoveAnnotation } from "./decision.js";
import { STYLE_FEATURES } from "./style.js";

const MOVE_ANNOTATIONS: readonly MoveAnnotation[] = [
  "good", "mistake", "brilliant", "blunder", "interesting", "dubious", "worst", "other",
];

async function writeSamples(path: string, samples: Iterable<HumanMoveSample>, append: boolean): Promise<number> {
  await mkdir(dirname(path), { recursive: true });
  const stream = createWriteStream(path, { encoding: "utf8", flags: append ? "a" : "w" });
  let count = 0;

  try {
    for (const sample of samples) {
      if (!stream.write(`${JSON.stringify(sample)}\n`)) {
        await new Promise<void>((resolve) => stream.once("drain", resolve));
      }
      count += 1;
    }
  } finally {
    stream.end();
    await new Promise<void>((resolve, reject) => {
      stream.once("finish", resolve);
      stream.once("error", reject);
    });
  }
  return count;
}

export function writeSamplesJsonl(path: string, samples: Iterable<HumanMoveSample>): Promise<number> {
  return writeSamples(path, samples, false);
}

export function appendSamplesJsonl(path: string, samples: Iterable<HumanMoveSample>): Promise<number> {
  return writeSamples(path, samples, true);
}

export async function* readSamplesJsonl(path: string): AsyncGenerator<HumanMoveSample> {
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let lineNumber = 0;

  for await (const line of lines) {
    lineNumber += 1;
    if (!line.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      throw new Error(`Invalid JSON on line ${lineNumber} of ${path}`);
    }
    if (!isSample(value)) throw new Error(`Invalid move sample on line ${lineNumber} of ${path}`);
    yield value;
  }
}

function isSample(value: unknown): value is HumanMoveSample {
  if (!value || typeof value !== "object") return false;
  const sample = value as Partial<HumanMoveSample>;
  return typeof sample.fen === "string"
    && typeof sample.moveUci === "string"
    && Number.isInteger(sample.moveIndex)
    && typeof sample.result === "string"
    && Number.isInteger(sample.ply)
    && (sample.playerElo === undefined || (typeof sample.playerElo === "number" && Number.isFinite(sample.playerElo) && sample.playerElo > 0))
    && (sample.opponentElo === undefined || (typeof sample.opponentElo === "number" && Number.isFinite(sample.opponentElo) && sample.opponentElo > 0))
    && (sample.playerId === undefined || typeof sample.playerId === "string")
    && (sample.opponentId === undefined || typeof sample.opponentId === "string")
    && (sample.gameDate === undefined || typeof sample.gameDate === "string")
    && (sample.styleFeatures === undefined || (
      Array.isArray(sample.styleFeatures)
      && sample.styleFeatures.length === STYLE_FEATURES
      && sample.styleFeatures.every((feature) => typeof feature === "number" && Number.isFinite(feature) && feature >= 0 && feature <= 1)
    ))
    && (sample.decisionFeatures === undefined || (
      Array.isArray(sample.decisionFeatures)
      && sample.decisionFeatures.length === DECISION_FEATURES
      && sample.decisionFeatures.every((feature) => typeof feature === "number" && Number.isFinite(feature) && feature >= 0 && feature <= 1)
    ))
    && (sample.annotationNags === undefined || (
      Array.isArray(sample.annotationNags)
      && sample.annotationNags.every((nag) => Number.isInteger(nag) && nag > 0)
    ))
    && (sample.moveAnnotation === undefined || MOVE_ANNOTATIONS.includes(sample.moveAnnotation))
    && (sample.elapsedSeconds === undefined || (typeof sample.elapsedSeconds === "number" && Number.isFinite(sample.elapsedSeconds) && sample.elapsedSeconds >= 0))
    && (sample.clockRemainingSeconds === undefined || (typeof sample.clockRemainingSeconds === "number" && Number.isFinite(sample.clockRemainingSeconds) && sample.clockRemainingSeconds >= 0));
}
