import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { appendSamplesJsonl, writeSamplesJsonl } from "../src/data/jsonl.js";
import type { HumanMoveSample } from "../src/data/pgn.js";

let temporaryDirectory = "";

afterEach(async () => {
  if (temporaryDirectory) {
    await rm(temporaryDirectory, { recursive: true, force: true });
    temporaryDirectory = "";
  }
});

describe("JSONL dataset writing", () => {
  it("appends separate import batches instead of replacing earlier samples", async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), "checkmate-jsonl-"));
    const path = join(temporaryDirectory, "nested", "moves.jsonl");
    const first: HumanMoveSample = {
      fen: "start",
      moveUci: "e2e4",
      moveIndex: 796,
      result: "*",
      ply: 1,
    };
    const second: HumanMoveSample = {
      fen: "after-e4",
      moveUci: "e7e5",
      moveIndex: 1820,
      result: "*",
      ply: 2,
    };

    expect(await appendSamplesJsonl(path, [first])).toBe(1);
    expect(await appendSamplesJsonl(path, [second])).toBe(1);
    const lines = (await readFile(path, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toEqual([first, second]);
  });

  it("keeps the explicit replace operation available for clean rebuilds", async () => {
    temporaryDirectory = await mkdtemp(join(tmpdir(), "checkmate-jsonl-"));
    const path = join(temporaryDirectory, "moves.jsonl");
    const sample: HumanMoveSample = {
      fen: "start",
      moveUci: "e2e4",
      moveIndex: 796,
      result: "*",
      ply: 1,
    };

    await appendSamplesJsonl(path, [sample, sample]);
    expect(await writeSamplesJsonl(path, [sample])).toBe(1);
    expect((await readFile(path, "utf8")).trim().split("\n")).toHaveLength(1);
  });
});
