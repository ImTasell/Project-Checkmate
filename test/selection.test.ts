import * as tf from "@tensorflow/tfjs";
import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import { rankLegalMovesByDecisionQuality } from "../src/model/decision-quality.js";
import {
  chooseHumanLikeMove,
  rankHumanLikeMoves,
  rankPositionForHumanSelection,
  sampleHumanMove,
  type PolicyMoveScore,
} from "../src/model/selection.js";
import { DECISION_QUALITY_LABELS } from "../src/model/decision-quality.js";

const policyScores: PolicyMoveScore[] = [
  { move: "e2e4", score: 0.7 },
  { move: "d2d4", score: 0.2 },
  { move: "g1f3", score: 0.1 },
];

const qualityScores = [
  {
    move: "e2e4",
    predictedLabel: "mistake" as const,
    confidence: 0.8,
    positiveAnnotationProbability: 0.1,
    interestingAnnotationProbability: 0.05,
    criticalAnnotationProbability: 0.8,
    probabilities: [],
  },
  {
    move: "d2d4",
    predictedLabel: "good" as const,
    confidence: 0.8,
    positiveAnnotationProbability: 0.8,
    interestingAnnotationProbability: 0.1,
    criticalAnnotationProbability: 0.05,
    probabilities: [],
  },
  {
    move: "g1f3",
    predictedLabel: "interesting" as const,
    confidence: 0.7,
    positiveAnnotationProbability: 0.3,
    interestingAnnotationProbability: 0.7,
    criticalAnnotationProbability: 0.05,
    probabilities: [],
  },
];

describe("human-like move selection", () => {
  it("preserves normalized policy probabilities when quality weight is zero", () => {
    const ranked = rankHumanLikeMoves(policyScores, qualityScores);
    expect(ranked[0]?.move).toBe("e2e4");
    expect(ranked[0]?.policyProbability).toBeCloseTo(0.7);
    expect(ranked[0]?.selectionProbability).toBeCloseTo(0.7);
    expect(ranked.reduce((sum, item) => sum + item.selectionProbability, 0)).toBeCloseTo(1);
  });

  it("allows the advisory quality signal to move candidates without replacing policy preferences", () => {
    const policyOnly = rankHumanLikeMoves(policyScores, qualityScores, { qualityWeight: 0 });
    const blended = rankHumanLikeMoves(policyScores, qualityScores, { qualityWeight: 1.5 });
    expect(policyOnly[0]?.move).toBe("e2e4");
    expect(blended[0]?.move).toBe("d2d4");
    expect(blended.find((item) => item.move === "e2e4")?.policyProbability).toBeCloseTo(0.7);
  });

  it("uses temperature to control sampling diversity", () => {
    const cool = rankHumanLikeMoves(policyScores, undefined, { temperature: 0.5 });
    const warm = rankHumanLikeMoves(policyScores, undefined, { temperature: 2 });
    expect(cool[0]?.selectionProbability).toBeGreaterThan(warm[0]?.selectionProbability ?? 1);
    expect(cool.reduce((sum, item) => sum + item.selectionProbability, 0)).toBeCloseTo(1);
    expect(warm.reduce((sum, item) => sum + item.selectionProbability, 0)).toBeCloseTo(1);
  });

  it("samples deterministically from the injected random source", () => {
    const ranked = rankHumanLikeMoves(policyScores);
    expect(sampleHumanMove(ranked, () => 0)?.move).toBe("e2e4");
    expect(sampleHumanMove(ranked, () => 0.99)?.move).toBe("g1f3");
    expect(sampleHumanMove([], () => 0.5)).toBeUndefined();
    expect(() => sampleHumanMove(ranked, () => 1)).toThrow();
  });

  it("requires quality candidates when blending and validates candidate coverage", () => {
    expect(() => rankHumanLikeMoves(policyScores, undefined, { qualityWeight: 0.2 })).toThrow();
    expect(() => rankHumanLikeMoves(policyScores, qualityScores.slice(1), { qualityWeight: 1 })).toThrow();
    expect(() => rankHumanLikeMoves(policyScores, qualityScores, { temperature: 0 })).toThrow();
    expect(() => rankHumanLikeMoves(policyScores, qualityScores, { qualityWeight: -1 })).toThrow();
  });

  it("constructs all legal candidates from a position and selects a legal move", () => {
    const fen = new Chess().fen();
    const policyModel = {
      predict: (input: tf.Tensor) => {
        const rows = input.shape[0] ?? 0;
        const output = new Float32Array(rows * 4096);
        const uniform = 1 / 4096;
        output.fill(uniform);
        return tf.tensor2d(output, [rows, 4096]);
      },
    } as unknown as tf.LayersModel;

    const ranked = rankPositionForHumanSelection(policyModel, { fen });
    expect(ranked).toHaveLength(20);
    expect(ranked.reduce((sum, item) => sum + item.selectionProbability, 0)).toBeCloseTo(1);

    const choice = chooseHumanLikeMove(policyModel, { fen }, { random: () => 0.4 });
    expect(new Chess(fen).moves().some((san) => {
      const board = new Chess(fen);
      const move = board.move(san);
      return move.from + move.to + (move.promotion ?? "") === choice?.move;
    })).toBe(true);
  });

  it("does not call the policy for terminal positions", () => {
    const policyModel = {
      predict: () => {
        throw new Error("Policy should not run for a terminal position.");
      },
    } as unknown as tf.LayersModel;
    expect(rankPositionForHumanSelection(policyModel, {
      fen: "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
    })).toEqual([]);
  });
});
