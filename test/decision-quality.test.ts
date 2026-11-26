import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import {
  DECISION_QUALITY_INPUT_FEATURES,
  DECISION_QUALITY_LABELS,
  encodeDecisionQualitySample,
  encodeMoveContext,
  encodeDecisionQualityTarget,
  evaluateDecisionQuality,
  getDecisionQualityLabel,
  MOVE_CONTEXT_FEATURES,
} from "../src/model/decision-quality.js";
import { MODEL_INPUT_FEATURES } from "../src/model/conditioning.js";
import { TACTICAL_FEATURES } from "../src/chess/tactics.js";
import type { HumanMoveSample } from "../src/data/pgn.js";

const startFen = new Chess().fen();

function sample(overrides: Partial<HumanMoveSample> = {}): HumanMoveSample {
  return {
    fen: startFen,
    moveUci: "e2e4",
    moveIndex: 796,
    result: "*",
    ply: 1,
    ...overrides,
  };
}

describe("human decision-quality learning", () => {
  it("maps explicit annotations to labels and excludes unannotated or generic NAGs", () => {
    expect(getDecisionQualityLabel(sample({ annotationNags: [2] }))).toBe("mistake");
    expect(getDecisionQualityLabel(sample({ annotationNags: [4] }))).toBe("blunder");
    expect(getDecisionQualityLabel(sample({ moveAnnotation: "brilliant" }))).toBe("brilliant");
    expect(getDecisionQualityLabel(sample())).toBeUndefined();
    expect(getDecisionQualityLabel(sample({ annotationNags: [10] }))).toBeUndefined();
  });

  it("encodes only the selected move and tactical context, never its target annotation", () => {
    const features = encodeMoveContext("e7e8q");
    expect(features).toHaveLength(MOVE_CONTEXT_FEATURES);
    expect(features[52]).toBe(1);
    expect(features[64 + 60]).toBe(1);
    expect(features[128]).toBe(1);
    expect(() => encodeMoveContext("invalid")).toThrow();

    const annotated = sample({ moveAnnotation: "blunder", annotationNags: [4] });
    const differentlyAnnotated = sample({ moveAnnotation: "good", annotationNags: [1] });
    expect(Array.from(encodeDecisionQualitySample(annotated))).toEqual(
      Array.from(encodeDecisionQualitySample(differentlyAnnotated)),
    );
    expect(DECISION_QUALITY_INPUT_FEATURES).toBe(
      MODEL_INPUT_FEATURES + MOVE_CONTEXT_FEATURES + TACTICAL_FEATURES,
    );
  });

  it("encodes promotions separately and normal moves without a promotion feature", () => {
    const normal = encodeMoveContext("a2a4");
    const underpromotion = encodeMoveContext("a7a8n");
    expect(normal[8]).toBe(1);
    expect(normal[128 + 3]).toBe(0);
    expect(underpromotion[128 + 3]).toBe(1);
  });

  it("computes accuracy, class precision, recall, F1 and confusion matrix", () => {
    const metrics = evaluateDecisionQuality([0, 0, 3, 5], [0, 3, 3, 5]);
    expect(metrics.samples).toBe(4);
    expect(metrics.accuracy).toBe(0.75);
    expect(metrics.classes[0]?.support).toBe(2);
    expect(metrics.classes[0]?.precision).toBe(1);
    expect(metrics.classes[0]?.recall).toBe(0.5);
    expect(metrics.confusionMatrix[0]?.[3]).toBe(1);
    expect(metrics.macroF1).toBeGreaterThan(0);
    expect(DECISION_QUALITY_LABELS).toContain("blunder");
  });

  it("requires at least two labeled classes to train", () => {
    expect(encodeDecisionQualityTarget(sample({ moveAnnotation: "worst" }))).toBe(6);
  });
});
