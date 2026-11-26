import { describe, expect, it } from "vitest";
import {
  classifyMoveAnnotation,
  MoveDecisionContext,
  parsePgnClock,
  parseTimeControlBaseSeconds,
} from "../src/data/decision.js";

describe("human decision context", () => {
  it("parses PGN clock formats and common time controls", () => {
    expect(parsePgnClock("0:04:12.5")).toBeCloseTo(252.5);
    expect(parsePgnClock("04:12")).toBe(252);
    expect(parsePgnClock("not-a-clock")).toBeUndefined();
    expect(parseTimeControlBaseSeconds("300+5")).toBe(300);
    expect(parseTimeControlBaseSeconds("40/7200:3600")).toBe(7200);
    expect(parseTimeControlBaseSeconds("?")).toBeUndefined();
  });

  it("classifies only explicit NAG values", () => {
    expect(classifyMoveAnnotation([4])).toBe("blunder");
    expect(classifyMoveAnnotation([2])).toBe("mistake");
    expect(classifyMoveAnnotation([6])).toBe("dubious");
    expect(classifyMoveAnnotation([3])).toBe("brilliant");
    expect(classifyMoveAnnotation([])).toBeUndefined();
  });

  it("snapshots available clock and prior annotation history before observing the target move", () => {
    const context = new MoveDecisionContext(300);
    expect(context.snapshot()).toEqual([0, 1, 0, 0]);

    context.observe([2], 150);
    expect(context.snapshot()).toEqual([0.5, 1, 1, 1]);

    context.observe([], 120);
    expect(context.snapshot()).toEqual([0.6, 1, 1, 0.5]);
  });

  it("keeps missing clock information explicitly unknown", () => {
    const context = new MoveDecisionContext();
    expect(context.snapshot()).toEqual([0, 0, 0, 0]);
    context.observe([], 20);
    expect(context.snapshot()).toEqual([0, 0, 0, 0]);
  });

  it("does not assume the starting clock is still available after an unreported move", () => {
    const context = new MoveDecisionContext(300);
    expect(context.snapshot()).toEqual([0, 1, 0, 0]);
    context.observe([], undefined);
    expect(context.snapshot()).toEqual([0, 0, 0, 0]);
    context.observe([], 120);
    expect(context.snapshot()).toEqual([0.6, 1, 0, 0]);
  });
});
