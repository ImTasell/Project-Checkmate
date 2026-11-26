/**
 * Context available to a human before choosing a move. Current-move elapsed
 * time and annotations are targets/metadata and are only observed afterwards.
 */
export const DECISION_FEATURE_NAMES = [
  "timePressure",
  "timeKnown",
  "priorMistakeRate",
  "annotationCoverage",
] as const;
export const DECISION_FEATURES = DECISION_FEATURE_NAMES.length;

export type MoveAnnotation =
  | "good"
  | "mistake"
  | "brilliant"
  | "blunder"
  | "interesting"
  | "dubious"
  | "worst"
  | "other";

const NEGATIVE_NAGS = new Set([2, 4, 6, 9]);

export function classifyMoveAnnotation(nags: readonly number[]): MoveAnnotation | undefined {
  if (nags.includes(4)) return "blunder";
  if (nags.includes(9)) return "worst";
  if (nags.includes(2)) return "mistake";
  if (nags.includes(6)) return "dubious";
  if (nags.includes(3)) return "brilliant";
  if (nags.includes(1)) return "good";
  if (nags.includes(5)) return "interesting";
  return nags.length ? "other" : undefined;
}

/** Parse PGN clock values such as 0:04:12.5 or 04:12. */
export function parsePgnClock(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parts = value.trim().split(":");
  if (parts.length < 2 || parts.length > 3) return undefined;

  const seconds = Number(parts[parts.length - 1]);
  const minutes = Number(parts[parts.length - 2]);
  const hours = parts.length === 3 ? Number(parts[0]) : 0;
  if (![seconds, minutes, hours].every(Number.isFinite)
    || seconds < 0 || seconds >= 60 || minutes < 0 || minutes >= 60 || hours < 0) {
    return undefined;
  }
  return hours * 3600 + minutes * 60 + seconds;
}

/**
 * Reads the initial base clock from common PGN TimeControl forms:
 * "300+5", "600", or "40/7200:3600". Unknown formats stay unknown.
 */
export function parseTimeControlBaseSeconds(value: string | undefined): number | undefined {
  if (!value || value === "?") return undefined;
  const firstStage = value.split(":")[0]?.trim();
  if (!firstStage) return undefined;

  const movesPerStage = firstStage.match(/^\d+\/(\d+(?:\.\d+)?)$/);
  if (movesPerStage?.[1]) {
    const seconds = Number(movesPerStage[1]);
    return Number.isFinite(seconds) && seconds > 0 ? seconds : undefined;
  }

  const standard = firstStage.match(/^(\d+(?:\.\d+)?)(?:\+\d+(?:\.\d+)?)?$/);
  if (!standard?.[1]) return undefined;
  const seconds = Number(standard[1]);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : undefined;
}

export interface MoveDecisionContextState {
  moves: number;
  annotatedMoves: number;
  negativeAnnotations: number;
  lastClockRemainingSeconds?: number;
}

export class MoveDecisionContext {
  private moves: number;
  private annotatedMoves: number;
  private negativeAnnotations: number;
  private lastClockRemainingSeconds?: number;

  constructor(
    private readonly baseTimeSeconds?: number,
    state?: Partial<MoveDecisionContextState>,
  ) {
    this.moves = Math.max(0, Math.trunc(state?.moves ?? 0));
    this.annotatedMoves = Math.max(0, Math.trunc(state?.annotatedMoves ?? 0));
    this.negativeAnnotations = Math.max(0, Math.trunc(state?.negativeAnnotations ?? 0));
    if (state?.lastClockRemainingSeconds !== undefined
      && Number.isFinite(state.lastClockRemainingSeconds)
      && state.lastClockRemainingSeconds >= 0) {
      this.lastClockRemainingSeconds = state.lastClockRemainingSeconds;
    } else if (baseTimeSeconds !== undefined) {
      this.lastClockRemainingSeconds = baseTimeSeconds;
    }
  }

  /** Four normalized values; take this snapshot before observing the target move. */
  snapshot(): number[] {
    const timeKnown = this.baseTimeSeconds !== undefined
      && this.lastClockRemainingSeconds !== undefined;
    const timePressure = timeKnown
      ? Math.max(0, Math.min(1, 1 - (this.lastClockRemainingSeconds ?? 0) / (this.baseTimeSeconds ?? 1)))
      : 0;

    return [
      timePressure,
      timeKnown ? 1 : 0,
      this.annotatedMoves ? this.negativeAnnotations / this.annotatedMoves : 0,
      this.moves ? this.annotatedMoves / this.moves : 0,
    ];
  }

  observe(nags: readonly number[], clockRemainingSeconds?: number): void {
    this.moves += 1;
    if (nags.length) {
      this.annotatedMoves += 1;
      this.negativeAnnotations += nags.filter((nag) => NEGATIVE_NAGS.has(nag)).length;
    }
    this.lastClockRemainingSeconds = clockRemainingSeconds !== undefined
      && Number.isFinite(clockRemainingSeconds)
      && clockRemainingSeconds >= 0
      ? clockRemainingSeconds
      : undefined;
  }
}
