/**
 * Online player-style fingerprint derived only from moves the player has
 * already made. Accumulator state is serializable so a player's history can
 * continue across separate PGN imports.
 */
export const STYLE_FEATURE_NAMES = [
  "captureRate",
  "checkRate",
  "pawnAdvanceRate",
  "centralization",
  "earlyQueenActivity",
  "castlingTendency",
  "kingMoveRate",
  "pieceActivityRate",
  "confidence",
] as const;

export const STYLE_FEATURES = STYLE_FEATURE_NAMES.length;
export type StyleFeatureVector = readonly number[];

export interface StyleObservation {
  piece: string;
  from: string;
  to: string;
  flags: string;
  san: string;
}

/** Raw counters, rather than averages, preserve exact weighting across games. */
export interface PlayerStyleState {
  moves: number;
  captures: number;
  checks: number;
  pawnAdvances: number;
  centralityTotal: number;
  centralityMoves: number;
  earlyQueenMoves: number;
  castled: boolean;
  kingMoves: number;
  activePieceMoves: number;
}

function safeCount(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}

function safeTotal(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 0;
}

function squareCentrality(square: string): number {
  if (!/^[a-h][1-8]$/.test(square)) return 0;
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return 1 - (Math.abs(file - 3.5) + Math.abs(rank - 3.5)) / 7;
}

function isForwardPawnMove(observation: StyleObservation, color: "w" | "b"): boolean {
  const fromRank = Number(observation.from[1]);
  const toRank = Number(observation.to[1]);
  return color === "w" ? toRank > fromRank : toRank < fromRank;
}

/**
 * Tracks a player's observable move tendencies across one or more games.
 * Call snapshot() before observe() for every move to avoid target leakage.
 */
export class PlayerStyleAccumulator {
  private moves: number;
  private captures: number;
  private checks: number;
  private pawnAdvances: number;
  private centralityTotal: number;
  private centralityMoves: number;
  private earlyQueenMoves: number;
  private castled: boolean;
  private kingMoves: number;
  private activePieceMoves: number;

  constructor(state?: Partial<PlayerStyleState>) {
    this.moves = safeCount(state?.moves);
    this.captures = safeCount(state?.captures);
    this.checks = safeCount(state?.checks);
    this.pawnAdvances = safeCount(state?.pawnAdvances);
    this.centralityTotal = safeTotal(state?.centralityTotal);
    this.centralityMoves = safeCount(state?.centralityMoves);
    this.earlyQueenMoves = safeCount(state?.earlyQueenMoves);
    this.castled = state?.castled === true;
    this.kingMoves = safeCount(state?.kingMoves);
    this.activePieceMoves = safeCount(state?.activePieceMoves);
  }

  snapshot(): number[] {
    const denominator = Math.max(this.moves, 1);
    return [
      this.captures / denominator,
      this.checks / denominator,
      this.pawnAdvances / denominator,
      this.centralityMoves ? this.centralityTotal / this.centralityMoves : 0,
      this.earlyQueenMoves / denominator,
      this.castled ? 1 : 0,
      this.kingMoves / denominator,
      this.activePieceMoves / denominator,
      Math.min(this.moves / 12, 1),
    ];
  }

  serialize(): PlayerStyleState {
    return {
      moves: this.moves,
      captures: this.captures,
      checks: this.checks,
      pawnAdvances: this.pawnAdvances,
      centralityTotal: this.centralityTotal,
      centralityMoves: this.centralityMoves,
      earlyQueenMoves: this.earlyQueenMoves,
      castled: this.castled,
      kingMoves: this.kingMoves,
      activePieceMoves: this.activePieceMoves,
    };
  }

  observe(observation: StyleObservation, color: "w" | "b"): void {
    const ownMoveNumber = this.moves + 1;
    this.moves += 1;

    if (observation.flags.includes("c") || observation.flags.includes("e")) {
      this.captures += 1;
    }
    if (observation.san.includes("+") || observation.san.includes("#")) {
      this.checks += 1;
    }
    if (observation.piece === "p" && isForwardPawnMove(observation, color)) {
      this.pawnAdvances += 1;
    }
    if (observation.piece !== "p" && observation.piece !== "k") {
      this.centralityTotal += squareCentrality(observation.to);
      this.centralityMoves += 1;
      this.activePieceMoves += 1;
    }
    if (observation.piece === "q" && ownMoveNumber <= 10) {
      this.earlyQueenMoves += 1;
    }
    if (observation.flags.includes("k") || observation.flags.includes("q")) {
      this.castled = true;
    } else if (observation.piece === "k") {
      this.kingMoves += 1;
    }
  }
}
