/**
 * Online player-style fingerprint derived only from moves the player has
 * already made in the current game. The policy never sees future moves.
 *
 * These are behavioral measurements, not hand-assigned personality labels.
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
 * Tracks a single player's observable move tendencies during one game.
 * Call snapshot() before observe() for each move to avoid target leakage.
 */
export class PlayerStyleAccumulator {
  private moves = 0;
  private captures = 0;
  private checks = 0;
  private pawnAdvances = 0;
  private centralityTotal = 0;
  private centralityMoves = 0;
  private earlyQueenMoves = 0;
  private castled = false;
  private kingMoves = 0;
  private activePieceMoves = 0;

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
