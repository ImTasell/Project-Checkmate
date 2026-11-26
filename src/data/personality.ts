import type { PlayerProfileDatabase, PlayerProfileRecord } from "./players.js";
import { PlayerStyleAccumulator } from "./style.js";

export const PLAYER_PERSONALITY_VERSION = 1;
export const DEFAULT_PERSONALITY_MIN_MOVES = 24;

export interface PlayerTendency {
  /** A rate in [0, 1], or null when there is not enough evidence. */
  value: number | null;
  numerator: number;
  denominator: number;
  evidenceMoves: number;
  description: string;
}

export interface PlayerPersonalityProfile {
  id: string;
  displayName: string;
  games: number;
  moves: number;
  eligible: boolean;
  minimumMoves: number;
  contextObservedMoves: number;
  tendencies: {
    captureFrequency: PlayerTendency;
    checkFrequency: PlayerTendency;
    forcingPlayFrequency: PlayerTendency;
    quietMoveFrequency: PlayerTendency;
    pawnExpansionFrequency: PlayerTendency;
    centralization: PlayerTendency;
    pieceActivity: PlayerTendency;
    earlyQueenActivity: PlayerTendency;
    kingMoveFrequency: PlayerTendency;
    unansweredExposureFrequency: PlayerTendency;
    acceptedExposureFrequency: PlayerTendency;
    castledAtLeastOnce: boolean | null;
  };
}

function rate(
  numerator: number,
  denominator: number,
  evidenceMoves: number,
  minimumMoves: number,
  description: string,
): PlayerTendency {
  const eligible = evidenceMoves >= minimumMoves && denominator > 0;
  return {
    value: eligible ? Math.max(0, Math.min(1, numerator / denominator)) : null,
    numerator,
    denominator,
    evidenceMoves,
    description,
  };
}

/**
 * Turns accumulated move counters into descriptive tendencies.
 *
 * These are observable frequencies, not psychological labels or measures of
 * playing strength. An unanswered exposure is not proof of an intentional
 * sacrifice: the move may simply be a mistake.
 */
export function buildPlayerPersonality(
  profile: PlayerProfileRecord,
  minimumMoves = DEFAULT_PERSONALITY_MIN_MOVES,
): PlayerPersonalityProfile {
  if (!Number.isInteger(minimumMoves) || minimumMoves < 1) {
    throw new Error("minimumMoves must be a positive integer.");
  }

  const state = new PlayerStyleAccumulator(profile.styleState).serialize();
  const moves = state.moves;
  const contextMoves = state.contextObservedMoves;
  const eligible = moves >= minimumMoves;
  const forcingMoves = Math.min(moves, state.captures + state.checks);

  return {
    id: profile.id,
    displayName: profile.displayName,
    games: profile.games,
    moves,
    eligible,
    minimumMoves,
    contextObservedMoves: contextMoves,
    tendencies: {
      captureFrequency: rate(state.captures, moves, moves, minimumMoves,
        "Share of the player's recorded moves that were captures."),
      checkFrequency: rate(state.checks, moves, moves, minimumMoves,
        "Share of the player's recorded moves that gave check."),
      forcingPlayFrequency: rate(forcingMoves, moves, moves, minimumMoves,
        "Share of moves that were captures or checks; a descriptive forcing-move proxy."),
      quietMoveFrequency: rate(state.quietMoves, contextMoves, contextMoves, minimumMoves,
        "Share of context-tracked moves that were neither captures nor checks."),
      pawnExpansionFrequency: rate(state.pawnAdvances, moves, moves, minimumMoves,
        "Share of moves that advanced a pawn toward the opponent's side."),
      centralization: rate(state.centralityTotal, state.centralityMoves, state.centralityMoves, minimumMoves,
        "Average geometric centrality of non-pawn, non-king piece destinations."),
      pieceActivity: rate(state.activePieceMoves, moves, moves, minimumMoves,
        "Share of moves made by knights, bishops, rooks, or queens."),
      earlyQueenActivity: rate(state.earlyQueenMoves, moves, moves, minimumMoves,
        "Share of moves spent moving the queen during the player's first ten moves."),
      kingMoveFrequency: rate(state.kingMoves, moves, moves, minimumMoves,
        "Share of moves made by the king, excluding castling."),
      unansweredExposureFrequency: rate(
        state.unansweredExposureOpportunities,
        contextMoves,
        contextMoves,
        minimumMoves,
        "Share of context-tracked moves that allowed an immediate legal capture with no immediate legal recapture.",
      ),
      acceptedExposureFrequency: rate(
        state.acceptedExposureOpportunities,
        state.unansweredExposureOpportunities,
        contextMoves,
        minimumMoves,
        "Share of unanswered exposure opportunities that the opponent accepted on the very next move.",
      ),
      castledAtLeastOnce: eligible ? state.castled : null,
    },
  };
}

export interface PlayerPersonalityReport {
  version: number;
  generatedAt: string;
  minimumMoves: number;
  totalPlayers: number;
  eligiblePlayers: number;
  excludedPlayers: number;
  profiles: PlayerPersonalityProfile[];
}

export function buildPlayerPersonalityReport(
  database: PlayerProfileDatabase,
  options: { minimumMoves?: number; generatedAt?: string } = {},
): PlayerPersonalityReport {
  const minimumMoves = options.minimumMoves ?? DEFAULT_PERSONALITY_MIN_MOVES;
  if (!Number.isInteger(minimumMoves) || minimumMoves < 1) {
    throw new Error("minimumMoves must be a positive integer.");
  }

  const profiles = Object.values(database.players)
    .map((profile) => buildPlayerPersonality(profile, minimumMoves))
    .sort((a, b) => b.moves - a.moves || a.id.localeCompare(b.id));
  const eligiblePlayers = profiles.filter((profile) => profile.eligible).length;

  return {
    version: PLAYER_PERSONALITY_VERSION,
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    minimumMoves,
    totalPlayers: profiles.length,
    eligiblePlayers,
    excludedPlayers: profiles.length - eligiblePlayers,
    profiles,
  };
}
