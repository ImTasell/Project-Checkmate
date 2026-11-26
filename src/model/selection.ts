import * as tf from "@tensorflow/tfjs";
import { Chess } from "chess.js";
import type { ArchetypeEncoderModel } from "../data/archetypes.js";
import { encodePosition } from "../chess/position.js";
import {
  rankLegalMovesByDecisionQuality,
  type LegalMoveQualityContext,
  type LegalMoveQualityScore,
} from "./decision-quality.js";
import { rankLegalMoves } from "./policy.js";

const EPSILON = 1e-8;

export interface PolicyMoveScore {
  move: string;
  score: number;
}

export interface HumanMoveMixOptions {
  /** Multiplier for the centered human-annotation signal. Defaults to zero. */
  qualityWeight?: number;
  /** Softmax temperature. Higher values increase sampling diversity. */
  temperature?: number;
}

export interface RankedHumanMove {
  move: string;
  policyProbability: number;
  positiveAnnotationProbability?: number;
  interestingAnnotationProbability?: number;
  criticalAnnotationProbability?: number;
  qualityAdjustment: number;
  selectionProbability: number;
}

export interface PositionSelectionOptions extends HumanMoveMixOptions {
  policyArchetypeModel?: ArchetypeEncoderModel;
  qualityModel?: tf.LayersModel;
  qualityArchetypeModel?: ArchetypeEncoderModel;
}

export interface HumanMoveChoice {
  move: string;
  rankedMoves: RankedHumanMove[];
}

function validateMixOptions(options: HumanMoveMixOptions): {
  qualityWeight: number;
  temperature: number;
} {
  const qualityWeight = options.qualityWeight ?? 0;
  const temperature = options.temperature ?? 1;

  if (!Number.isFinite(qualityWeight) || qualityWeight < 0 || qualityWeight > 2) {
    throw new Error("Quality weight must be a finite number between 0 and 2.");
  }
  if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 20) {
    throw new Error("Sampling temperature must be greater than 0 and at most 20.");
  }
  return { qualityWeight, temperature };
}

/**
 * Combines the human move-policy prior with optional annotation probabilities.
 * Policy scores are normalized over legal candidates. The quality signal is
 * positiveAnnotationProbability - criticalAnnotationProbability and is
 * centered across candidates before blending. The default quality weight is
 * zero because annotation quality is weak supervision, not chess truth.
 */
export function rankHumanLikeMoves(
  policyScores: readonly PolicyMoveScore[],
  qualityScores?: readonly LegalMoveQualityScore[],
  options: HumanMoveMixOptions = {},
): RankedHumanMove[] {
  const { qualityWeight, temperature } = validateMixOptions(options);
  if (policyScores.length === 0) return [];
  if (qualityWeight > 0 && !qualityScores) {
    throw new Error("A decision-quality model is required when quality weight is greater than zero.");
  }

  const seenMoves = new Set<string>();
  let policyTotal = 0;
  for (const item of policyScores) {
    if (!item.move || seenMoves.has(item.move)) {
      throw new Error("Policy candidate moves must be unique non-empty UCI strings.");
    }
    if (!Number.isFinite(item.score) || item.score < 0) {
      throw new Error("Policy candidate scores must be finite and non-negative.");
    }
    seenMoves.add(item.move);
    policyTotal += item.score;
  }
  if (!Number.isFinite(policyTotal) || policyTotal <= 0) {
    throw new Error("Policy scores must contain positive total probability.");
  }

  const qualityByMove = new Map<string, LegalMoveQualityScore>();
  if (qualityScores) {
    for (const item of qualityScores) {
      if (!seenMoves.has(item.move) || qualityByMove.has(item.move)) {
        throw new Error("Decision-quality candidates must match the legal policy candidates exactly.");
      }
      for (const value of [
        item.positiveAnnotationProbability,
        item.interestingAnnotationProbability,
        item.criticalAnnotationProbability,
      ]) {
        if (!Number.isFinite(value) || value < 0 || value > 1) {
          throw new Error("Decision-quality probabilities must be finite values between 0 and 1.");
        }
      }
      qualityByMove.set(item.move, item);
    }
    if (qualityByMove.size !== seenMoves.size) {
      throw new Error("Decision-quality scores are missing one or more legal moves.");
    }
  }

  const raw = policyScores.map((item) => {
    const quality = qualityByMove.get(item.move);
    const utility = quality
      ? quality.positiveAnnotationProbability - quality.criticalAnnotationProbability
      : 0;
    return {
      move: item.move,
      policyProbability: item.score / policyTotal,
      positiveAnnotationProbability: quality?.positiveAnnotationProbability,
      interestingAnnotationProbability: quality?.interestingAnnotationProbability,
      criticalAnnotationProbability: quality?.criticalAnnotationProbability,
      rawQualityUtility: utility,
    };
  });

  const meanQualityUtility = raw.reduce((sum, item) => sum + item.rawQualityUtility, 0) / raw.length;
  const adjusted = raw.map((item) => ({
    ...item,
    qualityAdjustment: item.rawQualityUtility - meanQualityUtility,
  }));
  const logits = adjusted.map((item) =>
    (Math.log(Math.max(EPSILON, item.policyProbability))
      + qualityWeight * item.qualityAdjustment) / temperature,
  );
  const maxLogit = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maxLogit));
  const normalizer = exponentials.reduce((sum, value) => sum + value, 0);

  return adjusted.map((item, index) => ({
    move: item.move,
    policyProbability: item.policyProbability,
    ...(item.positiveAnnotationProbability !== undefined
      ? { positiveAnnotationProbability: item.positiveAnnotationProbability }
      : {}),
    ...(item.interestingAnnotationProbability !== undefined
      ? { interestingAnnotationProbability: item.interestingAnnotationProbability }
      : {}),
    ...(item.criticalAnnotationProbability !== undefined
      ? { criticalAnnotationProbability: item.criticalAnnotationProbability }
      : {}),
    qualityAdjustment: item.qualityAdjustment,
    selectionProbability: (exponentials[index] ?? 0) / normalizer,
  })).sort((a, b) =>
    b.selectionProbability - a.selectionProbability || a.move.localeCompare(b.move),
  );
}

/** Samples from a previously ranked distribution; inject an RNG for reproducibility. */
export function sampleHumanMove(
  rankedMoves: readonly RankedHumanMove[],
  random: () => number = Math.random,
): RankedHumanMove | undefined {
  if (rankedMoves.length === 0) return undefined;
  const value = random();
  if (!Number.isFinite(value) || value < 0 || value >= 1) {
    throw new Error("Random source must return a value in the interval [0, 1).");
  }

  let cumulative = 0;
  for (const candidate of rankedMoves) {
    cumulative += candidate.selectionProbability;
    if (value < cumulative) return candidate;
  }
  return rankedMoves[rankedMoves.length - 1];
}

/** Builds the legal candidate set and scores it with the policy and optional quality model. */
export function rankPositionForHumanSelection(
  policyModel: tf.LayersModel,
  context: LegalMoveQualityContext,
  options: PositionSelectionOptions = {},
): RankedHumanMove[] {
  const chess = new Chess(context.fen);
  const legalMoves = chess.moves({ verbose: true });
  if (legalMoves.length === 0) return [];

  const policyScores = rankLegalMoves(
    policyModel,
    encodePosition(context.fen),
    legalMoves,
    context.playerElo,
    context.styleFeatures,
    options.policyArchetypeModel,
    context.decisionFeatures,
  );

  const qualityScores = options.qualityModel
    ? rankLegalMovesByDecisionQuality(options.qualityModel, context, options.qualityArchetypeModel)
    : undefined;

  return rankHumanLikeMoves(policyScores, qualityScores, options);
}

/** Selects one legal move while retaining the full distribution for inspection. */
export function chooseHumanLikeMove(
  policyModel: tf.LayersModel,
  context: LegalMoveQualityContext,
  options: PositionSelectionOptions & { random?: () => number } = {},
): HumanMoveChoice | undefined {
  const rankedMoves = rankPositionForHumanSelection(policyModel, context, options);
  const selected = sampleHumanMove(rankedMoves, options.random ?? Math.random);
  return selected ? { move: selected.move, rankedMoves } : undefined;
}
