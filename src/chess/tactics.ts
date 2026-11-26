import { Chess, type PieceSymbol, type Square } from "chess.js";

export const TACTICAL_FEATURE_NAMES = [
  "moveIsCapture",
  "moveIsEnPassant",
  "moveIsCastling",
  "moveIsPromotion",
  "moveGivesCheck",
  "moveGivesCheckmate",
  "moveGivesStalemate",
  "movedPiecePawn",
  "movedPieceKnight",
  "movedPieceBishop",
  "movedPieceRook",
  "movedPieceQueen",
  "movedPieceKing",
  "capturedPiecePawn",
  "capturedPieceKnight",
  "capturedPieceBishop",
  "capturedPieceRook",
  "capturedPieceQueen",
  "capturedPieceKing",
  "destinationAttackedByOpponentAfter",
  "destinationDefendedByMoverAfter",
  "opponentAttackersAfterNormalized",
  "moverDefendersAfterNormalized",
  "opponentHasLegalCaptureAfter",
  "opponentLegalCapturesAfterNormalized",
  "sideWasInCheckBefore",
  "originAttackedByOpponentBefore",
  "destinationDefendedByOpponentBefore",
  "destinationAttackedByMoverBefore",
] as const;

export const TACTICAL_FEATURES = TACTICAL_FEATURE_NAMES.length;

const PIECE_INDEX: Record<PieceSymbol, number> = {
  p: 0,
  n: 1,
  b: 2,
  r: 3,
  q: 4,
  k: 5,
};

const MOVED_PIECE_OFFSET = 7;
const CAPTURED_PIECE_OFFSET = MOVED_PIECE_OFFSET + 6;
const AFTER_MOVE_OFFSET = CAPTURED_PIECE_OFFSET + 6;
const BEFORE_MOVE_OFFSET = AFTER_MOVE_OFFSET + 6;

function normalizeCount(count: number): number {
  return Math.min(4, Math.max(0, count)) / 4;
}

/**
 * Encodes tactical properties of one legal candidate move.
 *
 * Attack-map counts are geometric signals: chess.js may include pinned pieces
 * in attackers(). The legal-capture fields instead inspect the opponent's
 * actual legal replies after the candidate move. These are descriptors, not
 * an exchange evaluation or an engine judgment.
 */
export function encodeMoveTactics(fen: string, moveUci: string): Float32Array {
  const match = moveUci.toLowerCase().match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/);
  if (!match?.[1] || !match[2]) {
    throw new Error("Invalid UCI move for tactical context: " + moveUci);
  }

  const from = match[1] as Square;
  const to = match[2] as Square;
  const promotion = match[3];
  const chess = new Chess(fen);
  const moverColor = chess.turn();
  const opponentColor = moverColor === "w" ? "b" : "w";

  const wasInCheck = chess.isCheck();
  const originAttackedBefore = chess.attackers(from, opponentColor).length > 0;
  const destinationDefendedByOpponentBefore =
    chess.attackers(to, opponentColor).length > 0;
  // Exclude the moving piece itself: it attacks its own destination before
  // moving, which would otherwise make this feature trivially true.
  const destinationAttackedByMoverBefore =
    chess.attackers(to, moverColor).some((square) => square !== from);

  const move = chess.move({
    from,
    to,
    ...(promotion ? { promotion: promotion as "q" | "r" | "b" | "n" } : {}),
  });

  const opponentAttackersAfter = chess.attackers(to, opponentColor).length;
  const moverDefendersAfter = chess.attackers(to, moverColor).length;
  const legalCaptureReplies = chess.moves({ verbose: true }).filter(
    (reply) => reply.to === to && reply.captured !== undefined,
  );

  const features = new Float32Array(TACTICAL_FEATURES);
  features[0] = move.captured !== undefined ? 1 : 0;
  features[1] = move.flags.includes("e") ? 1 : 0;
  features[2] = move.flags.includes("k") || move.flags.includes("q") ? 1 : 0;
  features[3] = move.promotion !== undefined ? 1 : 0;
  features[4] = chess.isCheck() ? 1 : 0;
  features[5] = chess.isCheckmate() ? 1 : 0;
  features[6] = chess.isStalemate() ? 1 : 0;

  features[MOVED_PIECE_OFFSET + PIECE_INDEX[move.piece]] = 1;
  if (move.captured !== undefined) {
    features[CAPTURED_PIECE_OFFSET + PIECE_INDEX[move.captured]] = 1;
  }

  features[AFTER_MOVE_OFFSET] = opponentAttackersAfter > 0 ? 1 : 0;
  features[AFTER_MOVE_OFFSET + 1] = moverDefendersAfter > 0 ? 1 : 0;
  features[AFTER_MOVE_OFFSET + 2] = normalizeCount(opponentAttackersAfter);
  features[AFTER_MOVE_OFFSET + 3] = normalizeCount(moverDefendersAfter);
  features[AFTER_MOVE_OFFSET + 4] = legalCaptureReplies.length > 0 ? 1 : 0;
  features[AFTER_MOVE_OFFSET + 5] = normalizeCount(legalCaptureReplies.length);

  features[BEFORE_MOVE_OFFSET] = wasInCheck ? 1 : 0;
  features[BEFORE_MOVE_OFFSET + 1] = originAttackedBefore ? 1 : 0;
  features[BEFORE_MOVE_OFFSET + 2] = destinationDefendedByOpponentBefore ? 1 : 0;
  features[BEFORE_MOVE_OFFSET + 3] = destinationAttackedByMoverBefore ? 1 : 0;

  return features;
}
