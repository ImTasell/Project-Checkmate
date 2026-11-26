import { Chess } from "chess.js";

/**
 * True when, immediately after a move, the opponent has a legal capture onto
 * the moved piece's destination and the mover has no immediate legal recapture
 * after that capture. This is a one-ply exposure descriptor, not an exchange
 * evaluation or a judgement about whether the offer was intentional.
 */
export function hasUnansweredCaptureExposure(afterMoveFen: string, destination: string): boolean {
  const responsePosition = new Chess(afterMoveFen);
  const captureReplies = responsePosition.moves({ verbose: true }).filter(
    (reply) => reply.to === destination && reply.captured !== undefined,
  );

  for (const reply of captureReplies) {
    const afterCapture = new Chess(afterMoveFen);
    afterCapture.move({
      from: reply.from,
      to: reply.to,
      ...(reply.promotion ? { promotion: reply.promotion } : {}),
    });
    const hasImmediateRecapture = afterCapture.moves({ verbose: true }).some(
      (recapture) => recapture.to === destination && recapture.captured !== undefined,
    );
    if (!hasImmediateRecapture) return true;
  }

  return false;
}
