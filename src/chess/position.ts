import { Chess } from "chess.js";

/**
 * 12 piece planes (6 piece types x 2 colors), one side-to-move feature,
 * and four castling-right features. Squares are indexed a1=0 ... h8=63.
 */
export const POSITION_FEATURES = 12 * 64 + 5;

const PIECE_PLANES: Record<string, number> = {
  p: 0, n: 1, b: 2, r: 3, q: 4, k: 5,
};

export function encodePosition(fen: string): Float32Array {
  const chess = new Chess(fen);
  const features = new Float32Array(POSITION_FEATURES);
  const board = chess.board();

  for (let row = 0; row < 8; row += 1) {
    for (let file = 0; file < 8; file += 1) {
      const piece = board[row]?.[file];
      if (!piece) continue;

      const piecePlane = PIECE_PLANES[piece.type];
      if (piecePlane === undefined) continue;

      const plane = piecePlane + (piece.color === "b" ? 6 : 0);
      const square = (7 - row) * 8 + file;
      features[plane * 64 + square] = 1;
    }
  }

  const fenFields = fen.split(/\\s+/);
  features[768] = fenFields[1] === "w" ? 1 : 0;
  const castling = fenFields[2] ?? "-";
  features[769] = castling.includes("K") ? 1 : 0;
  features[770] = castling.includes("Q") ? 1 : 0;
  features[771] = castling.includes("k") ? 1 : 0;
  features[772] = castling.includes("q") ? 1 : 0;

  return features;
}

/** UCI from/to encoding: a1a1=0, a1b1=1, ... h8h8=4095. */
export function encodeMove(uci: string): number {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) {
    throw new Error(`Invalid UCI move: ${uci}`);
  }

  const squareIndex = (square: string): number =>
    (Number(square[1]) - 1) * 8 + square.charCodeAt(0) - 97;

  return squareIndex(uci.slice(0, 2)) * 64 + squareIndex(uci.slice(2, 4));
}
