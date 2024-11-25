import chess
import numpy as np
import torch


def board_to_bitboard(board):
    # Initialize an empty 12-channel tensor of shape (12, 8, 8)
    board_tensor = np.zeros((12, 8, 8), dtype=int)

    for piece_type in range(1, 7):
        for color in [chess.WHITE, chess.BLACK]:
            piece_positions = list(board.pieces(piece_type, color))

            for pos in piece_positions:
                row, col = divmod(pos, 8)
                board_tensor[(color == chess.BLACK) * 6 +
                             piece_type - 1, 7 - row, col] = 1

    return torch.tensor(board_tensor)


def bitboard_to_board(board_tensor):
    board = chess.Board(fen=None)

    for piece_type in range(1, 7):
        for color in [chess.WHITE, chess.BLACK]:
            for row in range(8):
                for col in range(8):
                    if board_tensor[(color == chess.BLACK) * 6 +
                                    piece_type - 1, 7 - row, col] == 1:
                        square = chess.square(col, row)
                        piece = chess.Piece(piece_type, color)
                        board.set_piece_at(square, piece)

    return board


def tensor_square_to_chess_square(square):
        if torch.is_tensor(square):
            square = square.item()
        row = 7 - square // 8
        col = square % 8
        return chess.square(col, row)


def encode_piece_from_to(piece_type, from_square, to_square):
    # BIT SHITFTING WOOO
    encoded_value = (piece_type << 12) | (from_square << 6) | to_square
    return encoded_value


def decode_piece_from_to(target):
    piece_type = (target >> 12) & 0xF
    from_square = (target >> 6) & 0x3F
    to_square = target & 0x3F

    return piece_type, from_square, to_square

def decode_piece_from_to_tensor(target):
    if isinstance(target, list):
        target = torch.tensor(target, dtype=torch.long)
    
    piece_type = (target >> 12) & 0xF
    from_square = (target >> 6) & 0x3F
    to_square = target & 0x3F
    
    return torch.stack((piece_type, from_square, to_square), dim=-1)


def encode_target_from(piece_type, from_square):
    # Leave Piece as 1-12

    encoded_value = (piece_type << 6) | from_square
    return encoded_value


def decode_target_from(target):
    piece_type = (target >> 6) & 0xF
    from_square = target & 0x3F

    return piece_type, from_square


def print_tensor_board(board_tensor):
    """
    Convert a 12x8x8 tensor into a readable chess board and print it.

    Args:
        board_tensor (torch.Tensor): A 12x8x8 tensor where:
            - Index 0-5 represent white pieces (P, N, B, R, Q, K)
            - Index 6-11 represent black pieces (p, n, b, r, q, k)

    Returns:
        list: An 8x8 list of strings representing the chess board.
    """
    # Define piece mapping
    piece_map = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',  # White pieces
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'  # Black pieces
    }

    # Initialize an empty 8x8 board
    chess_board = [["." for _ in range(8)] for _ in range(8)]

    # Iterate over each layer (piece type) of the tensor
    for piece_idx in range(12):
        # Get the positions of the current piece type
        piece_positions = (
            board_tensor[piece_idx] == 1).nonzero(as_tuple=False)
        for pos in piece_positions:
            row, col = pos.tolist()
            chess_board[row][col] = piece_map[piece_idx]

    # Print the board
    print("Chess Board:")
    print("-" * 17)
    for row in chess_board:
        print(" ".join(row))
    print("-" * 17)

    return chess_board
