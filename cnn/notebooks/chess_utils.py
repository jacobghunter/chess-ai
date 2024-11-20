import os
import chess
import numpy as np

DATABASE_DIR = 'Lichess Elite Database'


def get_dataset():
    if not os.path.isdir(DATABASE_DIR):
        raise Exception(
            'Lichess Elite Database directory not found. Please download the dataset from https://database.nikonoel.fr/ and extract it to the root directory of this project.')
    files = os.listdir(DATABASE_DIR)[1:]

    all_data = []
    with chess.engine.SimpleEngine.popen_uci(
            "stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
        engine.configure({"Skill Level": 20})
        for file in files:
            data = load_data_to_tensors(files[0], engine)
            all_data.extend(data)
        engine.quit()
    return all_data


def bitboard_to_tensor(board):
    # Initialize an empty 12-channel tensor of shape (8, 8, 12)
    board_tensor = np.zeros((8, 8, 12), dtype=int)

    # Loop through each piece type (6 for white and 6 for black)
    for piece_type in range(1, 7):  # piece types range from 1 (pawn) to 6 (king)
        for color in [chess.WHITE, chess.BLACK]:
            piece = chess.Piece(piece_type, color)
            piece_positions = board.pieces(piece.piece_type, color)

            # Update the corresponding bitboard channel
            for pos in piece_positions:
                row, col = divmod(pos, 8)
                board_tensor[row, col, (color == chess.WHITE)
                             * 6 + piece.piece_type - 1] = 1
            break

    return board_tensor


def load_data_to_tensors(pgn_file, engine, max_games=None):
    data = []
    with open(DATABASE_DIR + '/' + pgn_file, 'r') as pgn_file:
        count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None or (max_games and count >= max_games):
                break

            board = game.board()
            for move in game.mainline_moves():

                position_tensor = bitboard_to_tensor(board)
                available_moves = np.zeros(64, dtype=int)

                for move in board.legal_moves:
                    available_moves[move.to_square] = 1

                data.append(
                    (position_tensor, available_moves, get_best_move(board)))
                board.push(move)
            count += 1
    return data


def get_best_move(board, engine):
    result = engine.play(board, chess.engine.Limit(
        time=0.1))  # Adjust time as needed
    return result.move
