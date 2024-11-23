import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from chess_cnn import ChessCNN
from chess_dataset import ChessDataset
from torch.utils.data import DataLoader
from custom_collate_fn import custom_collate_fn
from custom_loss import CustomMoveLoss
from torch.optim.lr_scheduler import StepLR
import os
from utilities import board_to_bitboard, bitboard_to_board, decode_target_from, encode_piece_from_to, decode_piece_from_to, encode_target_from
from torch.amp import autocast, GradScaler

PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

DATABASE_DIR = '../Lichess Elite Database'
ENGINE_PATH = '../stockfish/stockfish-windows-x86-64-avx2.exe'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# EVERYTHING IS 12x8x8


def encode_move(move, board):
    # Initialize a vector of 130 zeros
    encoded_move = np.zeros(130, dtype=int)

    # Get the piece at the start square once
    piece = board.piece_at(move.from_square)
    if piece is None:
        return encoded_move  # If there's no piece, return empty encoding

    # Step 1: Encode the starting square (64-dimensional)
    # This is an integer representing the square (0-63)
    start_square = move.from_square
    encoded_move[start_square] = 1  # Set the corresponding index to 1

    # Step 2: Encode the ending square (64-dimensional)
    # This is an integer representing the square (0-63)
    end_square = move.to_square
    # Set the corresponding index to 1 (after the first 64)
    encoded_move[64 + end_square] = 1

    # Step 3: Encode the piece type (1-dimensional)
    piece_type = piece.piece_type  # Get piece type (pawn=1, knight=2, etc.)
    encoded_move[128] = PIECE_MAP.get(
        piece_type, -1)  # Map piece type to index

    # Step 4: Encode the piece color (1-dimensional)
    piece_color = piece.color  # Get piece color (white=True, black=False)
    # 0 for white, 1 for black
    encoded_move[129] = 0 if piece_color == chess.WHITE else 1

    return encoded_move


def move_to_tensor(board, move):
    # Initialize empty tensors for "to" and "from" positions
    move_to_tensor = np.zeros((12, 8, 8), dtype=int)
    move_from_tensor = np.zeros((12, 8, 8), dtype=int)

    # Get the piece that moved
    piece = board.piece_at(move.from_square)

    if piece:
        # Piece type: white pieces (0-5), black pieces (6-11)
        piece_idx = PIECE_MAP[piece.piece_type] if piece.color == chess.WHITE else PIECE_MAP[piece.piece_type + 6]

        # Set the starting square ("from" position)
        from_row, from_col = divmod(move.from_square, 8)
        move_from_tensor[piece_idx, from_row, from_col] = 1

        # Set the target square ("to" position)
        to_row, to_col = divmod(move.to_square, 8)
        move_to_tensor[piece_idx, to_row, to_col] = 1

    return move_to_tensor, move_from_tensor


def available_moves_tensor(board):
    # Create tensors to store available moves (to/from) for each piece
    # One for each piece type (12)
    available_moves_to = np.zeros((12, 8, 8))
    available_moves_from = np.zeros(
        (12, 8, 8))  # One for each piece type (12)

    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square

        piece = board.piece_at(from_square)
        piece_type = piece.piece_type - 1  # Convert to 0-indexed piece type

        # Mark valid moves (to/from) for the piece
        from_row = from_square // 8
        from_col = from_square % 8
        to_row = to_square // 8
        to_col = to_square % 8

        # Adjust for black pieces
        if piece.color == chess.WHITE:
            from_row = 7 - from_row
            to_row = 7 - to_row

        available_moves_from[piece_type, from_row, from_col] = 1
        available_moves_to[piece_type, to_row, to_col] = 1

        # if piece.color == chess.BLACK:
        #     print(board_to_bitboard(board))

        #     print(move)
        #     print(from_row, from_col)
        #     print(to_row, to_col)
        #     print(available_moves_from[piece_type])
        #     print(available_moves_to[piece_type])

        #     exit(0)

    return available_moves_to, available_moves_from


def load_data_to_tensors(pgn_file, engine, max_games=None):
    data = []
    with open(DATABASE_DIR + '/' + pgn_file, 'r') as pgn_file:
        count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None or (max_games and count >= max_games):
                break

            if game.headers['Result'] != '1-0':
                continue

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if i % 2 == 0:
                    position_bitboard = board_to_bitboard(board)
                    legal_moves = [encode_piece_from_to(board.piece_at(
                        move.from_square).piece_type, move.from_square, move.to_square) for move in list(board.legal_moves)]

                    target_from_square = move.from_square
                    target_to = move.to_square
                    target_from = encode_target_from(
                        board.piece_at(target_from_square).piece_type, target_from_square)

                    # in the form ( 12x8x8, integers representing legal moves (type, from square, to square), the target from (type, from square), the target to (to square) )
                    data.append(
                        (position_bitboard, legal_moves, target_from, target_to))

                board.push(move)
            count += 1
    return data


def move_to_index(move):
    """
    Convert a move into a corresponding index for the target output.
    The move must be from a square (0-63) to another square (0-63).
    """
    from_square = move.from_square
    to_square = move.to_square
    # Combine the from and to squares into a single index (if needed)
    return from_square * 64 + to_square


def get_best_move(board, engine):
    result = engine.play(board, chess.engine.Limit(
        time=0.0001))
    return encode_move(result.move, board)


def create_move_tensor(square, piece_type=None):
    if piece_type is None:
        tensor = torch.zeros((8, 8), dtype=torch.long)
        tensor[square // 8, square % 8] = 1
        return tensor
    else:
        tensor = torch.zeros((12, 8, 8), dtype=torch.long)
        tensor[piece_type, square // 8, square % 8] = 1
        return tensor


def train(model, dataloader, criterion, optimizer, epochs, scheduler=None):
    print("Starting training")
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (position_bitboards, legal_moves, target_froms, target_tos) in enumerate(dataloader):
            print(f'Batch {i+1}/{len(dataloader)}')
            position_bitboards = position_bitboards.to(DEVICE).float()
            legal_moves = legal_moves.to(DEVICE)
            target_froms = target_froms.to(DEVICE)
            target_tos = target_tos.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                logits_pieces, logits_moves = model(
                    position_bitboards, legal_moves
                )

                logits_pieces_flat = logits_pieces.view(-1, 12 * 8 * 8)
                logits_moves_flat = logits_moves.view(-1, 8 * 8)

                # Compute losses
                loss_piece = criterion(logits_pieces_flat, target_froms)
                loss_move = criterion(logits_moves_flat, target_tos)
                total_loss = loss_piece + loss_move

            # Backpropagation
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()

            predicted_piece = torch.argmax(logits_pieces_flat, dim=1)
            predicted_move = torch.argmax(logits_moves_flat, dim=1)

            correct_predictions += (
                (predicted_piece == target_froms).sum().item() +
                (predicted_move == target_tos).sum().item()
            )
            total_predictions += target_froms.size(0) + target_tos.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model, board, device):
    """
    Test the model's move prediction for a given chess board state.

    Args:
        model (torch.nn.Module): The trained model for predicting chess moves.
        board (chess.Board): The current chess board state.
        device (torch.device): The device to run inference on (CPU or GPU).

    Returns:
        chess.Move or None: The predicted legal move, or None if no valid move is predicted.
    """
    # Encode the position bitboard
    position_bitboard = board_to_bitboard(board)

    # Encode legal moves for the current board state
    legal_moves = [
        encode_piece_from_to(
            board.piece_at(move.from_square).piece_type,
            move.from_square,
            move.to_square
        ) for move in list(board.legal_moves)
    ]

    # Simulate targets (dummy values for testing)
    # Targets are typically provided during training; for testing, we don't have true labels.
    target_from, target_to = 0, 0

    # Create a single "batch" entry
    batch = [(position_bitboard, legal_moves, target_from, target_to)]

    # Use the custom collate function to prepare the batch
    position_bitboards, padded_legal_moves, target_froms, target_tos = custom_collate_fn(
        batch)

    # Move tensors to the appropriate device
    position_bitboards = position_bitboards.to(device)
    padded_legal_moves = padded_legal_moves.to(device)

    # Perform inference with the model
    with torch.no_grad():  # No gradient tracking during inference
        logits_piece, logits_move = model(
            position_bitboards.float(), padded_legal_moves)

    # Decode predicted piece and move indices
    predicted_piece_index = torch.argmax(logits_piece[0]).item()
    predicted_move_index = torch.argmax(logits_move[0]).item()

    # Ensure the predicted move index is valid
    predicted_legal_move = padded_legal_moves[0, predicted_move_index].item()

    if predicted_legal_move != 0:  # Ignore padded zero entries
        # Decode the legal move using its encoded format
        piece_type, from_square, to_square = decode_piece_from_to(
            predicted_legal_move)

        # Create the move object
        move = chess.Move(from_square, to_square)

        # Check if the move is legal
        if move in board.legal_moves:
            print(f"Predicted move: {board.san(move)}")
            return move
        else:
            print(f"Invalid move: {move}")
    else:
        print("Predicted move corresponds to a padded entry (invalid).")

    return None


def retrieve_dataset(files, games_per_file=None):
    all_data = []
    # with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
    for file in files:
        print(f"Processing {file}")
        data = load_data_to_tensors(file, None, games_per_file)
        all_data.extend(data)
        # engine.quit()

    return ChessDataset(all_data)


def save_dataset(data, file_path):
    torch.save(data, file_path)
    print(f"Dataset saved to {file_path}")


def load_dataset(file_path):
    return torch.load(file_path)


def train_model(data_path, load=None, model_path=None, epochs=10):
    dataloader = DataLoader(
        load_dataset(data_path),
        batch_size=12288,  # this could def be too big but it seems to be going strong for now
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    print("Dataset loaded")
    model = ChessCNN().to(device=DEVICE)
    # try tuning the hyperparamteres
    # criterion = CustomMoveLoss().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Reduces learning rate by a factor of 1/2 every 10 epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    if load is None:
        train(model, dataloader, criterion, optimizer, epochs, None)

    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model


def load_model(model_path):
    model = ChessCNN().to(device=DEVICE)
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    # if not os.path.isdir(DATABASE_DIR):
    #     raise Exception(
    #         'Lichess Elite Database directory not found. Please download the dataset from https://database.nikonoel.fr/ and extract it to the root directory of this project.')
    # files = os.listdir(DATABASE_DIR)[30:]

    # dataset = retrieve_dataset(files, 50)
    # save_dataset(dataset, 'chess_dataset.pth')

    model = train_model('chess_dataset.pth',
                        model_path='chess_model.pth', epochs=10)

    # model = load_model('chess_model.pth')

    # mate in 1 correct is e2h5
    b = chess.Board("r2qkbnr/1b5p/p1n2p2/1pN1pNp1/1P1p4/1Q4P1/PBPPBP1P/R3K2R")

    print(b)

    # mate in 3, correct is g1g7
    # b = chess.Board("r3k3/1p1p1ppp/1p1B4/1p6/Pp6/P3P3/8/4K1QR")

    # print(engine.play(b, chess.engine.Limit(time=0.0001)).move)

    # engine.quit()

    move = test(model, b, DEVICE)
    print(move)


if __name__ == "__main__":
    main()
