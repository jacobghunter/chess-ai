import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from chess_cnn import ChessCNN
from chess_dataset import ChessDataset
import os
from torch.utils.data import DataLoader

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


def bitboard_to_tensor(board):
    # Initialize an empty 12-channel tensor of shape (8, 8, 12)
    board_tensor = np.zeros((12, 8, 8), dtype=int)

    # Loop through each piece type (6 for white and 6 for black)
    for piece_type in range(1, 7):  # piece types range from 1 (pawn) to 6 (king)
        for color in [chess.WHITE, chess.BLACK]:
            piece = chess.Piece(piece_type, color)
            piece_positions = board.pieces(piece.piece_type, color)

            # Update the corresponding bitboard channel
            for pos in piece_positions:
                row, col = divmod(pos, 8)
                # Mark the position of the piece in the correct channel
                board_tensor[(color == chess.WHITE)
                             * 6 + piece.piece_type - 1, row, col] = 1

    return torch.tensor(board_tensor)


def move_to_tensor(board, move):
    # TODO: maybe I need the rest of the pieces on this as well? unsure

    # Initialize an empty 12x8x8 tensor for the label
    move_tensor = np.zeros((12, 8, 8), dtype=int)

    # Get the piece that moved
    piece = board.piece_at(move.from_square)

    if piece:
        # Piece type: white pieces (0-5), black pieces (6-11)
        piece_idx = PIECE_MAP[piece.piece_type] if piece.color == chess.WHITE else PIECE_MAP[piece.piece_type + 6]

        # Reset the starting square (from_square)
        from_row, from_col = divmod(move.from_square, 8)
        move_tensor[piece_idx, from_row, from_col] = 0

        # Set the target square (to_square)
        to_row, to_col = divmod(move.to_square, 8)
        # Mark the piece at the new position
        move_tensor[piece_idx, to_row, to_col] = 1

    return move_tensor


def available_moves_to_tensor(board):
    available_moves_tensor = np.zeros((12, 8, 8), dtype=int)

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece:
            # Set 1 where the move is legal for this piece type
            piece_idx = PIECE_MAP[piece.piece_type] if piece.color == chess.WHITE else PIECE_MAP[piece.piece_type + 6]
            row, col = divmod(move.to_square, 8)
            available_moves_tensor[piece_idx, row, col] = 1

    return available_moves_tensor


def load_data_to_tensors(pgn_file, engine, max_games=None):
    data = []
    with open(DATABASE_DIR + '/' + pgn_file, 'r') as pgn_file:
        # TODO: load only the games white wins
        # make the bot good at playing winning white moves and only play white (maybe still do losing and draw games)
        # for the best move, just put the next move made by white (maybe, check how fast stockfish is)

        # for the available moves, just ignore dupes and put a 1 even if 3 pieces can move there (should be chessboard sized)
        # just have the ouput be 64x64 and add more complexity as time goes on
        # that should limit the output of illegal moves

        # custom loss function to heavily penalize illegal moves

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
                    position_tensor = bitboard_to_tensor(board)

                    available_moves = available_moves_to_tensor(board)

                    next_move = move_to_tensor(board, move)

                    data.append((position_tensor, available_moves, next_move))
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


def train(model, dataloader, criterion, optimizer, epochs):
    print("Starting training")
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in dataloader:
            positions, available_moves, best_move = batch
            positions, available_moves, best_move = (
                positions.to(DEVICE).float(),
                available_moves.to(DEVICE).float(),
                best_move.to(DEVICE),
            )

            optimizer.zero_grad()
            # Get the model's predictions
            logits = model(positions, available_moves)

            # Flatten best_move to get the target index (best move index)
            # We will use argmax across the channels to select the piece being moved
            # The best_move has shape [batch_size, 12, 8, 8], we need to flatten it to indices.
            best_move_flat = best_move.view(best_move.size(0), -1)
            # Get the index of the active move (where the best move is non-zero)
            best_move_index = torch.argmax(best_move_flat, dim=1)
            # Ensure best_move_index is of type long, as expected by CrossEntropyLoss
            best_move_index = best_move_index.long()

            # Flatten logits to match the shape required for CrossEntropyLoss (batch_size, 12*8*8)
            # Flatten logits to [batch_size, 12*8*8]
            logits_flat = logits.view(logits.size(0), -1)

            loss = criterion(logits_flat, best_move_index)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted_flat = torch.argmax(
                logits_flat, dim=1)  # Predicted indices (flat)
            correct_predictions += (predicted_flat ==
                                    best_move_index).sum().item()
            total_predictions += best_move_index.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model, board, device):
    # Convert the chess board to a tensor
    position_tensor = bitboard_to_tensor(board).unsqueeze(
        0).to(device)  # Add batch dimension and move to device

    # Create the available moves tensor (12x8x8 encoding)
    available_moves_tensor = torch.zeros(
        (12, 8, 8), dtype=torch.float32).to(device)
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        # Convert piece to index (0-indexed)
        piece_type = board.piece_at(from_square).piece_type - 1

        # Mark the available move in the corresponding channel (piece type) and position (from -> to)
        available_moves_tensor[piece_type, to_square // 8, to_square % 8] = 1

    # Pass the tensor through the model
    with torch.no_grad():  # No gradient tracking during inference
        outputs = model(position_tensor.float(), available_moves_tensor)

    # Flatten the output tensor and apply the available moves mask
    outputs = outputs.view(-1)  # Flatten to a 1D tensor of shape (12*8*8,)

    # Apply the available moves mask: set the invalid moves to a very low value
    # Flatten the available moves tensor
    available_moves_mask = available_moves_tensor.view(-1)
    # Mask the logits with the available moves
    outputs = outputs * available_moves_mask

    # Get the index of the highest predicted value (outputs are logits)
    predicted_move_index = torch.argmax(outputs).item()

    # Decode the predicted move index to (from_square, to_square)
    # Row of the piece (which piece is moved)
    from_square = predicted_move_index // 64
    to_square = predicted_move_index % 64   # Column of the destination square

    move = chess.Move(from_square, to_square)

    # If the move is legal, return it; otherwise, return None (invalid move)
    if move in board.legal_moves:
        print(f"Predicted move: {move}")
        return move
    else:
        print(f"Invalid move: {move}")
        return None


def retrieve_dataset(files, games_per_file=None):
    all_data = []
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        for file in files:
            print(f"Processing {file}")
            data = load_data_to_tensors(file, engine, games_per_file)
            all_data.extend(data)
        engine.quit()

    return ChessDataset(all_data)


def save_dataset(data, file_path):
    torch.save(data, file_path)
    print(f"Dataset saved to {file_path}")


def load_dataset(file_path):
    return torch.load(file_path)


def train_model(data_path, load=None, model_path=None, epochs=10):
    dataloader = DataLoader(
        load_dataset(data_path), batch_size=32, shuffle=True)
    print("Dataset loaded")
    model = ChessCNN().to(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if load is None:
        train(model, dataloader, criterion, optimizer, epochs)

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
    # files = os.listdir(DATABASE_DIR)[1:]

    # dataset = retrieve_dataset(files, 100)
    # save_dataset(dataset, 'chess_dataset.pth')

    model = train_model('chess_dataset.pth',
                        model_path='chess_model.pth', epochs=5)

    # model = load_model('chess_model.pth')

    # mate in 1 correct is e2h5
    b = chess.Board("r2qkbnr/1b5p/p1n2p2/1pN1pNp1/1P1p4/1Q4P1/PBPPBP1P/R3K2R")

    # mate in 3, correct is g1g7
    # b = chess.Board("r3k3/1p1p1ppp/1p1B4/1p6/Pp6/P3P3/8/4K1QR")

    # print(engine.play(b, chess.engine.Limit(time=0.0001)).move)

    # engine.quit()
    print(b)

    move = test(model, b, DEVICE)
    print(move)


if __name__ == "__main__":
    main()
