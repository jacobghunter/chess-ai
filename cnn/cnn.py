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

                best_move = get_best_move(board, engine)

                data.append((position_tensor, available_moves, best_move))
                board.push(move)
            count += 1
    return data


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
            positions, available_moves = positions.to(
                DEVICE), available_moves.to(DEVICE)

            optimizer.zero_grad()
            # Get the model's predictions
            outputs = model(positions.float(), available_moves.float())

            # Convert the best_move to the index of the correct move in the one-hot vector
            # Index of the best move in the 130-dimensional vector
            target = torch.argmax(best_move, dim=1).to(DEVICE)

            loss = criterion(outputs, target)  # Use CrossEntropyLoss
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get the predicted move
            # Count correct predictions
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model, board, device):
    # Convert the chess board to a tensor
    position_tensor = bitboard_to_tensor(board).unsqueeze(
        0).to(device)  # Add batch dimension and move to device

    available_moves = np.zeros(64, dtype=int)
    for move in board.legal_moves:
        available_moves[move.to_square] = 1

    # Pass the tensor through the model
    with torch.no_grad():  # No gradient tracking during inference
        outputs = model(position_tensor.float(), torch.tensor(available_moves))

    # Get the index of the highest predicted value (outputs are logits)
    # Get the move with highest logit
    _, predicted_move = torch.max(outputs, 1)

    # Decode the predicted move index
    predicted_move = predicted_move.item()

    # Extract start and end squares from the predicted move
    # Dividing by 128 to get the start square index
    from_square = predicted_move // (64 * 6 * 2)
    # Getting the end square index
    to_square = (predicted_move % (64 * 6 * 2)) // (6 * 2)

    # Extract piece color and type
    color_index = (predicted_move % (6 * 2)) // 6  # 0 for white, 1 for black
    piece_type_index = predicted_move % 6  # 0-5 for pawn, knight, etc.

    # Create a piece object for the predicted piece type and color
    color = chess.WHITE if color_index == 0 else chess.BLACK
    # Piece type is 1-indexed in chess
    piece = chess.Piece(piece_type_index + 1, color)

    # Create the chess Move object
    move = chess.Move(from_square, to_square)

    # If the move is legal, return it; otherwise, return None (invalid move)
    if move in board.legal_moves:
        print(f"Predicted move: {move} (Piece: {piece})")
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


def train_model(data_path, model_path=None, epochs=10):
    dataloader = DataLoader(
        load_dataset(data_path), batch_size=16, shuffle=True)
    print("Dataset loaded")
    model = ChessCNN().to(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if model_path is None:
        train(model, dataloader, criterion, optimizer, 10)

    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model


def main():
    # if not os.path.isdir(DATABASE_DIR):
    #     raise Exception(
    #         'Lichess Elite Database directory not found. Please download the dataset from https://database.nikonoel.fr/ and extract it to the root directory of this project.')
    # files = os.listdir(DATABASE_DIR)[1:]

    # get and save
    # dataset = retrieve_dataset(files, 100)
    # save_dataset(dataset, 'chess_dataset.pth')

    train_model('chess_dataset.pth', model_path='chess_model.pth')

    # mate in 1 correct is e2h5
    # b = chess.Board("r2qkbnr/1b5p/p1n2p2/1pN1pNp1/1P1p4/1Q4P1/PBPPBP1P/R3K2R")

    # print(b)

    # mate in 3, correct is g1g7
    # b = chess.Board("r3k3/1p1p1ppp/1p1B4/1p6/Pp6/P3P3/8/4K1QR")

    # print(engine.play(b, chess.engine.Limit(time=0.0001)).move)

    # engine.quit()

    # move = test(model, b, DEVICE)
    # print(move)


if __name__ == "__main__":
    main()
