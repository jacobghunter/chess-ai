import glob
import time
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
from utilities import bitboard_to_board, board_to_bitboard, decode_target_from, encode_piece_from_to, encode_target_from, swap_chess_array_square, tensor_square_to_chess_square
from torch.amp import autocast, GradScaler
from torch.utils.data import random_split

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

                    # Format: [type, from square, to square]
                    legal_moves = torch.zeros(len(list(board.legal_moves)), 3)
                    valid_pieces_mask = torch.zeros(12 * 8 * 8)
                    for i, legal_move in enumerate(list(board.legal_moves)):
                        piece_type, from_square, to_square  = board.piece_at(legal_move.from_square).piece_type, legal_move.from_square, legal_move.to_square
                        piece_type = piece_type - 1
                        legal_moves[i] = torch.tensor((piece_type, from_square, to_square))

                        valid_pieces_mask[(piece_type) * 64 + from_square] = 1

                    target_to_square = move.to_square

                    target_from = encode_target_from(
                        board.piece_at(move.from_square).piece_type - 1, move.from_square)
                        

                    # in the form ( 12x8x8, legal moves , the target from (type, from square), the target to (to square) )
                    # SQUARE LABELS ARE IN ARRAY FORM
                    data.append(
                        (position_bitboard, valid_pieces_mask, legal_moves, target_from, target_to_square))

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


def train(model, dataloader, val_dataloader, criterion, optimizer, epochs, alpha, beta, scheduler=None, warmup_scheduler=None):
    print("Starting training")
    scaler = GradScaler()
    for epoch in range(epochs):
        # prioritize piece prediction at the start of training, then prioritize move prediction
        alpha = 1.0 - epoch / epochs
        beta = epoch / epochs

        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        is_warming_up = True
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True
        #     ) as prof:
        for i, (position_bitboards, valid_pieces_masks, legal_moves, target_froms, target_to_squares) in enumerate(dataloader):
            # start_time = time.time()
            # print(f'Batch {i+1}/{len(dataloader)}')
            position_bitboards = position_bitboards.to(DEVICE).float()
            valid_pieces_masks = valid_pieces_masks.to(DEVICE)
            legal_moves = legal_moves.to(DEVICE)
            target_froms = target_froms.to(DEVICE)
            target_to_squares = target_to_squares.to(DEVICE)

            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            
            with autocast(device_type='cuda', enabled=False):
                logits_pieces_flat, logits_moves_flat = model(
                    position_bitboards, valid_pieces_masks, legal_moves, current_epoch=1, total_epochs=epochs, inference=False
                )

                # Compute losses
                loss_piece = criterion(logits_pieces_flat, target_froms)
                loss_move = criterion(logits_moves_flat, target_to_squares)
                total_loss = alpha * loss_piece + beta * loss_move

            # Backpropagation
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            # prof.step()

            # unsure if neccesary
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.update()

            running_loss += total_loss.item()

            predicted_piece = torch.argmax(logits_pieces_flat, dim=1)
            predicted_move = torch.argmax(logits_moves_flat, dim=1)

            # Count correct predictions
            correct_predictions += (
                (predicted_piece == target_froms).sum().item() +
                (predicted_move == target_to_squares).sum().item()
            )
            total_predictions += target_froms.size(0) * 2  # Since there are 2 predictions per sample (piece and move)

            # Clear unnecessary variables to save memory
            del position_bitboards, valid_pieces_masks, legal_moves, target_froms, target_to_squares
            del logits_pieces_flat, logits_moves_flat, predicted_piece, predicted_move
            torch.cuda.empty_cache()
            # print(f"Batch {i} finished in {time.time() - start_time:.4f} seconds")

        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        val_loss, val_accuracy = validate(model, val_dataloader, criterion)

        if scheduler is not None:
            if is_warming_up and warmup_scheduler:
                warmup_scheduler.step()
                if warmup_scheduler.last_epoch == warmup_scheduler.total_iters:
                    is_warming_up = False
                    # match the lr when the regular scheduler comes in
                    scheduler.base_lrs = [group['lr'] for group in optimizer.param_groups]
            else:
                scheduler.step()

        # if epoch % 10 == 0:
        torch.save(model.state_dict(), f"chess_model_{epoch}.pth")

        # if epoch % 10 == 0:
        # print(
        #     f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

def validate(model, val_dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for i, (position_bitboards, valid_pieces_masks, legal_moves, target_froms, target_to_squares) in enumerate(val_dataloader):
            position_bitboards = position_bitboards.to(DEVICE).float()
            valid_pieces_masks = valid_pieces_masks.to(DEVICE)
            legal_moves = legal_moves.to(DEVICE)
            target_froms = target_froms.to(DEVICE)
            target_to_squares = target_to_squares.to(DEVICE)

            logits_pieces_flat, logits_moves_flat = model(
                position_bitboards, valid_pieces_masks, legal_moves, inference=False
            )

            # Compute losses
            loss_piece = criterion(logits_pieces_flat, target_froms)
            loss_move = criterion(logits_moves_flat, target_to_squares)
            total_loss = loss_piece + loss_move

            running_loss += total_loss.item()

            predicted_piece = torch.argmax(logits_pieces_flat, dim=1)
            predicted_move = torch.argmax(logits_moves_flat, dim=1)

            # Count correct predictions
            correct_predictions += (
                (predicted_piece == target_froms).sum().item() +
                (predicted_move == target_to_squares).sum().item()
            )
            total_predictions += target_froms.size(0) * 2  # Since there are 2 predictions per sample (piece and move)

    avg_loss = running_loss / len(val_dataloader)
    accuracy = correct_predictions / total_predictions * 100
    return avg_loss, accuracy

def test(model, board):
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
    legal_moves = torch.zeros(len(list(board.legal_moves)), 3)
    valid_pieces_mask = torch.zeros(12 * 8 * 8)
    for i, legal_move in enumerate(list(board.legal_moves)):
        piece_type, from_square, to_square  = board.piece_at(legal_move.from_square).piece_type, legal_move.from_square, legal_move.to_square
        piece_type = piece_type - 1
        legal_moves[i] = torch.tensor((piece_type, from_square, to_square))

        valid_pieces_mask[(piece_type) * 64 + from_square] = 1

    # Create a single "batch" entry
    batch = [(position_bitboard, valid_pieces_mask, legal_moves, 0, 0)]

    # Use the custom collate function to prepare the batch
    position_bitboards, valid_pieces_masks, padded_legal_moves, _, _ = custom_collate_fn(
        batch)

    # Move tensors to the appropriate device
    position_bitboards = position_bitboards.to(DEVICE).float()
    valid_pieces_masks = valid_pieces_masks.to(DEVICE).float()
    padded_legal_moves = padded_legal_moves.to(DEVICE).long()

    # Perform inference with the model
    with torch.no_grad():  # No gradient tracking during inference
        from_square, to_square = model(
                    position_bitboards, valid_pieces_masks, padded_legal_moves, inference=True
                )

    # Validate the predicted move
    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        print(f"Predicted move: {board.san(move)}")
        return move
    else:
        print(f"Invalid move: {move}")

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


def load_model(model_path):
    model = ChessCNN().to(device=DEVICE)
    model.load_state_dict(torch.load(model_path))
    return model


def play(model):
    board = chess.Board()
    print(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = test(model, board)
            if move:
                board.push(move)
            else:
                exit()
            print(board)
        else:
            move = input("Enter move: ")
            move = chess.Move.from_uci(move)
            if move:
                board.push(move)
            print(board)


def clear_models():
    files_to_delete = glob.glob("chess_model_*.pth")

    # Delete each file
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}. Error: {e}")

def train_model(data_path, load=None, model_path=None, epochs=10, val_split=0.1):
    dataset = load_dataset(data_path)
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Split into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=12,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=12,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    print("Dataset loaded")
    model = ChessCNN(dropout_prob=0.4).to(device=DEVICE)
    # this could be an issue
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # try adam vs adamw
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
    # Reduces learning rate by a factor of 1/2 every 10 epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=len(train_dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    if load is None:
        train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, alpha=0.5, beta=0.5, scheduler=scheduler, warmup_scheduler=warmup_scheduler)

    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model

def main():
    file = 'chess_dataset.pth'


    # TODO: Label the data with percentages of best moves (like first, second, third best) with stockfish
    # Also consider using black moves and getting better training data
    # pad the legal moves with -1 here so i dont need custom collate fn
    # get the top five pieces to move by % then get the top 5 moves for each piece by % and use those as the targets
    # this also means itll prioritize choosing the right piece fist and then the right move (with the loss scaling i have currently)



    # if not os.path.isdir(DATABASE_DIR):
    #     raise Exception(
    #         'Lichess Elite Database directory not found. Please download the dataset from https://database.nikonoel.fr/ and extract it to the root directory of this project.')
    # files = os.listdir(DATABASE_DIR)
    # files = [file for file in files if file[0] != '1']

    # dataset = retrieve_dataset(files, 200)
    # save_dataset(dataset, file)

    # clear_models()
    # model = train_model(file,
    #                     model_path='chess_model.pth', epochs=30)

    model = load_model('chess_model_20.pth')

    # mate in 1 correct is e2h5
    m1 = chess.Board("r2qkbnr/1b5p/p1n2p2/1pN1pNp1/1P1p4/1Q4P1/PBPPBP1P/R3K2R")

    # m1 = chess.Board("3r4/1K6/2Nb4/2kb4/8/8/3PB3/8 w - - 0 1")

    # mate in 3, correct is g1g7
    m3 = chess.Board("r3k3/1p1p1ppp/1p1B4/1p6/Pp6/P3P3/8/4K1QR")

    # print(engine.play(b, chess.engine.Limit(time=0.0001)).move)

    # engine.quit()

    # print(m1)
    # print(test(model, m1))
    # print(m3)
    # print(test(model, m3))

    # start = chess.Board()
    # print(start)
    # print(test(model, start))

    play(model)

if __name__ == "__main__":
    main()

# 5e-4

# 3 resblocks
# Epoch [1/20], Loss: 2.6255, Accuracy: 23.33%, Validation Loss: 8.4826, Validation Accuracy: 28.66%
# Epoch [2/20], Loss: 2.4276, Accuracy: 33.75%, Validation Loss: 8.1491, Validation Accuracy: 28.93%
# Epoch [3/20], Loss: 2.3822, Accuracy: 37.71%, Validation Loss: 8.1080, Validation Accuracy: 30.96%
# Epoch [4/20], Loss: 2.3640, Accuracy: 40.46%, Validation Loss: 8.0637, Validation Accuracy: 32.86%
# Epoch [5/20], Loss: 2.3467, Accuracy: 43.25%, Validation Loss: 7.8327, Validation Accuracy: 35.81%
# Epoch [6/20], Loss: 2.3179, Accuracy: 46.35%, Validation Loss: 7.9954, Validation Accuracy: 35.64%
# Epoch [7/20], Loss: 2.2770, Accuracy: 49.58%, Validation Loss: 7.8962, Validation Accuracy: 36.37%
# Epoch [8/20], Loss: 2.2212, Accuracy: 53.24%, Validation Loss: 7.9540, Validation Accuracy: 36.80%
# Epoch [9/20], Loss: 2.1555, Accuracy: 57.12%, Validation Loss: 7.8916, Validation Accuracy: 37.61%
# Epoch [10/20], Loss: 2.0861, Accuracy: 60.86%, Validation Loss: 7.8199, Validation Accuracy: 39.39%
# Epoch [11/20], Loss: 2.0196, Accuracy: 64.36%, Validation Loss: 7.8232, Validation Accuracy: 39.40%
# Epoch [12/20], Loss: 1.9684, Accuracy: 67.19%, Validation Loss: 7.9221, Validation Accuracy: 39.39%
# Epoch [13/20], Loss: 1.9191, Accuracy: 69.76%, Validation Loss: 7.9118, Validation Accuracy: 39.40%
# Epoch [14/20], Loss: 1.8831, Accuracy: 71.68%, Validation Loss: 8.0144, Validation Accuracy: 40.20%
# Epoch [15/20], Loss: 1.8628, Accuracy: 72.89%, Validation Loss: 7.9251, Validation Accuracy: 39.85%
# Epoch [16/20], Loss: 1.8460, Accuracy: 73.77%, Validation Loss: 7.9829, Validation Accuracy: 39.26%


# 1 resblock
# Epoch [1/20], Loss: 2.5773, Accuracy: 24.34%, Validation Loss: 8.3636, Validation Accuracy: 28.39%
# Epoch [2/20], Loss: 2.4220, Accuracy: 33.78%, Validation Loss: 8.2665, Validation Accuracy: 29.01%
# Epoch [3/20], Loss: 2.3963, Accuracy: 36.94%, Validation Loss: 8.2862, Validation Accuracy: 29.05%
# Epoch [4/20], Loss: 2.3861, Accuracy: 39.40%, Validation Loss: 8.1782, Validation Accuracy: 31.71%
# Epoch [5/20], Loss: 2.3786, Accuracy: 41.71%, Validation Loss: 8.1406, Validation Accuracy: 32.41%
# Epoch [6/20], Loss: 2.3638, Accuracy: 44.23%, Validation Loss: 8.0771, Validation Accuracy: 33.78%
# Epoch [7/20], Loss: 2.3371, Accuracy: 47.09%, Validation Loss: 8.0079, Validation Accuracy: 34.88%
# Epoch [8/20], Loss: 2.2957, Accuracy: 50.33%, Validation Loss: 7.9313, Validation Accuracy: 36.03%
# Epoch [9/20], Loss: 2.2444, Accuracy: 53.78%, Validation Loss: 7.8821, Validation Accuracy: 37.20%
# Epoch [10/20], Loss: 2.1871, Accuracy: 57.25%, Validation Loss: 7.9604, Validation Accuracy: 37.83%
# Epoch [11/20], Loss: 2.1312, Accuracy: 60.48%, Validation Loss: 7.8636, Validation Accuracy: 37.92%
# Epoch [12/20], Loss: 2.0806, Accuracy: 63.43%, Validation Loss: 7.8952, Validation Accuracy: 38.29%
# Epoch [13/20], Loss: 2.0400, Accuracy: 65.84%, Validation Loss: 7.8911, Validation Accuracy: 39.33%
# Epoch [14/20], Loss: 2.0074, Accuracy: 67.76%, Validation Loss: 8.0027, Validation Accuracy: 38.33%

# 5 resblocks, 1e-4 lr (PLAYS REASONABLE MOVES)
# Epoch [1/20], Loss: 2.6275, Accuracy: 23.75%, Validation Loss: 7.7489, Validation Accuracy: 28.54%
# Epoch [2/20], Loss: 2.4743, Accuracy: 31.85%, Validation Loss: 8.2004, Validation Accuracy: 27.41%
# Epoch [3/20], Loss: 2.4128, Accuracy: 35.98%, Validation Loss: 7.9005, Validation Accuracy: 31.07%
# Epoch [4/20], Loss: 2.3862, Accuracy: 39.16%, Validation Loss: 8.2618, Validation Accuracy: 30.31%
# Epoch [5/20], Loss: 2.3661, Accuracy: 42.10%, Validation Loss: 7.7590, Validation Accuracy: 34.47%
# Epoch [6/20], Loss: 2.3420, Accuracy: 45.04%, Validation Loss: 8.1338, Validation Accuracy: 33.11%
# Epoch [7/20], Loss: 2.3091, Accuracy: 48.30%, Validation Loss: 7.5937, Validation Accuracy: 37.09%
# Epoch [8/20], Loss: 2.2668, Accuracy: 51.66%, Validation Loss: 7.7802, Validation Accuracy: 37.76%
# Epoch [9/20], Loss: 2.2142, Accuracy: 55.15%, Validation Loss: 7.6513, Validation Accuracy: 39.06%
# Epoch [10/20], Loss: 2.1580, Accuracy: 58.51%, Validation Loss: 7.7086, Validation Accuracy: 39.32%
# Epoch [11/20], Loss: 2.0944, Accuracy: 61.96%, Validation Loss: 7.8723, Validation Accuracy: 39.93%
# Epoch [12/20], Loss: 2.0370, Accuracy: 64.95%, Validation Loss: 7.8590, Validation Accuracy: 39.87%
# Epoch [13/20], Loss: 1.9821, Accuracy: 67.66%, Validation Loss: 7.8888, Validation Accuracy: 40.19%
# Epoch [14/20], Loss: 1.9352, Accuracy: 69.88%, Validation Loss: 7.8891, Validation Accuracy: 39.59%
# Epoch [15/20], Loss: 1.8925, Accuracy: 71.72%, Validation Loss: 7.9921, Validation Accuracy: 39.90%
# Epoch [16/20], Loss: 1.8553, Accuracy: 73.21%, Validation Loss: 7.8585, Validation Accuracy: 40.25%
# Epoch [17/20], Loss: 1.8299, Accuracy: 74.20%, Validation Loss: 8.0914, Validation Accuracy: 39.73%
# Epoch [18/20], Loss: 1.8188, Accuracy: 74.55%, Validation Loss: 8.1485, Validation Accuracy: 39.71%
# Epoch [19/20], Loss: 1.8212, Accuracy: 74.40%, Validation Loss: 8.0358, Validation Accuracy: 39.19%
# Epoch [20/20], Loss: 1.8399, Accuracy: 73.77%, Validation Loss: 7.9859, Validation Accuracy: 39.22%