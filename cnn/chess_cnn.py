import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import decode_piece_from_to, print_tensor_board


class ChessCNN(nn.Module):
    def __init__(self, input_channels=12):
        super(ChessCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)

        # Separate outputs for piece selection and move selection
        self.fc_piece = nn.Linear(512, 12 * 8 * 8)  # Piece selection logits
        self.fc_move = nn.Linear(512, 8 * 8)  # Move selection logits

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_piece.weight)
        nn.init.xavier_uniform_(self.fc_move.weight)

    def forward(self, bitboard, legal_moves):
        legal_moves = [decode_piece_from_to(move) for move in legal_moves]

        # TODO: choosing black pieces?
        # Convolutional layers
        x = torch.relu(self.conv1(bitboard))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))

        # Outputs for piece and move selection
        logits_piece = self.fc_piece(x).view(-1, 12, 8, 8)
        logits_move = self.fc_move(x).view(-1, 8, 8)

        # print(logits_piece[logits_piece > 0])
        # print(logits_move[logits_move > 0])

        # Mask logits to ensure only valid pieces are considered
        valid_pieces_mask = self.generate_valid_pieces_mask(
            legal_moves).to(x.device)
        logits_piece = logits_piece * valid_pieces_mask

        logits_flat = logits_piece.view(-1, 12 * 8 * 8)
        logits_summed = logits_flat.view(-1, 12, 8, 8).sum(dim=1)
        from_square = torch.argmax(logits_summed.view(-1, 8 * 8), dim=1)

        # Mask logits_move to ensure only valid moves are considered
        valid_moves_mask = self.generate_valid_moves_mask(
            from_square, legal_moves).to(x.device)
        logits_move = logits_move * valid_moves_mask  # Apply mask to enforce valid moves

        # TODO: its possible it returns None if it chooses a piece with no valid moves

        return logits_piece, logits_move

    def generate_valid_pieces_mask(self, legal_moves):
        # Initialize an empty mask with shape [batch_size, 12, 8, 8]
        batch_size = len(legal_moves)
        valid_pieces_mask = torch.zeros(
            batch_size, 12, 8, 8, device=legal_moves[0][0].device)

        # Create a tensor of piece types and from_square indices
        # Flatten the legal_moves for batch processing
        all_piece_types, all_from_squares = [], []
        batch_indices = []  # To store batch index for each move
        for i, sample_moves in enumerate(legal_moves):

            piece_types, from_squares = zip(
                *[(pt, fs) for pt, fs in zip(sample_moves[0], sample_moves[1]) if pt > 0]
            )

            # Append to the overall list of piece types and from squares
            all_piece_types.extend(piece_types)
            all_from_squares.extend(from_squares)

            # Also track the batch index for each move
            batch_indices.extend([i] * len(piece_types))

        # Create tensors from the lists of piece types and from squares
        all_piece_types = torch.tensor(
            all_piece_types, dtype=torch.long, device=valid_pieces_mask.device)
        all_from_squares = torch.tensor(
            all_from_squares, dtype=torch.long, device=valid_pieces_mask.device)
        batch_indices = torch.tensor(
            batch_indices, dtype=torch.long, device=valid_pieces_mask.device)

        # Convert from_square to row, col using integer division and modulo
        from_rows = 7 - (all_from_squares // 8)  # Integer division for row
        from_cols = all_from_squares % 8       # Modulo for column

        # Map the piece types to valid indices in the valid_pieces_mask
        # Adjust piece type to match mask index (0-11)
        piece_indices = all_piece_types - 1

        # Use advanced indexing to fill in the valid_pieces_mask with the appropriate values
        valid_pieces_mask[batch_indices,
                          piece_indices, from_rows, from_cols] = 1

        return valid_pieces_mask

    def generate_valid_moves_mask(self, piece_square, legal_moves):
        batch_size = len(legal_moves)
        valid_moves_mask = torch.zeros(
            batch_size, 8, 8, device=legal_moves[0][0].device)

        chess_squares = [self.indicies_to_square(
            square) for square in piece_square]

        # Iterate over all legal moves for the board
        all_to_squares = []
        batch_indices = []
        for i, moves in enumerate(legal_moves):
            to_squares = [ts for _, fs, ts in zip(
                moves[0], moves[1], moves[2]) if fs == chess_squares[i]]

            # Unpack "from" and "to" squares, if any moves are valid
            if to_squares:
                all_to_squares.extend(to_squares)
                batch_indices.extend([i] * len(to_squares))

        if all_to_squares:
            # Convert to tensors
            all_to_squares = torch.tensor(
                all_to_squares, dtype=torch.long, device=valid_moves_mask.device
            )
            batch_indices = torch.tensor(
                batch_indices, dtype=torch.long, device=valid_moves_mask.device
            )

            # Calculate row and column indices for the "to" squares
            to_rows = 7 - (all_to_squares // 8)
            to_cols = all_to_squares % 8

            # Use advanced indexing to mark valid moves in the mask
            valid_moves_mask[batch_indices, to_rows, to_cols] = 1

        # row = 7 - (piece_square[0] // 8)
        # col = piece_square[0] % 8
        # print(col, row)
        # print(valid_moves_mask[0])

        return valid_moves_mask

    def indicies_to_square(self, square):
        if torch.is_tensor(square):
            square = square.item()
        row = 7 - square // 8
        col = square % 8
        return chess.square(col, row)
