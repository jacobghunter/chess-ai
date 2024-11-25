import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import decode_piece_from_to, decode_piece_from_to_tensor, tensor_square_to_chess_square

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ChessCNN(nn.Module):
    def __init__(self, input_channels=12, dropout_prob=0.5):
        super(ChessCNN, self).__init__()

        # Initial layer
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.resblock1 = SimpleResidualBlock(512, 512, 3)
        self.resblock2 = SimpleResidualBlock(512, 512, 3)
        self.resblock3 = SimpleResidualBlock(512, 512, 3)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout_fc = nn.Dropout(p=dropout_prob)

        # Separate outputs for piece selection and move selection
        self.fc_piece = nn.Linear(512, 12 * 8 * 8)  # Piece selection logits
        self.fc_move = nn.Linear(512, 8 * 8)  # Move selection logits

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc_piece.weight)
        # nn.init.xavier_uniform_(self.fc_move.weight)

    def forward(self, bitboard, legal_moves, inference=False):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(bitboard)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        # Outputs for piece and move selection
        logits_piece = self.fc_piece(x).view(-1, 12, 8, 8)
        logits_move = self.fc_move(x).view(-1, 8, 8)

        if inference:
            legal_moves = decode_piece_from_to_tensor(legal_moves).to(x.device)

            # Normalized to ensure it chooses a real piece
            norm_logits_piece = torch.softmax(logits_piece.view(-1), dim=0).view(logits_piece.shape)

            # Mask logits to ensure only valid pieces are considered
            valid_pieces_mask = self.generate_valid_pieces_mask(legal_moves).to(x.device)
            # print("Valid pieces mask:", valid_pieces_mask)
            # print("Any NaN or inf values in valid_pieces_mask:", torch.any(torch.isnan(valid_pieces_mask)), torch.any(torch.isinf(valid_pieces_mask)))

            norm_logits_piece = norm_logits_piece * valid_pieces_mask

            logits_flat = norm_logits_piece.view(-1, 12 * 8 * 8)
            logits_summed = logits_flat.view(-1, 12, 8, 8).sum(dim=1)
            from_square = torch.argmax(logits_summed.view(-1, 8 * 8), dim=1)

            # Mask logits_move to ensure only valid moves are considered
            valid_moves_mask = self.generate_valid_moves_mask(from_square, legal_moves).to(x.device)
            norm_logits_move = torch.softmax(logits_move.view(-1), dim=0).view(logits_move.shape)
            norm_logits_move = norm_logits_move * valid_moves_mask

            return norm_logits_piece, norm_logits_move
        else:
            return logits_piece, logits_move

    def generate_valid_pieces_mask(self, legal_moves):
        # Initialize an empty mask with shape [batch_size, 12, 8, 8]
        batch_size = len(legal_moves)
        valid_pieces_mask = torch.zeros(
            batch_size, 12, 8, 8, device=legal_moves[0][0].device)

        # Create a tensor of piece types and from_square indices
        # Flatten the legal_moves for batch processing
        all_piece_types, all_from_squares, batch_indices = [], [], []
        for i, sample_moves in enumerate(legal_moves):
            piece_types, from_squares = zip(
                *[(pt, fs) for pt, fs, _ in sample_moves if pt > 0]
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

        chess_squares = [tensor_square_to_chess_square(
            square) for square in piece_square]

        # Iterate over all legal moves for the board
        all_to_squares = []
        batch_indices = []
        for i, moves in enumerate(legal_moves):
            to_squares = [ts for _, fs, ts in moves if fs == chess_squares[i]]

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

        return valid_moves_mask
