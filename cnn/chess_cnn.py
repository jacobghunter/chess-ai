import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from utilities import bitboard_to_board

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels do not match, use a 1x1 convolution to match the dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()  # If the channels are already equal, identity shortcut

    def forward(self, x):
        identity = self.shortcut(x)  # Apply the shortcut transformation (1x1 convolution if necessary)
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
        self.resblock4 = SimpleResidualBlock(512, 512, 3)
        self.resblock5 = SimpleResidualBlock(512, 512, 3)
        self.resblock6 = SimpleResidualBlock(512, 512, 3)
        self.resblock7 = SimpleResidualBlock(512, 512, 3)
        self.resblock8 = SimpleResidualBlock(512, 512, 3)
        self.resblock9 = SimpleResidualBlock(512, 512, 3)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout_fc = nn.Dropout(p=dropout_prob)

        # Separate outputs for piece selection and move selection
        self.fc_piece = nn.Linear(512, 12 * 8 * 8)  # Piece selection logits
        self.fc_move = nn.Linear(512, 8 * 8)  # Move selection logits

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_piece.weight)
        nn.init.xavier_uniform_(self.fc_move.weight)

    def forward(self, bitboards, flat_valid_pieces_masks, legal_moves, current_epoch=1, total_epochs=2, inference=False):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(bitboards)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        # Outputs for piece and move selection
        fc_piece = self.fc_piece(x)
        fc_move = self.fc_move(x)

        max_invalid_penalty = -1e1
        min_invalid_penalty = -1e0
        scaling_factor = max(0.1, 1 - current_epoch / total_epochs)
        dynamic_penalty = max_invalid_penalty * scaling_factor + min_invalid_penalty * (1 - scaling_factor)

        if inference:
            piece_softmax = torch.where(flat_valid_pieces_masks.bool(), fc_piece, float('-inf'))
            piece_softmax = F.softmax(piece_softmax, dim=1)
            from_square = torch.argmax(piece_softmax.view(-1, 12, 8 * 8).sum(dim=1).flatten(1), dim=1)  # Get the best move (from square)

            valid_moves_mask = self.generate_valid_moves_mask(from_square, legal_moves).to(x.device)
            move_softmax = torch.where(valid_moves_mask.bool(), fc_move, float('-inf'))
            move_softmax = F.softmax(move_softmax, dim=1)
            to_square = torch.argmax(move_softmax, dim=1)

            return from_square, to_square
        else:
            piece_softmax = torch.where(flat_valid_pieces_masks.bool(), fc_piece, float('-inf'))
            piece_softmax = F.softmax(piece_softmax, dim=1)
            from_square = torch.argmax(piece_softmax.view(-1, 12, 8 * 8).sum(dim=1).flatten(1), dim=1)  # Get the best move (from square)

            fc_piece = torch.where(flat_valid_pieces_masks.bool(), fc_piece, dynamic_penalty)
            
            # Generate and apply valid moves mask
            valid_moves_mask = self.generate_valid_moves_mask(from_square, legal_moves).to(x.device)
            fc_move = torch.where(valid_moves_mask.bool(), fc_move, dynamic_penalty)

            del from_square 
            del valid_moves_mask
            del piece_softmax

            return fc_piece, fc_move


    def generate_valid_moves_mask(self, piece_square, legal_moves):
        """
        Generate a mask for valid moves from a given piece square in a batch.
        
        Parameters:
            piece_square (torch.Tensor): Tensor of shape [batch_size] containing the square of the selected piece (0-63).
            legal_moves (torch.Tensor): Tensor of shape [batch_size, max_legal_moves, 3], 
                                        where each entry is (piece_type, from_square, to_square), padded with -1.
                                        
        Returns:
            torch.Tensor: Tensor of shape [batch_size, 8 * 8] with 1s for valid moves and 0s elsewhere.
        """
        batch_size = piece_square.size(0)
        max_legal_moves = legal_moves.size(1)
        
        # Create an empty mask of shape [batch_size, 8 * 8]
        valid_moves_mask = torch.zeros((batch_size, 8 * 8), device=legal_moves.device, dtype=torch.int)

        # Reshape the legal_moves tensor for easier indexing
        legal_moves_flat = legal_moves.view(batch_size * max_legal_moves, 3)

        # Extract from squares and to squares
        from_squares = legal_moves_flat[:, 1]
        to_squares = legal_moves_flat[:, 2]

        # Create a mask for valid legal moves (exclude padded entries with -1)
        valid_mask = (from_squares != -1)

        # Filter moves where the from_square matches the piece_square
        batch_indices = torch.arange(batch_size, device=legal_moves.device).repeat_interleave(max_legal_moves)
        match_mask = (from_squares == piece_square[batch_indices]) & valid_mask

        # Filter valid moves and their respective batch indices
        valid_batch_indices = batch_indices[match_mask]
        valid_to_squares = to_squares[match_mask]

        # Compute indices for the 8 * 8 mask (to_squares map directly to the 8 * 8 board)
        move_indices = valid_to_squares  # Direct mapping to 8 * 8 board

        # Set the valid moves in the mask
        valid_moves_mask[valid_batch_indices, move_indices] = 1

        del legal_moves_flat
        del from_squares
        del to_squares
        del valid_mask
        del batch_indices
        del match_mask
        del valid_batch_indices
        del valid_to_squares

        return valid_moves_mask