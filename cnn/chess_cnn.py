import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    def __init__(self, input_channels=12):
        super(ChessCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layers after the convolutional layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)

        # Output layer: 130 values representing the encoded best move (start square, end square, color, piece type)
        self.fc2 = nn.Linear(512, 130)

    def forward(self, x, available_moves):
        # Process the board state through the convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the tensor after convolutional layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256 * 8 * 8)

        # Pass through the first fully connected layer
        x = torch.relu(self.fc1(x))

        # Output layer to get logits for the best move (130 values)
        move_logits = self.fc2(x)

        return move_logits
