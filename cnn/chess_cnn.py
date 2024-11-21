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

        # 24 for both to and from move
        self.fc2 = nn.Linear(512, 24 * 8 * 8)

    def forward(self, x, available_moves_mask):
        # Process the board state through the convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the tensor after convolutional layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256 * 8 * 8)

        # Pass through the first fully connected layer
        x = torch.relu(self.fc1(x))

        # Generate logits for all possible "to" and "from" locations
        logits = self.fc2(x)

        # Reshape logits to (batch_size, 24, 8, 8)
        logits = logits.view(-1, 24, 8, 8)

        # Split the logits into "to" and "from" predictions
        logits_to = logits[:, :12, :, :]
        logits_from = logits[:, 12:, :, :]

        # Split the available moves mask into "to" and "from" masks
        available_moves_to = available_moves_mask[:, :12, :, :]
        available_moves_from = available_moves_mask[:, 12:, :, :]

        # Apply the masks to enforce valid moves
        masked_logits_to = logits_to * available_moves_to
        masked_logits_from = logits_from * available_moves_from

        return masked_logits_to, masked_logits_from
