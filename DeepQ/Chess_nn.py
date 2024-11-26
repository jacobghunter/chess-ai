import torch
import torch.nn as nn
# Enable Metal for Mac


if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(f"Using device: {device}")
class ChessNN(nn.Module):
  def __init__(self):
      super(ChessNN, self).__init__()

      # Convolutional layers with batch norm
      self.conv_layers = nn.Sequential(
          nn.Conv2d(12, 256, kernel_size=3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()
      )

      # Fully connected layers
      self.fc_layers = nn.Sequential(
          nn.Linear(512 * 8 * 8, 2048),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(1024, 4672)  # Output size for all possible moves
      )

      # Move model to appropriate device
      self.to(device)

  def forward(self, x):
      x = self.conv_layers(x)
      x = x.view(-1, 512 * 8 * 8)
      x = self.fc_layers(x)
      return x
