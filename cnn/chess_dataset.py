from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position_tensor, available_moves_tensor, best_move = self.data[idx]
        return position_tensor, available_moves_tensor, best_move
