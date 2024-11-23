from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position_bitboard, legal_moves, target_from, target_to = self.data[
            idx]
        return position_bitboard, legal_moves, target_from, target_to
