from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position_bitboard, valid_pieces_mask, legal_moves, target_from, target_to_square = self.data[
            idx]
        return position_bitboard, valid_pieces_mask, legal_moves, target_from, target_to_square
