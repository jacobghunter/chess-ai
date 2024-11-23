import torch


def custom_collate_fn(batch):
    position_bitboards = []
    padded_legal_moves = []
    target_froms = []
    target_tos = []

    max_legal_moves = max(len(item[1]) for item in batch)

    for position_bitboard, legal_moves, target_from, target_to in batch:
        position_bitboards.append(position_bitboard)

        # Pad legal_moves to match the max length in the batch
        padded_legal_moves.append(
            legal_moves + [0] * (max_legal_moves - len(legal_moves))
        )
        target_froms.append(target_from)
        target_tos.append(target_to)

    # Convert to tensors
    position_bitboards = torch.stack(position_bitboards)
    padded_legal_moves = torch.tensor(padded_legal_moves, dtype=torch.long)
    target_froms = torch.tensor(target_froms, dtype=torch.long)
    target_tos = torch.tensor(target_tos, dtype=torch.long)

    return position_bitboards, padded_legal_moves, target_froms, target_tos
