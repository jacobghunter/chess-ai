import torch


def custom_collate_fn(batch):
    position_bitboards = []
    max_legal_moves = max(len(item[1]) for item in batch)
    
    # Create tensors for the batched data
    position_bitboards_tensor = torch.zeros(len(batch), *batch[0][0].shape, dtype=torch.float32)
    padded_legal_moves_tensor = torch.zeros(len(batch), max_legal_moves, dtype=torch.long)
    target_froms_tensor = torch.zeros(len(batch), dtype=torch.long)
    target_tos_tensor = torch.zeros(len(batch), dtype=torch.long)
    
    for i, (position_bitboard, legal_moves, target_from, target_to) in enumerate(batch):
        # Position bitboards are appended directly into the preallocated tensor
        position_bitboards_tensor[i] = position_bitboard
        
        # Pad legal moves directly into the tensor
        padded_legal_moves_tensor[i, :len(legal_moves)] = torch.tensor(legal_moves, dtype=torch.long)
        
        # Targets are appended directly into the tensors
        target_froms_tensor[i] = target_from
        target_tos_tensor[i] = target_to

    return position_bitboards_tensor, padded_legal_moves_tensor, target_froms_tensor, target_tos_tensor
