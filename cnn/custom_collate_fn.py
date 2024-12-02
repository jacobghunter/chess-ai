import torch


def custom_collate_fn(batch):
    position_bitboards = []
    max_legal_moves = max(item[2].size(0) for item in batch)  
    
    # Create tensors for the batched data
    position_bitboards_tensor = torch.zeros(len(batch), *batch[0][0].shape, dtype=torch.float16)
    valid_pieces_mask_tensor = torch.zeros(len(batch), 12 * 8 * 8, dtype=torch.int)
    # pad with -1s to show it was padding
    padded_legal_moves_tensor = torch.full(
        (len(batch), max_legal_moves, 3), -1, dtype=torch.int
    )  # Default to -1 for padding
    target_froms_tensor = torch.zeros(len(batch), dtype=torch.long)
    target_tos_tensor = torch.zeros(len(batch), dtype=torch.long)
    
    for i, (position_bitboard, valid_pieces_mask, legal_moves, target_from, target_to_square) in enumerate(batch):
        # Position bitboards are appended directly into the preallocated tensor
        position_bitboards_tensor[i] = position_bitboard
        valid_pieces_mask_tensor[i] = valid_pieces_mask
        
        # Pad legal moves directly into the tensor
        padded_legal_moves_tensor[i, :len(legal_moves)] = legal_moves
        
        # Targets are appended directly into the tensors
        target_froms_tensor[i] = target_from
        target_tos_tensor[i] = target_to_square

        del position_bitboard, valid_pieces_mask, legal_moves, target_from, target_to_square

    return position_bitboards_tensor, valid_pieces_mask_tensor, padded_legal_moves_tensor, target_froms_tensor, target_tos_tensor
