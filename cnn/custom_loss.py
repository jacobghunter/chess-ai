import torch
import torch.nn as nn


class CustomMoveLoss(nn.Module):
    def __init__(self, to_weight=0.5, from_weight=0.5):
        super(CustomMoveLoss, self).__init__()
        self.to_weight = to_weight
        self.from_weight = from_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits_to, logits_from, target_to, target_from):
        # Calculate loss for "to" and "from" predictions
        loss_to = self.cross_entropy(logits_to, target_to)
        loss_from = self.cross_entropy(logits_from, target_from)

        # Weighted combination of both losses
        total_loss = self.to_weight * loss_to + self.from_weight * loss_from
        return total_loss
