from typing import Optional
import torch.nn.functional as F
from torch import Tensor, nn
import torch

nn.CrossEntropyLoss

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This loss focuses training on hard, misclassified examples by down-weighting
    easy examples. Critical for improving recall on spoofed faces.

    Args:
        alpha: Weighting factor for class balance (higher = more weight on spoof class)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            inputs: Raw logits from model (batch_size, num_classes)
            targets: Ground truth labels (batch_size)
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Apply alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Final focal loss
        focal_loss = alpha_t * focal_term * ce_loss

        return focal_loss.mean()
