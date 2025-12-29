import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Trọng số cho class hiếm (thường set là 0.25 nếu Spoof ít hơn)
            gamma: Độ tập trung vào mẫu khó (thường là 2.0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross Entropy loss cơ bản
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # Xác suất dự đoán đúng

        # Focal term: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Alpha weighting (nếu cần cân bằng class)
        if self.alpha is not None:
            # Tạo tensor alpha tương ứng với từng mẫu trong batch
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss