import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, list):
            if len(alpha) != num_classes:
                raise ValueError("alpha length must match num_classes")
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            if alpha >= 1:
                raise ValueError("alpha must be less than 1 when passed as scalar")
            weights = torch.zeros(num_classes, dtype=torch.float32)
            weights[0] = alpha
            if num_classes > 1:
                weights[1:] = (1 - alpha) / (num_classes - 1)
            self.alpha = weights

    def forward(self, preds, labels):
        if preds.dim() == 1:
            preds = torch.stack([-preds, preds], dim=1)
        preds = preds.contiguous().view(-1, preds.size(-1))
        labels = labels.contiguous().view(-1)
        alpha = self.alpha.to(preds.device)
        probs = F.softmax(preds, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        log_probs = F.log_softmax(preds, dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
        weights = alpha.gather(0, labels) * torch.pow(1 - probs, self.gamma)
        loss = -weights * log_probs
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()



