import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, criterion, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs, targets):
        # 适配形状
        if targets.dim() == 4 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        elif targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        hard_loss = self.criterion(student_outputs, targets)

        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

