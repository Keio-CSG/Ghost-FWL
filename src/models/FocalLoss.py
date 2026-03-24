from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Union[float, Sequence[float], torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, Sequence):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            raise ValueError(f"Invalid type for alpha: {type(alpha)}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.alpha = self.alpha.to(logits.device)
        ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        if self.alpha.shape[0] == 1:
            alpha = self.alpha[0]
        else:
            valid_mask = targets != self.ignore_index
            alpha = torch.ones_like(targets, dtype=torch.float32, device=logits.device)
            if valid_mask.any():
                alpha[valid_mask] = self.alpha[targets[valid_mask]]

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        return focal_loss


if __name__ == "__main__":
    logits = torch.randn(4, 5, 32, 64, 64)  # N, C, D, H, W
    targets = torch.randint(0, 5, (4, 32, 64, 64))  # N, D, H, W
    loss = FocalLoss(alpha=[0.1, 0.2, 0.3, 0.4, 0.5])
    loss_value = loss(logits, targets)
    print(loss_value)
