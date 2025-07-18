import torch
from torch import nn
from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    GRID_HEIGHT,
    GRID_WIDTH,
)


class GridPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Conv2d(16, 1, kernel_size=1)
        self.pool10 = nn.AdaptiveAvgPool2d((GRID_HEIGHT, GRID_WIDTH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for grid prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, 10, 10).
        """
        x = self.features(x)
        x = self.classifier(x)
        x = self.pool10(x)
        return x.squeeze(1)
