import torch
from torch import nn


class GridPredictor(nn.Module):
    def __init__(self, grid_height: int, grid_width: int) -> None:
        """
        Initialize the GridPredictor.

        Args:
            grid_height (int): The height of the grid.
            grid_width (int): The width of the grid.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Conv2d(128, 1, kernel_size=1)
        self.pool10 = nn.AdaptiveAvgPool2d((grid_height, grid_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for grid prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, H, W) representing batch of greyscale images.

        Returns:
            torch.Tensor: Output tensor of shape (N, 10, 10) containing grid cell predictions.
        """
        x = self.features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.pool10(x)
        return x.squeeze(1)
