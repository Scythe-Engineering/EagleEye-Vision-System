import torch
import torch.nn as nn


class PositionPredictor(nn.Module):
    """Lightweight single-stage anchor-free detector for AprilTag position detection."""

    def __init__(self, grid_size: int = 40, max_detections: int = 12):
        """Initialize the position predictor.

        Args:
            grid_size: Size of the output grid (grid_size x grid_size).
            max_detections: Maximum number of detections to return.
        """
        super().__init__()
        self.grid_size = grid_size
        self.max_detections = max_detections

        # Lightweight backbone with progressive channel increase
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, H/2, W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/4, W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/8, W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # (B, 128, H/16, W/16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Detection head: 1x1 conv producing 4 channels per grid cell
        # [obj_logits, dx_hat, dy_hat, ds_hat]
        self.head = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detector.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Tensor of shape (B, 4, grid_size, grid_size) containing
            [obj_logits, dx_hat, dy_hat, ds_hat] in NCHW layout to avoid costly transposes
        """
        features = self.backbone(x)
        output = self.head(features)  # (B, 4, grid_size, grid_size)

        # previous code # return output.permute(0, 2, 3, 1)
        # commensing code #
        return output
