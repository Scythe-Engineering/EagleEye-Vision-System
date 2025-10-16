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
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, H/2, W/2)
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
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Reduced downscaling: keep H/8, W/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to produce a fixed grid_size x grid_size feature map
        self.pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))

        # Detection head: 1x1 conv producing 4 channels per grid cell
        # [obj_logits, dx_hat, dy_hat, ds_hat]
        self.head = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detector.

        Args:
            x: Input image tensor of dtype uint8 in NHWC layout with 3 channels (0-255 RGB).

        Returns:
            Tensor of shape (B, 4, grid_size, grid_size) containing
            [obj_logits, dx_hat, dy_hat, ds_hat] in NCHW layout to avoid costly transposes
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(torch.float32).div_(255.0)
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        features = self.backbone(gray)
        features = self.pool(features)
        output = self.head(features)  # (B, 4, grid_size, grid_size)
        return output

    def decode(
        self, logits: torch.Tensor, conf_threshold: float = 0.5
    ) -> list[tuple[int, int, float, float, float, float]]:
        """Decode grid logits into a limited list of detections.

        Args:
            logits: Tensor with shape (4, H, W) or (B, 4, H, W) containing
                [obj_logits, dx_hat, dy_hat, ds_hat].
            conf_threshold: Minimum objectness (after sigmoid) to keep a cell.

        Returns:
            A list of up to max_detections tuples (grid_i, grid_j, dx, dy, ds, confidence)
            in descending confidence order. dx, dy are in [0,1] (after sigmoid), ds is raw (log-space).
        """
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim != 3 or logits.shape[0] != 4:
            return []

        obj_logits = logits[0]
        dx_hat = logits[1]
        dy_hat = logits[2]
        ds_hat = logits[3]
        obj_probs = torch.sigmoid(obj_logits)
        mask = obj_probs > conf_threshold
        if not torch.any(mask):
            return []

        scores = obj_probs[mask]
        indices = mask.nonzero(as_tuple=False)
        scores, order = torch.sort(scores, descending=True)
        indices = indices[order]
        dx_vals = torch.sigmoid(dx_hat[mask])[order]
        dy_vals = torch.sigmoid(dy_hat[mask])[order]
        ds_vals = ds_hat[mask][order]

        num_keep = min(self.max_detections, indices.shape[0])
        kept = []
        for k in range(num_keep):
            i, j = int(indices[k, 0].item()), int(indices[k, 1].item())
            kept.append(
                (
                    i,
                    j,
                    float(dx_vals[k].item()),
                    float(dy_vals[k].item()),
                    float(ds_vals[k].item()),
                    float(scores[k].item()),
                )
            )
        return kept
