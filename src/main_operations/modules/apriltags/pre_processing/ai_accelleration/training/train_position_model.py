import json
import os
from typing import Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.onnx
from cv2 import imread
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.position_detectors.predictor import (
    PositionPredictor,
)
from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    LetterboxTransform,
)


def calculate_tag_center_and_scale(
    corners: list, img_width: int, img_height: int
) -> Tuple[float, float, float]:
    """Calculate the center and scale of an AprilTag from its corners.

    Args:
        corners: List of 4 corner points with x,y coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (center_x_norm, center_y_norm, scale_norm)
    """
    xs = [corner["x"] for corner in corners]
    ys = [corner["y"] for corner in corners]

    # Calculate bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Scale as the larger dimension of the bounding box
    scale = max(max_x - min_x, max_y - min_y)

    # Normalize to [0,1]
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    scale_norm = scale / max(
        img_width, img_height
    )  # Normalize by larger image dimension

    return center_x_norm, center_y_norm, scale_norm


class PositionDataset(Dataset):
    """Dataset of raw frames and position targets for AprilTag detection with preloaded data."""

    JSON_EXTENSION = ".json"

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose,
        max_detections: int = 12,
        target_width: int = 320,
        target_height: int = 320,
    ):
        """Initialize the PositionDataset with preloaded data.

        Args:
            data_dir (str): The directory containing the training data.
            transform (transforms.Compose): The transform to apply to the images.
            max_detections (int): Maximum number of detections per image.
            target_width (int): Target width for processed images.
            target_height (int): Target height for processed images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_detections = max_detections
        self.target_width = target_width
        self.target_height = target_height

        # Get all valid base names
        files = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".png"))
        bases = [
            b
            for b in files
            if os.path.isfile(os.path.join(data_dir, b + self.JSON_EXTENSION))
        ]

        # Preload all images and labels into memory
        self.images = []
        self.labels = []

        print(f"Preloading {len(bases)} training samples into RAM...")
        for base in tqdm(bases, desc="Loading data", unit="sample"):
            # Load and process image
            img = imread(os.path.join(data_dir, base + ".png"))
            img_t = self.transform(img)
            img_t = cast(torch.Tensor, img_t)
            self.images.append(img_t)

            # Load and process JSON labels
            with open(os.path.join(data_dir, base + self.JSON_EXTENSION), "r") as jf:
                data = json.load(jf)
                label = self._process_ground_truth(data)
                self.labels.append(label)

        print(f"Successfully loaded {len(self.images)} samples into RAM")

    def _process_ground_truth(self, data: dict) -> torch.Tensor:
        """Process JSON ground truth into position targets.

        Args:
            data: JSON data containing frame info and tags

        Returns:
            torch.Tensor: Shape (max_detections, 4) with [x_norm, y_norm, scale_norm, confidence]
        """
        img_width = data["image_width"]
        img_height = data["image_height"]
        tags = data["tags"]

        # Initialize targets tensor with zeros (no detections)
        targets = torch.zeros(self.max_detections, 4, dtype=torch.float32)

        # Compute centers and sort tags top-to-bottom for deterministic ordering
        sortable_tags = []
        for tag in tags:
            corners = tag["corners"]
            center_x_norm, center_y_norm, scale_norm = calculate_tag_center_and_scale(
                corners, img_width, img_height
            )
            sortable_tags.append((center_x_norm, center_y_norm, scale_norm))
        sortable_tags.sort(key=lambda t: t[1])

        # Fill targets up to max_detections without per-row tensor allocations
        for i, (cx, cy, sc) in enumerate(sortable_tags[: self.max_detections]):
            targets[i, 0] = float(cx)
            targets[i, 1] = float(cy)
            targets[i, 2] = float(sc)
            targets[i, 3] = 1.0

        return targets

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed image and position targets.
        """
        return self.images[idx], self.labels[idx]


def build_grid_targets(
    targets_batch: torch.Tensor,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-image grid target maps from per-detection normalized labels.

    Args:
        targets_batch: Tensor of shape (B, max_detections, 4) with [x_norm, y_norm, scale_norm, conf]
        grid_h: Output grid height (Gh)
        grid_w: Output grid width (Gw)
        device: Torch device for target tensors

    Returns:
        Tuple of (obj_map, dx_map, dy_map, ds_map), each of shape (B, Gh, Gw)
        obj_map uses label smoothing: 0.95 for positives, 0.05 for negatives
    """
    batch_size = targets_batch.size(0)
    obj_map = torch.full(
        (batch_size, grid_h, grid_w), 0.05, device=device, dtype=torch.float32
    )  # Label smoothing: negative class
    dx_map = torch.zeros_like(obj_map)
    dy_map = torch.zeros_like(obj_map)
    ds_map = torch.zeros_like(obj_map)

    # Assume model input is square W=H. Targets are normalized 0..1 already
    # ds_target = log(scale_pixels / cell_w) with scale_pixels = scale_norm * W, cell_w = W / Gw => log(scale_norm * Gw)
    for b in range(batch_size):
        sample = targets_batch[b]
        valid_mask = sample[:, 3] > 0.5
        if not torch.any(valid_mask):
            continue
        sample_valid = sample[valid_mask]
        gx = sample_valid[:, 0].clamp(0.0, 0.9999)
        gy = sample_valid[:, 1].clamp(0.0, 0.9999)
        sc = sample_valid[:, 2].clamp(min=1e-6)

        j = torch.floor(gx * grid_w).to(torch.long)
        i = torch.floor(gy * grid_h).to(torch.long)

        dx = gx * grid_w - j.float()
        dy = gy * grid_h - i.float()
        ds = torch.log(sc * grid_w)

        # In case of collisions, last one wins (acceptable for sparse tags)
        obj_map[b, i, j] = 0.95  # Label smoothing: positive class
        dx_map[b, i, j] = dx
        dy_map[b, i, j] = dy
        ds_map[b, i, j] = ds

    return obj_map, dx_map, dy_map, ds_map


# ——— Config ———
data_dir = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/training_data"
epochs = 500
batch_size = 36
lr = 1e-3
output = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/position_model.pth"

target_width = 320
target_height = 320
max_detections = 12

early_stopping_patience = 10
early_stopping_min_delta = 2e-3


def train() -> None:
    """Train the position predictor with early stopping and export the best model.

    Trains the model on preloaded dataset, applies early stopping based on epoch
    average loss with configurable patience and minimum improvement, saves only
    the best-performing weights, and exports the best model to ONNX.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tf = transforms.Compose(
        [
            LetterboxTransform((target_width, target_height)),
            transforms.ToTensor(),
        ]
    )

    dataset = PositionDataset(
        data_dir=data_dir,
        transform=tf,
        max_detections=max_detections,
        target_width=target_width,
        target_height=target_height,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,  # Increased from 4 to leverage all CPU cores
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch 4 batches per worker
    )

    # Instantiate model (grid size is derived by backbone stride; for 320x320 -> ~20x20)
    model = PositionPredictor()
    model = model.to(device)

    # Loss functions
    bce_logits = nn.BCEWithLogitsLoss()
    l1_loss = nn.SmoothL1Loss(reduction="none")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []

    best_epoch_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        model.train()
        running_loss = 0.0

        for imgs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False
        ):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)  # (B, 4, Gh, Gw)
            obj_logits = outputs[:, 0, ...]
            dx_hat = outputs[:, 1, ...]
            dy_hat = outputs[:, 2, ...]
            ds_hat = outputs[:, 3, ...]

            _, grid_h, grid_w = obj_logits.shape

            # Build target maps for this batch
            obj_map, dx_map, dy_map, ds_map = build_grid_targets(
                targets, grid_h, grid_w, device
            )

            # Losses
            obj_loss = bce_logits(obj_logits, obj_map)

            positive_mask = obj_map > 0.5
            num_pos = positive_mask.sum().clamp(min=1)

            dx_l = (
                l1_loss(
                    torch.sigmoid(dx_hat)[positive_mask], dx_map[positive_mask]
                ).sum()
                / num_pos
            )
            dy_l = (
                l1_loss(
                    torch.sigmoid(dy_hat)[positive_mask], dy_map[positive_mask]
                ).sum()
                / num_pos
            )
            ds_l = l1_loss(ds_hat[positive_mask], ds_map[positive_mask]).sum() / num_pos

            # Weights can be tuned; start equal
            loss = obj_loss + dx_l + dy_l + ds_l

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f}")

        if (best_epoch_loss - avg_loss) > early_stopping_min_delta:
            best_epoch_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)."
                )
                break

    print(f"Best model saved to {output}")

    # Export to ONNX
    onnx_output_path = (
        output.replace(".pt", ".onnx") if output.endswith(".pt") else output + ".onnx"
    )
    dummy_input = torch.randn(1, 1, target_height, target_width).to(device)
    try:
        state_dict = torch.load(output, map_location=device)
        model.load_state_dict(state_dict)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )
        print(f"Model successfully exported to ONNX format at {onnx_output_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
