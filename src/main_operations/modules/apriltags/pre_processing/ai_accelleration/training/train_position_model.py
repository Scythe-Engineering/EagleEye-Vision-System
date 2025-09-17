import copy
import json
import os
from typing import List, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import torch
import torch.onnx
from cv2 import imread
from line_profiler import profile
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


def dataloader_worker_init(worker_id: int) -> None:
    """Initialize DataLoader worker to avoid OpenCV thread contention.

    Args:
        worker_id (int): The worker process index.

    Returns:
        None
    """
    cv2.setNumThreads(0)


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
    """Dataset of raw frames and position targets for AprilTag detection with on-demand disk loading."""

    JSON_EXTENSION = ".json"

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose,
        max_detections: int = 12,
        target_width: int = 320,
        target_height: int = 320,
        preload_workers: Optional[int] = None,
    ):
        """Initialize the PositionDataset with on-demand loading.

        Args:
            data_dir (str): The directory containing the training data.
            transform (transforms.Compose): The transform to apply to the images.
            max_detections (int): Maximum number of detections per image.
            target_width (int): Target width for processed images.
            target_height (int): Target height for processed images.
            preload_workers (Optional[int]): Unused. Kept for API compatibility.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_detections = max_detections
        self.target_width = target_width
        self.target_height = target_height

        image_basenames = sorted(
            f[:-4] for f in os.listdir(data_dir) if f.endswith(".png")
        )
        self.sample_base_names: List[str] = [
            base
            for base in image_basenames
            if os.path.isfile(os.path.join(data_dir, base + self.JSON_EXTENSION))
        ]
        print(f"Found {len(self.sample_base_names)} samples in {data_dir}")

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
        return len(self.sample_base_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load an item from disk.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed image and position targets.
        """
        base_name = self.sample_base_names[idx]
        img = imread(os.path.join(self.data_dir, base_name + ".png"))
        img_tensor = cast(torch.Tensor, self.transform(img))
        with open(
            os.path.join(self.data_dir, base_name + self.JSON_EXTENSION), "r"
        ) as jf:
            data = json.load(jf)
            label_tensor = self._process_ground_truth(data)
        return img_tensor, label_tensor


@profile
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


class ExponentialMovingAverage:
    """Maintain an exponential moving average (EMA) of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = self._clone_model(model).to(next(model.parameters()).device)
        self.decay = decay
        self.ema.requires_grad_(False)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        model_copy = copy.deepcopy(model)
        for p in model_copy.parameters():
            p.requires_grad = False
        return model_copy

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for (name, ema_p), p in zip(
            self.ema.state_dict().items(), model.state_dict().values()
        ):
            if p.dtype.is_floating_point:
                ema_p.data.mul_(self.decay).add_(
                    p.data.to(ema_p.device), alpha=1.0 - self.decay
                )
            else:
                ema_p.data.copy_(p.data)


# ——— Config ———
data_dir = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/augmented_training_data"
epochs = 500
batch_size = 36
lr = 5e-3
output = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/position_model.pth"

target_width = 320
target_height = 320
max_detections = 12

early_stopping_patience = 10
early_stopping_min_delta = 5e-3


@profile
def train() -> None:
    """Train the position predictor with validation, EMA, and grid-based losses.

    Trains the model with a train/validation split, uses grid targets with label
    smoothing and BCE loss for objectness and L1 regression losses, applies EMA of
    weights, and uses cosine LR with warmup. Early stopping and checkpointing are
    based on validation loss.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
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

    # Split dataset into train/val
    val_ratio = 0.2
    num_total = len(dataset)
    num_val = max(1, int(num_total * val_ratio))
    num_train = num_total - num_val
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=dataloader_worker_init,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=dataloader_worker_init,
    )

    # Instantiate model
    model = PositionPredictor()
    model = model.to(device)

    # Loss functions
    l1_loss = nn.SmoothL1Loss(reduction="none")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    # Cosine LR with warmup
    total_steps = max(1, epochs * max(1, len(train_loader)))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265))).item()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA of weights
    ema = ExponentialMovingAverage(model, decay=0.999)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float("inf")
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

            # Build grid targets with label smoothing
            obj_map, dx_map, dy_map, ds_map = build_grid_targets(
                targets, grid_h, grid_w, device
            )

            # Losses
            obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                obj_logits, obj_map, reduction="mean"
            )

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

            loss = obj_loss + dx_l + dy_l + ds_l

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update and scheduler step per iteration
            ema.update(model)
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)

        avg_train_loss = running_loss / len(train_subset)
        train_losses.append(avg_train_loss)

        # Validation with EMA weights
        ema_model = ema.ema
        eval_model = ema_model.to(device)
        eval_model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = eval_model(imgs)
                obj_logits = outputs[:, 0, ...]
                dx_hat = outputs[:, 1, ...]
                dy_hat = outputs[:, 2, ...]
                ds_hat = outputs[:, 3, ...]

                _, grid_h, grid_w = obj_logits.shape
                obj_map, dx_map, dy_map, ds_map = build_grid_targets(
                    targets, grid_h, grid_w, device
                )

                obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    obj_logits, obj_map, reduction="mean"
                )
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
                ds_l = (
                    l1_loss(ds_hat[positive_mask], ds_map[positive_mask]).sum()
                    / num_pos
                )

                val_loss = obj_loss + dx_l + dy_l + ds_l
                val_running += val_loss.item() * imgs.size(0)

        avg_val_loss = val_running / len(val_subset)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping on validation loss
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save EMA weights as the best checkpoint
            torch.save(ema.ema.state_dict(), output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)."
                )
                break

    print(f"Best EMA model saved to {output}")

    # Export to ONNX using EMA weights
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
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
