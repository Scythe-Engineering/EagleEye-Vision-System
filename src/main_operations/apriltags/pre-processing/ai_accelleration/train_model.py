import json
import os
from typing import Tuple

import torch
import torch.optim.lr_scheduler as lr_scheduler
from cv2 import imread
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from grid_detectors.predictor import GridPredictor
from utils import TARGET_WIDTH, TARGET_HEIGHT, MODEL_PATH, LetterboxTransform, GRID_WIDTH, GRID_HEIGHT


class GridDataset(Dataset):
    """Dataset of raw frames and occupancy grids of size GRID_HEIGHT×GRID_WIDTH saved as JSON."""

    def __init__(
            self, data_dir: str, transform: transforms.Compose, cache: bool = False
    ):
        """ Initialize the GridDataset.

        Args:
            data_dir (str): The directory containing the training data.
            transform (transforms.Compose): The transform to apply to the images.
            cache (bool, optional): Whether to cache the images and labels in memory. Defaults to False.
        """
        self.data_dir = data_dir
        self.transform = transform
        files = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".png"))
        self.bases = [
            b for b in files if os.path.isfile(os.path.join(data_dir, b + ".json"))
        ]
        self.cache = cache
        if self.cache:
            self.imgs = []
            self.labels = []
            for b in self.bases:
                img = imread(os.path.join(data_dir, b + ".png"))
                self.imgs.append(self.transform(img))
                with open(os.path.join(data_dir, b + ".json")) as jf:
                    grid = json.load(jf)["grid"]
                self.labels.append(torch.tensor(grid, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.bases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and label.
        """
        if self.cache:
            return self.imgs[idx], self.labels[idx]
        base = self.bases[idx]
        img = imread(os.path.join(self.data_dir, base + ".png"))
        img_t = self.transform(img)
        with open(os.path.join(self.data_dir, base + ".json"), "r") as jf:
            grid = json.load(jf)["grid"]
        label = torch.tensor(grid, dtype=torch.float32)
        return img_t, label


# ——— Config ———
data_dir = "apriltags/ai_accelleration/training_data"
epochs = 400
batch_size = 64
lr = 1e-3
output = MODEL_PATH
patience = 10  # early stopping


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tf = transforms.Compose(
        [
            LetterboxTransform((TARGET_WIDTH, TARGET_HEIGHT)),
            transforms.ToTensor(),
        ]
    )

    dataset = GridDataset(data_dir, transform=tf, cache=True)
    train_sz = int(0.9 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, os.cpu_count() - 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, os.cpu_count() - 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    model = GridPredictor()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scaler = GradScaler()

    best_val, no_improve = float("inf"), 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        model.train()
        running_loss = 0.0

        for imgs, grids in tqdm(
            train_loader, desc=f" Epoch {epoch}/{epochs}", unit="batch", leave=False
        ):
            imgs = imgs.to(device, non_blocking=True)
            grids = grids.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, grids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, grids in val_loader:
                imgs, grids = imgs.to(device, non_blocking=True), grids.to(
                    device, non_blocking=True
                )
                logits = model(imgs)
                val_loss += criterion(logits, grids).item() * imgs.size(0)
        val_loss /= len(val_ds)
        print(
            f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f} — Val Loss: {val_loss:.4f}"
        )

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), output)
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f"Model saved to {output}")


if __name__ == "__main__":
    train()
