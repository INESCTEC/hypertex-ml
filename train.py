#!/usr/bin/env python3
"""Training script for hyperspectral dataset.
Adds model checkpoint saving after training.
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

from dataloader import HDF5HSIDataset


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# Transforms
# =====================================================

class RandomGaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


# =====================================================
# Dataset Wrapper
# =====================================================

class AugmentedHSIDataset(Dataset):
    """Wrap an existing dataset and apply transforms on‑the‑fly."""

    def __init__(self, dataset: Dataset, transform: T.Compose = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target


# =====================================================
# Models
# =====================================================

class SimpleMLP(nn.Module):
    """Spectral‑only MLP.

    Expected input shape: (B, 1, N_BANDS)
    Use with ``patch_size = 1``.
    """

    def __init__(self, n_bands: int, n_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        # (B,1,N_BANDS) -> (B,N_BANDS)
        x = x.squeeze(1)
        return torch.softmax(self.net(x), dim=1)


class Simple3DCNN(nn.Module):
    """Spectral‑spatial 3D CNN.

    Expected input shape: (B, 1, N_BANDS, H, W)
    Use with ``patch_size > 1``.
    """

    def __init__(self, n_outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(16, n_outputs)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return torch.softmax(self.head(x), dim=1)


# =====================================================
# Validation utilities
# =====================================================

def compute_multilabel_metrics(labels, predictions, threshold: float = 0.01):
    """Compute a suite of classification metrics.

    ``labels`` and ``predictions`` are expected to be ``np.ndarray`` of shape
    ``(N, H, W, C)`` where ``C`` is the number of classes.
    """
    from sklearn.metrics import (
        f1_score,
        multilabel_confusion_matrix,
        precision_score,
        r2_score,
        recall_score,
    )

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    binary_labels = (labels > threshold).astype(int)
    binary_predictions = (predictions > threshold).astype(int)

    mcm = multilabel_confusion_matrix(binary_labels, binary_predictions)
    tp = mcm[:, 1, 1].sum()
    tn = mcm[:, 0, 0].sum()
    fp = mcm[:, 0, 1].sum()
    fn = mcm[:, 1, 0].sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

    mismatched = np.count_nonzero(binary_labels != binary_predictions)
    matched = np.count_nonzero(binary_labels * binary_predictions)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "f1_macro": float(
            f1_score(binary_labels, binary_predictions, average="macro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(binary_labels, binary_predictions, average="micro", zero_division=0)
        ),
        "hamming": float((binary_labels != binary_predictions).mean()),
        "subset_acc": float((binary_labels == binary_predictions).all(axis=1).mean()),
        "ratio": float(matched / (mismatched + 1e-8)),
        "r2": float(r2_score(labels.ravel(), predictions.ravel())),
        "rmse": float(np.sqrt(np.mean((labels - predictions) ** 2))),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "f1_per_class": np.round(
            f1_score(binary_labels, binary_predictions, average=None, zero_division=0), 3
        ).tolist(),
        "precision_per_class": np.round(
            precision_score(binary_labels, binary_predictions, average=None, zero_division=0), 3
        ).tolist(),
        "recall_per_class": np.round(
            recall_score(binary_labels, binary_predictions, average=None, zero_division=0), 3
        ).tolist(),
    }


def validate(model, loader, criterion, device, threshold: float = 0.01):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            all_targets.append(target.cpu().numpy())
            all_outputs.append(output.cpu().numpy())
    metrics = compute_multilabel_metrics(
        np.concatenate(all_targets), np.concatenate(all_outputs), threshold=threshold
    )
    return running_loss / len(loader), metrics


# =====================================================
# Training loop
# =====================================================

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs: int,
):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1:03d} | Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | MAE: {val_metrics['mae']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | F1‑micro: {val_metrics['f1_micro']:.4f}"
        )


# =====================================================
# Main entry point
# =====================================================

def main(args):
    set_global_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    if args.patch_size > 1:
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                RandomGaussianNoise(0.01),
            ]
        )
    else:
        train_transform = T.Compose([RandomGaussianNoise(0.01)])

    dataset = HDF5HSIDataset(
        hsi_file=args.dataset, window_size=args.patch_size, step=args.step
    )
    train_idx, val_idx = dataset.create_split(
        val_split=args.val_split, samples_per_image=args.samples_per_image
    )
    train_dataset = AugmentedHSIDataset(
        Subset(dataset, train_idx), transform=train_transform
    )
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -------------------------------------------------
    # Sample inspection
    # -------------------------------------------------
    sample_x, sample_y = train_dataset[0]
    print(f"Sample X shape: {sample_x.shape}")
    print(f"Sample Y shape: {sample_y.shape}")
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch X shape : {batch_x.shape}")
    print(f"Batch Y shape : {batch_y.shape}")

    # -------------------------------------------------
    # Model selection
    # -------------------------------------------------
    n_outputs = sample_y.shape[0]
    if args.model == "mlp":
        if args.patch_size != 1:
            raise ValueError("MLP requires --patch_size 1")
        n_bands = sample_x.shape[-1]
        model = SimpleMLP(n_bands=n_bands, n_outputs=n_outputs)
    elif args.model == "3dcnn":
        if args.patch_size <= 1:
            raise ValueError("3DCNN requires --patch_size > 1")
        model = Simple3DCNN(n_outputs=n_outputs)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
    )

    # -------------------------------------------------
    # Save trained model checkpoint
    # -------------------------------------------------
    if args.checkpoint:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Model checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hyperspectral classification model")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--model", choices=["mlp", "3dcnn"], default="mlp")
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--samples_per_image", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_checkpoint.pth",
        help="File path to save the trained model weights",
    )
    args = parser.parse_args()
    main(args)
