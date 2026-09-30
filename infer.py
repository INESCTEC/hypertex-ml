#!/usr/bin/env python3
"""Inference script for hyperspectral classification models.

Supports running inference from either an HDF5 dataset file or a raw capture directory.

Usage examples:
    # Inference from an HDF5 file:
    python infer.py --model_path checkpoints/model_mlp.pth --model mlp \
        --dataset_path path/to/train.hdf5 --sample_name SAMPLE_ID --cuda

    # Inference from a raw capture directory:
    python infer.py --model_path checkpoints/model_mlp.pth --model mlp \
        --dataset_path path/to/dataset_dir --sample_name SAMPLE_ID --cuda
"""

import argparse
import os
import numpy as np
import torch
import spectral
import h5py
import matplotlib.pyplot as plt

try:
    from train import SimpleMLP, Simple3DCNN
except ImportError:
    from .train import SimpleMLP, Simple3DCNN


FIBRE_CLASSES = [
    "Unknown",
    "Cotton",
    "Wool",
    "Lyocell",
    "Viscose",
    "Polyester",
    "Linen",
    "Elastane",
    "Polyamide",
    "Acrylic",
]


def infer_data(net, img, patch_size, batch_size, device, n_classes, mask=None):
    """Run inference on a single hyperspectral image.

    Args:
        net: Trained PyTorch model.
        img: numpy array of shape (H, W, C).
        patch_size: int, size of square patch.
        batch_size: int, batch size for processing.
        device: torch device.
        n_classes: number of output classes.
        mask: optional binary mask to limit inference region.
    """
    pad = patch_size // 2
    H, W, C = img.shape
    probs = np.zeros((H, W, n_classes), dtype=np.float32)

    # valid coordinates
    valid_coords = [
        (x, y)
        for x in range(pad, H - pad)
        for y in range(pad, W - pad)
        if mask is None or mask[x, y]
    ]

    def generate_batches(coords, size):
        for i in range(0, len(coords), size):
            yield coords[i : i + size]

    for batch_coords in generate_batches(valid_coords, batch_size):
        patches = []
        for x, y in batch_coords:
            patch = img[x - pad : x + pad + 1, y - pad : y + pad + 1]
            patch = patch.transpose(2, 0, 1)  # (C, H, W)
            patches.append(patch)
        patches = np.stack(patches)  # (B, C, H, W)
        if patch_size == 1:
            data = patches[:, :, 0, 0]
            data = data[:, np.newaxis, :]
        else:
            data = patches[:, np.newaxis, :, :, :]
        data = torch.from_numpy(data).float().to(device)
        with torch.no_grad():
            out = net(data)
            if isinstance(out, tuple):
                out = out[0]
            out = out.cpu().numpy()
        for (x, y), o in zip(batch_coords, out):
            probs[x, y] += o
    return probs


def load_model(model_path, model_type, n_bands, n_outputs, device):
    if model_type == "mlp":
        model = SimpleMLP(n_bands=n_bands, n_outputs=n_outputs)
    else:
        model = Simple3DCNN(n_outputs=n_outputs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_sample_hsi(dataset_path, sample_name):
    """Load hyperspectral image array from an HDF5 dataset or a raw capture directory."""
    if os.path.isfile(dataset_path) and h5py.is_hdf5(dataset_path):
        print(f"Loading sample '{sample_name}' from HDF5 file: {dataset_path}")
        with h5py.File(dataset_path, "r") as f:
            if "data" not in f or sample_name not in f["data"]:
                raise KeyError(
                    f"Sample '{sample_name}' not found under /data in HDF5 file {dataset_path}"
                )
            img = f["data"][sample_name][:]
            return np.asarray(img, dtype=np.float32)

    elif os.path.isdir(dataset_path):
        sample_dir = os.path.join(dataset_path, sample_name, "capture")
        reflectance_path = os.path.join(sample_dir, f"REFLECTANCE_{sample_name}.hdr")

        if not os.path.exists(reflectance_path) and os.path.isdir(sample_dir):
            candidates = [p for p in os.listdir(sample_dir) if p.endswith(".hdr")]
            reflectance_path = next(
                (os.path.join(sample_dir, p) for p in candidates if p.startswith("REFLECTANCE_")),
                os.path.join(sample_dir, candidates[0]) if candidates else reflectance_path,
            )

        if not os.path.exists(reflectance_path):
            raise FileNotFoundError(
                f"HDR reflectance header for sample '{sample_name}' not found at {reflectance_path}"
            )

        print(f"Loading sample '{sample_name}' from ENVI capture: {reflectance_path}")
        img = spectral.open_image(reflectance_path).load()
        return np.asarray(img, dtype=np.float32)

    else:
        raise ValueError(
            f"Dataset path '{dataset_path}' is neither a valid HDF5 file nor an existing directory."
        )


def plot_results(img, probs, sample_name, save_path=None, show_plot=True):
    """Display and optionally save RGB visualization alongside predicted argmax classification map."""
    H, W, C = img.shape
    argmax_map = np.argmax(probs, axis=-1)

    # Generate false-color RGB preview
    if C >= 220:
        rgb_bands = [19, 149, 219]
    else:
        rgb_bands = [0, C // 2, C - 1]

    rgb_img = img[:, :, rgb_bands]
    rgb_min, rgb_max = rgb_img.min(), rgb_img.max()
    if rgb_max > rgb_min:
        rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)
    rgb_img = np.clip(rgb_img, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Sample: {sample_name}")
    axes[0].axis("off")

    num_classes = len(FIBRE_CLASSES)
    cmap = plt.get_cmap("tab10", num_classes)
    cax = axes[1].imshow(argmax_map, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[1].set_title(f"Predicted Class Map (argmax)")
    axes[1].axis("off")

    cbar = fig.colorbar(cax, ax=axes[1], ticks=range(num_classes), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(FIBRE_CLASSES)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Inference plot saved to {save_path}")

    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot interactively: {e}")

    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Load hyperspectral image data
    img = load_sample_hsi(args.dataset_path, args.sample_name)
    n_bands = img.shape[-1]
    n_outputs = args.n_classes

    # Load model
    model = load_model(args.model_path, args.model, n_bands, n_outputs, device)

    # Run inference
    print(f"Running inference (patch_size={args.patch_size}, batch_size={args.batch_size})...")
    probs = infer_data(
        net=model,
        img=img,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=device,
        n_classes=n_outputs,
    )

    # Save output probabilities (.npy)
    out_path = args.output if args.output else os.path.join("runs", f"{args.sample_name}_probs.npy")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, probs)
    print(f"Inference completed. Probabilities saved to {out_path}")

    # Plot & visualize results
    plot_path = args.plot_path if args.plot_path else os.path.join("runs", f"{args.sample_name}_inference.png")
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    show_plot = not args.no_show
    plot_results(img, probs, args.sample_name, save_path=plot_path, show_plot=show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--model_path", required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument(
        "--model",
        choices=["mlp", "3dcnn"],
        required=True,
        help="Model architecture used for training",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to HDF5 dataset file (.hdf5) OR raw capture directory containing sample folders",
    )
    parser.add_argument("--sample_name", required=True, help="Identifier of the sample to infer")
    parser.add_argument(
        "--patch_size", type=int, default=1, help="Patch size used during training"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of target classes")
    parser.add_argument("--output", help="File path to store output probabilities (.npy)")
    parser.add_argument(
        "--plot_path", help="File path to save the inference visualization plot (.png)"
    )
    parser.add_argument(
        "--no_show", action="store_true", help="Do not display the matplotlib plot interactively"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    main(args)
