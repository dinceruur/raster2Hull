#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from config.loader import cfg, PROJECT_ROOT
from model.model import ImageToPointCloudModel
from model.dataset_loader import DrawingDataset


# Input loading + preprocessing
def load_grayscale(image_path: Path, dpi: int = 300) -> np.ndarray:
    ext = image_path.suffix.lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(image_path)
        pix = doc[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        doc.close()
        return arr.astype(np.float32) / 255.0

    from PIL import Image
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return arr / 255.0


def preprocess(gray: np.ndarray, img_width: int) -> torch.Tensor:
    img = torch.from_numpy(gray).float()
    if img.max() > 1.0:
        img = img / 255.0
    img = img.unsqueeze(0)                     # (1, H, W)
    img = 1.0 - img                            # invert: background 0, ink 1
    img = DrawingDataset.resize_width_keep_ratio(img, target_width=img_width)
    return img.unsqueeze(0)                    # (1, 1, H, W)


def save_ply(points: np.ndarray, path: Path) -> None:
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def render_png(points: np.ndarray, path: Path, show: bool = False) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=12, azim=-60)
    z = points[:, 2]
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-8)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=z_norm, cmap="viridis", s=6, edgecolors="none")
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Predicted hull point cloud")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Infer a 3D hull point cloud from a single 2D drawing.")
    ap.add_argument("--image", required=True,
                    help="Input drawing: .png/.jpg/… or a .pdf (first page).")
    ap.add_argument("--checkpoint", default=None,
                    help="Model checkpoint (default: "
                         "checkpoints/checkpoint_best.pth.tar).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write outputs (default: ./predictions/<stem>/).")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Render DPI when the input is a PDF (default 300).")
    ap.add_argument("--ply", action="store_true",
                    help="Also write points.ply (ASCII PLY).")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip the render.png 3D preview.")
    ap.add_argument("--show", action="store_true",
                    help="Display the 3D preview interactively.")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available.")
    return ap.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"ERROR: input image not found: {image_path}", file=sys.stderr)
        return 2

    ckpt = (Path(args.checkpoint).expanduser().resolve() if args.checkpoint
            else PROJECT_ROOT / cfg.training.checkpoints_dir / "checkpoint_best.pth.tar")
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    out_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
               else PROJECT_ROOT / "predictions" / image_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")

    # Model + weights
    model = ImageToPointCloudModel(cfg=cfg.model).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    # Input
    gray = load_grayscale(image_path, dpi=args.dpi)
    img = preprocess(gray, img_width=int(cfg.dataset.img_width)).to(device)

    print(f"Image      : {image_path.name}  (model input {tuple(img.shape)})")
    print(f"Checkpoint : {ckpt.name}  | device: {device.type}")

    # Inference
    with torch.no_grad():
        out = model(img)
        points = out[0] if isinstance(out, tuple) else out
        pred = points["refined"][0].cpu().numpy().astype(np.float32)  # (N, 3)

    # Save
    np.save(out_dir / "points.npy", pred)
    print(f"Predicted  : {pred.shape[0]} points -> {out_dir / 'points.npy'}")
    if args.ply:
        save_ply(pred, out_dir / "points.ply")
        print(f"  PLY      -> {out_dir / 'points.ply'}")
    if not args.no_render:
        render_png(pred, out_dir / "render.png", show=args.show)
        print(f"  Preview  -> {out_dir / 'render.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
