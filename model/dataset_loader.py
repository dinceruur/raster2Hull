import math
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
from config.loader import PROJECT_ROOT
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Literal


# View class mapping (shared across lines & GA plans)
VIEW_CLASS_MAP = {
    "YZ": 1,  # Body plan or mid-section
    "XY": 2,  # Half-breadth or top-view
    "XZ": 3,  # Sheer plan or profile view
}

# Supported 2D drawing types and the .npz keys each one reads.
DRAWING_TYPES = {
    "lines": {
        "image": "gray_lines_image",
        "annotations": "lines_annotations",
        "canvas": ("lines_canvas_properties",),
    },
    "ga_simple": {
        "image": ("gray_ga_simple_image", "gray_ga_image"),
        "annotations": "ga_annotations",
        "canvas": ("ga_canvas_properties", "lines_canvas_properties"),
    },
    "ga_detailed": {
        "image": ("gray_ga_detailed_image",),
        "annotations": "ga_annotations",
        "canvas": ("ga_canvas_properties", "lines_canvas_properties"),
    },
    "ga": {
        "image": ("gray_ga_image", "gray_ga_simple_image"),
        "annotations": "ga_annotations",
        "canvas": ("ga_canvas_properties", "lines_canvas_properties"),
    },
}


def resolve_drawing_type(cfg) -> str:
    drawing_type = str(getattr(cfg, "drawing_type", "lines")).lower()
    if drawing_type not in DRAWING_TYPES:
        raise ValueError(
            f"Invalid drawing_type '{drawing_type}'. "
            f"Expected one of: {sorted(DRAWING_TYPES)}"
        )
    return drawing_type


def _read_canvas_properties(data, canvas_keys):
    for key in canvas_keys:
        if key in data:
            props = data[key].item()
            return props["canvas_width"], props["canvas_height"]
    raise KeyError(
        f"None of the canvas property keys {canvas_keys} were found in the sample."
    )


def _read_image(data, image_keys):
    if isinstance(image_keys, str):
        image_keys = (image_keys,)
    for key in image_keys:
        if key in data:
            return data[key]
    raise KeyError(
        f"None of the image keys {image_keys} were found in the sample."
    )


def normalize_bbox(bbox_pdf, image_w, image_h, pdf_w, pdf_h):
    x0, y_bottom0, x1, y_bottom1 = bbox_pdf

    # Flip the Y-axis
    # In PDF: y increases going UP. In Image: y increases going DOWN.
    y_top0 = pdf_h - y_bottom0
    y_top1 = pdf_h - y_bottom1

    # Calculate Ratios
    rx0, ry0 = x0 / pdf_w, y_top0 / pdf_h
    rx1, ry1 = x1 / pdf_w, y_top1 / pdf_h

    # Handle Min/Max
    # After flipping, y_top0 might be larger than y_top1.
    # Standard bboxes expect [xmin, ymin, xmax, ymax]
    xmin, xmax = min(rx0, rx1), max(rx0, rx1)
    ymin, ymax = min(ry0, ry1), max(ry0, ry1)

    # Scale to current Image Pixels
    return torch.tensor(
        [xmin * image_w, ymin * image_h, xmax * image_w, ymax * image_h],
        dtype=torch.float32
    )


def img_rotation(img, max_deg=3.0, fill=0.0):
    angle = random.uniform(-max_deg, max_deg)
    img_rot = TF.rotate(
        img, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False, fill=fill)
    return img_rot, angle


def rotate_bboxes(annotations, angle_deg, H, W):
    rad = math.radians(angle_deg)
    cx_img, cy_img = W / 2, H / 2  # image center

    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot_mat = torch.tensor([[cos_a, -sin_a],
                            [sin_a,  cos_a]],
                           dtype=torch.float32)

    cxcy_tensor = torch.tensor(
        [cx_img, cy_img], dtype=torch.float32)

    for ann in annotations:
        x0, y0, x1, y1 = ann['bbox']

        corners = torch.tensor([
            [x0, y0],
            [x1, y0],
            [x0, y1],
            [x1, y1]
        ], dtype=torch.float32)

        # shift to image center
        corners_centered = corners - cxcy_tensor

        # rotate
        rotated = (corners_centered @ rot_mat.T) + cxcy_tensor

        x_min, y_min = rotated[:, 0].min(), rotated[:, 1].min()
        x_max, y_max = rotated[:, 0].max(), rotated[:, 1].max()

        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        ann['bbox'] = torch.tensor([
            x_min.clamp(0, W-1),
            y_min.clamp(0, H-1),
            x_max.clamp(0, W-1),
            y_max.clamp(0, H-1)
        ], dtype=torch.float32)

    return annotations


def random_rotation(img, annotations, max_deg=3.0, p=0.5, fill=0.0):
    if random.random() > p:
        return img, annotations

    _, H, W = img.shape
    img_rot, angle = img_rotation(img, max_deg=max_deg, fill=fill)

    annotations_rot = copy.deepcopy(annotations)
    if annotations_rot:
        annotations_rot = rotate_bboxes(annotations_rot, -angle, H, W)

    return img_rot, annotations_rot


def random_gaussian_blur(img, p=0.3):
    if random.random() < p:
        img = TF.gaussian_blur(img, kernel_size=3)
    return img


def random_noise(img, p=0.3):
    if random.random() < p:
        noise = torch.randn_like(img) * 0.02
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)
    return img


def uneven_lighting_torch1(img, p=0.2):
    if random.random() > p:
        return img

    _, h, w = img.shape

    # Create coordinate grid (-1 to 1)
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    xv, yv = torch.meshgrid(x, y, indexing="xy")

    # Calculate distance from center squared (radius^2)
    # Center = 0.0, Corners ≈ 2.0
    radius_sq = xv**2 + yv**2

    # Define how bright the corners should get
    # 0.1 to 0.2 means the black background becomes dark gray
    strength = random.uniform(0.1, 0.2)

    # Create the 'Light Leak' Mask
    # Center is 0.0 (Pure Black)
    # Edges increase in brightness
    # Normalize roughly to 0-1 range * strength
    light_map = strength * (radius_sq / 2.0)

    # Apply: Lift the floor of the image
    # Background (0.0) becomes (light_map value)
    # Lines (1.0) stay (1.0) because 1.0 > 0.15
    return torch.maximum(img, light_map)


def uneven_lighting_torch(img, p=0.2):
    if random.random() > p:
        return img

    _, h, w = img.shape

    x = torch.linspace(-1, 1, w, device=img.device).view(1, 1, w)
    y = torch.linspace(-1, 1, h, device=img.device).view(1, h, 1)

    # Broadcasting naturally expands the 1D vectors into a 2D distance map
    radius_sq = x**2 + y**2

    strength = random.uniform(0.1, 0.2)
    light_map = strength * (radius_sq / 2.0)

    return torch.maximum(img, light_map)


def paper_gray_torch1(img, p=0.2):
    if random.random() > p:
        return img

    # We want the background to be slightly non-zero (e.g., 0.04 to 0.14)
    # This simulates "dark gray" paper instead of perfect void black.
    gray_level = random.uniform(0.04, 0.14)

    # use 'maximum' to raise the floor.
    # Pixels that were 0.0 become 'gray_level'.
    # Pixels that are 1.0 (Lines) stay 1.0.
    return torch.maximum(img, torch.tensor(gray_level, device=img.device))


def paper_gray_torch(img, p=0.2):
    if random.random() > p:
        return img

    gray_level = random.uniform(0.04, 0.14)

    return torch.clamp(img, min=gray_level)


class DrawingDataset(Dataset):
    def __init__(self, cfg, split: Literal["train", "val", "test"]):
        if cfg is None:
            raise ValueError("DrawingDataset requires a dataset config")

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self.is_train = split == "train"
        self.augment = bool(cfg.augment and self.is_train)

        # Select which drawing: lines plan or general arrangement
        self.drawing_type = resolve_drawing_type(cfg)
        self.keys = DRAWING_TYPES[self.drawing_type]

        self.num_points = int(cfg.num_points)
        self.image_width = int(cfg.img_width)

        dataset_dir = (PROJECT_ROOT / cfg.dir / split).resolve()

        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}")

        self.file_list = sorted(dataset_dir.glob("*.npz"))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npz files found in {dataset_dir}")

        print(
            f"[DrawingDataset] split={split} | drawing_type={self.drawing_type} "
            f"| samples={len(self.file_list)} | augment={self.augment}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx], allow_pickle=True)

        identifiers = (Path(self.file_list[idx]).stem)

        # Load hull parameters
        if not "vector" in data:
            print(idx)
            print(self.file_list[idx])

        params = torch.from_numpy(data["vector"]).float()

        # Load sized and centered point cloud
        pc = data["target_pc"].astype(np.float32)
        ns = torch.from_numpy(data["target_ns"]).float()

        # Random sampling
        if pc.shape[0] > self.num_points:
            idx_rand = torch.randperm(pc.shape[0])[:self.num_points]
            pc = pc[idx_rand]
            ns = ns[idx_rand]

        img_np = _read_image(data, self.keys["image"])
        img = torch.from_numpy(img_np).float()

        # Add channel dimension early: (H, W) -> (1, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        if img.max() > 1.0:
            img /= 255.0

        # Invert: Background = 0.0, Lines = 1.0
        img = 1.0 - img

        # Resize (TF.resize operates natively on (C, H, W) tensors)
        img = self.resize_width_keep_ratio(img, target_width=self.image_width)
        _, H, W = img.shape

        # Annotations
        raw_annotations = data[self.keys["annotations"]]
        pdf_w, pdf_h = _read_canvas_properties(data, self.keys["canvas"])

        annotations = []
        for ann in raw_annotations:
            bbox_abs = normalize_bbox(ann["bbox_pdf"], W, H, pdf_w, pdf_h)
            annotations.append({
                "class": VIEW_CLASS_MAP[ann["class"]],
                "bbox": bbox_abs,
                "semantic": ann.get("semantic", None)
            })

        # Augmentations
        if self.augment and annotations:
            # Rotation (Geometric)
            img, annotations = random_rotation(
                img, annotations, max_deg=3.0, p=0.5, fill=0.0)

            # Blur (Destructive)
            img = random_gaussian_blur(img, p=0.15)

            # Noise (Additive)
            img = random_noise(img, p=0.3)

            # Lighting & Gray (Background Domain)
            img = uneven_lighting_torch(img, p=0.4)
            img = paper_gray_torch(img, p=0.4)

        bboxes_list = []
        views_list = []

        for ann in annotations:
            b = ann["bbox"]
            b_norm = torch.tensor([
                b[0] / W, b[1] / H, b[2] / W, b[3] / H
            ], dtype=torch.float32)

            bboxes_list.append(b_norm)
            views_list.append(ann["class"])

        if bboxes_list:
            bboxes = torch.stack(bboxes_list)
            views = torch.tensor(views_list, dtype=torch.long)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            views = torch.zeros((0,), dtype=torch.long)

        return {
            "id": identifiers,
            "image": img,
            "points": torch.from_numpy(pc),
            "normals": ns,
            "bboxes": bboxes,
            "views": views,
            "params": params,
        }

    @staticmethod
    def resize_width_keep_ratio(img, target_width=512):
        """Resize image keeping aspect ratio."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        _, h, w = img.shape
        new_h = int(round(h * (target_width / w)))
        return TF.resize(img, [new_h, target_width], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


def drawing_collate_fn(batch):
    identifiers = [item["id"] for item in batch]

    # Dynamic Image Padding
    raw_imgs = [item["image"] for item in batch]
    max_h = max([img.shape[1] for img in raw_imgs])
    max_w = max([img.shape[2] for img in raw_imgs])
    padded_imgs = []
    for img in raw_imgs:
        _, h, w = img.shape
        # Pad format: (left, right, top, bottom)
        # We pad the bottom to reach max_h
        pad_bottom = max_h - h
        pad_right = max_w - w
        padded = torch.nn.functional.pad(
            img, (0, pad_right, 0, pad_bottom), value=0.0)
        padded_imgs.append(padded)

    imgs = torch.stack(padded_imgs, dim=0)
    pcs = torch.stack([item["points"] for item in batch], dim=0)
    ns = torch.stack([item["normals"] for item in batch], dim=0)
    params = torch.stack([item["params"] for item in batch], dim=0)
    bboxes = [item["bboxes"] for item in batch]
    views = [item["views"] for item in batch]

    # Handle Variable BBoxes (Padding)
    B = len(bboxes)

    max_n = max([b.shape[0] for b in bboxes]) if any(
        b.shape[0] > 0 for b in bboxes) else 0

    batch_bboxes = torch.zeros(B, max_n, 4, dtype=torch.float32)
    batch_views = torch.zeros(B, max_n, dtype=torch.long)
    bbox_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i in range(B):
        n = bboxes[i].shape[0]
        if n > 0:
            batch_bboxes[i, :n] = bboxes[i]
            batch_views[i, :n] = views[i]
            bbox_mask[i, :n] = True

    return {
        "identifiers": identifiers,
        "images": imgs,
        "points": pcs,
        "normals": ns,
        "params": params,
        "bboxes_coordinates": batch_bboxes,
        "bbox_classes": batch_views,
        "bbox_masks": bbox_mask,
    }
