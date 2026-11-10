import sys
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Allow running as `python scripts/generate_mean.py` from the repo root:
# put the project root (parent of scripts/) on sys.path so `config`/`model` import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import cfg, PROJECT_ROOT
from model.dataset_loader import DrawingDataset
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def get_ship_hull_cmap(transition_point=0.35):
    """
    Creates a custom colormap:
    - Bottom (0.0 to transition): Antifouling Red (#800000 -> #A52A2A)
    - Transition: Sharp line
    - Top (transition to 1.0): Marine Gray (#708090 -> #B0C4DE)
    """

    # TODO: transition_point can be calculated, 0.35 is just default not waterline
    colors = [
        (0.0,  "#500000"),  # Deep Dark Red (Keel)
        # Brownish Red (Just below water)
        (transition_point - 0.01, "#A52A2A"),
        # Dark Slate Gray (Just above water)
        (transition_point, "#2F4F4F"),
        (1.0,  "#B0C4DE")   # Light Steel Blue (Deck level)
    ]
    return mcolors.LinearSegmentedColormap.from_list("ShipHull", colors)

def viz_mesh_comparison(target_pc):

    # Create figure with 3 subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Style
    ax.view_init(elev=10, azim=-30)
    ax.set_facecolor("white")
    ax.grid(False)

    # Create the Colormap
    custom_cmap = get_ship_hull_cmap(transition_point=0.4)

    # Normalize Z data for the colormap [0, 1]
    z = target_pc[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min())

    ax.scatter(
        target_pc[:, 0],
        target_pc[:, 1],
        target_pc[:, 2],
        c=z_norm,
        cmap=custom_cmap,
        s=10,
        edgecolors='none',
        alpha=1.0
    )

    # Limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_zlim(-1, 1)

    # Labeling
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    plt.tight_layout()

    plt.show()

def generate_mean_template():
    # Load TRAIN dataset only (Avoid data leakage)
    # Turn off augmentation to get pure parameters
    cfg.dataset.augment = False 
    dataset = DrawingDataset(cfg.dataset, split="train")
    
    print(f"Calculating Mean Shape from {len(dataset)} training samples...")

    # Collect all Parameter Vectors
    all_params = []
    
    # We only need the parameters, not the heavy images
    for i in tqdm(range(len(dataset))):
        # We access the file directly to avoid loading images (slow)
        data = np.load(dataset.file_list[i], allow_pickle=True)
        param = data["vector"].astype(np.float32)
        all_params.append(param)

    # Stack into (N_samples, N_params)
    all_params = np.stack(all_params)
    
    # Calculate Mean Vector
    mean_vector = np.mean(all_params, axis=0)
    print("Mean Parameter Vector calculated.")

    # Find the the ship closest to the mean
    # Euclidian distance from every ship to the mean vector
    dists = np.linalg.norm(all_params - mean_vector, axis=1)
    
    best_idx = np.argmin(dists)
    best_file = dataset.file_list[best_idx]
    
    print(f" Selected Representative Ship: {os.path.basename(best_file)}")
    print(f" Distance to Mean: {dists[best_idx]:.6f}")

    # Extract that ship's Point Cloud
    # This point cloud becomes the mean shape
    data = np.load(best_file, allow_pickle=True)
    mean_pc = data["target_pc"].astype(np.float32)
    
    # Ensuring it has the correct number of points
    if mean_pc.shape[0] > dataset.num_points:
        from model.utility import farthest_point_sampling
        pc_tensor = torch.from_numpy(mean_pc).unsqueeze(0) # (1, N, 3)
        idx_fps = farthest_point_sampling(pc_tensor, dataset.num_points).squeeze(0)
        mean_pc = mean_pc[idx_fps.numpy()]

    # Visualize
    viz_mesh_comparison(mean_pc)

    # Save (anchored to the project root so train.py finds it regardless of CWD)
    output_path = PROJECT_ROOT / "mean_ship.npy"
    np.save(output_path, mean_pc)
    print(f"Saved Mean Template to: {output_path}")
    
    # Print shape
    print(f"Template Shape: {mean_pc.shape}")

if __name__ == "__main__":
    generate_mean_template()