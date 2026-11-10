import sys
import shutil
import random
import argparse
from pathlib import Path

# Allow running as `python scripts/prepare_dataset.py` from the repo root:
# put the project root (parent of scripts/) on sys.path so `config` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import cfg, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test dataset split from raw .npz files"
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Path to raw .npz files (overrides config)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=39,
        help="Random seed for deterministic split"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_cfg = cfg.dataset

    # Source Directory
    if args.source_dir is not None:
        raw_dir = Path(args.source_dir).expanduser().resolve()
    else:
        raw_dir = Path(dataset_cfg.raw_dir).expanduser().resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset dir not found: {raw_dir}")

    # Target Directories
    target_root = (PROJECT_ROOT / dataset_cfg.dir).resolve()
    train_dir = target_root / "train"
    val_dir = target_root / "val"
    test_dir = target_root / "test"

    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Split Params
    train_ratio = dataset_cfg.train_ratio
    val_ratio = dataset_cfg.val_ratio
    test_ratio = dataset_cfg.test_ratio

    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {total_ratio})"
        )

    seed = args.seed

    # Load Files
    raw_files = sorted(raw_dir.glob("*.npz"))
    if len(raw_files) == 0:
        raise RuntimeError(f"No .npz files found in {raw_dir}")

    random.seed(seed)
    random.shuffle(raw_files)

    n_total = len(raw_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_files = raw_files[:n_train]
    val_files = raw_files[n_train:n_train + n_val]
    test_files = raw_files[n_train + n_val:]

    print("Dataset split summary")
    print("-" * 40)
    print(f"Source dir: {raw_dir}")
    print(f"Target dir: {target_root}")
    print(f"Total samples: {n_total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")
    print(f"Seed:  {seed}")
    print("-" * 40)

    def copy_files(files, dst):
        for f in files:
            shutil.copy2(f, dst / f.name)

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    print("Dataset prepared successfully.")


if __name__ == "__main__":
    main()
