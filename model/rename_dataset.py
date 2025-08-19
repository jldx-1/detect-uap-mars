#!/usr/bin/env python3
"""
rename_dataset.py

Batch‑renames your YOLO folders so that train/val/test indices
get offset by a fixed amount (e.g. +10000), preventing name collisions
when you merge in a second dataset with overlapping filenames.

Usage:
    python rename_dataset.py --root PATH/TO/data --offset 10000
    # Optional: --splits train val test  (default: all three)
"""

import argparse
import os
import re
from pathlib import Path

def rename_split(root: Path, split: str, offset: int):
    """
    Rename files in:
      root/{split}/images/*.jpg
      root/{split}/labels/*.txt
      root/{split}/mask/*.{png,jpg}

    Patterns handled:
      <split>_<idx>.jpg
      <split>_<idx>.txt
      <split>_<idx>_mask.png
    """
    pattern = re.compile(rf"^({split})_(\d+)(?:_mask)?\.(jpg|png|txt)$", re.IGNORECASE)
    for sub in ("images", "labels", "mask"):
        folder = root / split / sub
        if not folder.exists():
            continue
        for f in folder.iterdir():
            m = pattern.match(f.name)
            if not m:
                continue
            base, idx, ext = m.group(1), m.group(2), m.group(3)
            new_idx = int(idx) + offset
            # preserve _mask suffix if present
            mask_suffix = "_mask" if "_mask" in f.stem else ""
            new_name = f"{base}_{new_idx:05d}{mask_suffix}.{ext.lower()}"
            old = f
            new = folder / new_name
            print(f"Renaming: {old.relative_to(root)} → {new.relative_to(root)}")
            old.rename(new)

def main():
    p = argparse.ArgumentParser(
        description="Offset YOLO split filenames to avoid collisions"
    )
    p.add_argument(
        "--root", "-r", required=True,
        help="Path to your data folder (with train/, val/, test/ subfolders)"
    )
    p.add_argument(
        "--offset", "-o", type=int, default=10000,
        help="Integer offset to add to all indices (default 10000)"
    )
    p.add_argument(
        "--splits", "-s", nargs="+", default=["train", "val", "test"],
        help="Which splits to process (default: all three)"
    )
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: data root {root} does not exist.")
        return

    for split in args.splits:
        print(f"\nProcessing split '{split}' with offset {args.offset}…")
        rename_split(root, split, args.offset)

    print("\n✅ Done renaming.")

if __name__ == "__main__":
    main()
