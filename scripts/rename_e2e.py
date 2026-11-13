from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
from src.utils import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/e2e/raw",
                    help="directory containing raw CSV files")
    ap.add_argument("--rename", action="store_true",
                    help="rename files to train.csv, dev.csv, test.csv")
    args = ap.parse_args()

    raw_dir = args.raw_dir
    if not os.path.isdir(raw_dir):
        print(f"Directory does not exist: {raw_dir}")
        sys.exit(1)

    # Expected original filenames from E2E dataset
    original_names = ["trainset.csv", "devset.csv", "testset_w_refs.csv"]
    target_names = ["train.csv", "dev.csv", "test.csv"]

    files = os.listdir(raw_dir)
    print(f"Files in {raw_dir}: {files}")

    # Check if original files exist
    for orig, target in zip(original_names, target_names):
        orig_path = os.path.join(raw_dir, orig)
        target_path = os.path.join(raw_dir, target)
        if os.path.exists(orig_path):
            print(f"Found {orig}")
            if args.rename:
                shutil.move(orig_path, target_path)
                print(f"  Renamed to {target}")
        elif os.path.exists(target_path):
            print(f"Already named {target}")
        else:
            print(f"Missing {orig} (or {target})")

    print("Done.")


if __name__ == "__main__":
    main()