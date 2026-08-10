"""
view_clip.py

Standalone viewer for a single preprocessed TUAB clip (.pkl file produced
by data_preprocessing.py). Load one file, print its contents, and save a plot of all 16 channels.

Usage:
    python3 view_clip.py                                  # picks a random file from data/processed/train
    python3 view_clip.py path/to/some_chunk_0.pkl         # view a specific file
"""

import sys
import glob
import random
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_default_file():
    candidates = glob.glob("data/processed/train/*.pkl")
    if not candidates:
        raise FileNotFoundError(
            "No .pkl files found under data/processed/train. "
        )
    return random.choice(candidates)


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else find_default_file()

    with open(file_path, "rb") as f:
        clip = pickle.load(f)

    X = clip["X"]
    y = clip["y"]

    print(f"file:  {file_path}")
    print(f"shape: {X.shape}   dtype: {X.dtype}")
    
    label_name = "Abnormal" if y == 1 else "Normal"
    print(f"Label: {y} ({label_name})")

    fig, axes = plt.subplots(X.shape[0], 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(X[i], linewidth=0.5)
        ax.set_ylabel(f"ch{i}", fontsize=6, rotation=0, labelpad=20)
        ax.set_yticks([])
    axes[-1].set_xlabel("samples (200Hz, 10s window)")
    fig.suptitle(f"{file_path}  |  label={y} ({'abnormal' if y == 1 else 'normal'})")

    out_path = "clip_preview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()
