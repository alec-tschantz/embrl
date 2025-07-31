import os
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from tqdm import tqdm

from tokenizer import Tokenizer


@dataclass
class Args:
    data_dir: str = "data/craftax"
    output_path: str = "data/checkpoints/tokenizer.eqx"
    patch_size: int = 7
    codebook_size: int = 512
    threshold: float = 0.75
    seed: int = 0


def load_data(data_dir: Path):
    metadata = np.load(data_dir / "metadata.npy", allow_pickle=True).item()

    train_frames = np.memmap(
        data_dir / "train" / "frames.npy",
        dtype=np.float32,
        mode="r",
        shape=(metadata["train_frames"], 3, metadata["img_size"], metadata["img_size"]),
    )

    return train_frames, metadata


def main(args: Args):
    data_path = Path(args.data_dir)
    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    frames_mmap, metadata = load_data(data_path)
    n_frames = frames_mmap.shape[0]
    img_size = metadata["img_size"]

    print(f"\nDataset info:")
    print(f"  Total frames: {n_frames:,}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Patches per frame: {(img_size // args.patch_size) ** 2}")

    dim = args.patch_size * args.patch_size * 3
    tokenizer = Tokenizer(
        dim=dim,
        thr=args.threshold,
        max_codes=args.codebook_size,
        key=jax.random.PRNGKey(args.seed),
    )

    print(f"\nTokenizer config:")
    print(f"  Codebook size: {args.codebook_size}")
    print(f"  Patch dimension: {dim}")
    print(f"  Threshold: {args.threshold}")

    print(f"\nTraining tokenizer on {n_frames:,} frames...")

    for i in tqdm(range(n_frames), desc="Training tokenizer"):
        frame = frames_mmap[i : i + 1, :, None, :, :]
        tokenizer = tokenizer.update(frame, args.patch_size)

    print(
        f"  Tokenizer ready: {int(tokenizer.active.sum())}/{args.codebook_size} codes"
    )

    eqx.tree_serialise_leaves(output_file, tokenizer)
    print(f"\n✓ Tokenizer saved to: {output_file}")


if __name__ == "__main__":
    tyro.cli(main)
