import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
from dataclasses import dataclass

import tyro
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name


@dataclass
class Args:
    output_dir: str = "data/craftax"
    total_frames: int = 100_000
    episode_length: int = 10_000
    chunk_size: int = 10_000
    img_size: int = 63
    train_ratio: float = 0.8
    seed: int = 0


def preprocess(rgb: jnp.ndarray, size: int) -> jnp.ndarray:
    img = rgb.astype(jnp.float32)
    img = jnp.transpose(img, (2, 0, 1))
    return jax.image.resize(img, (3, size, size), method="bilinear")


def generate_episode(env, size: int, max_steps: int, key: jax.Array):
    params = env.default_params
    rng, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)

    frames, actions = [preprocess(obs, size)], []
    for _ in range(max_steps):
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = env.action_space(params).sample(action_key)
        actions.append(action)

        obs, state, reward, done, _ = env.step(step_key, state, action, params)
        frames.append(preprocess(obs, size))

        if done:
            break

    return jnp.stack(frames[:-1]), jnp.array(actions)


def main(args: Args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    n_actions = env.action_space(env.default_params).n

    print(f"\nGenerating dataset:")
    print(f"  Total frames: {args.total_frames:,}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Train/test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print(f"  Output directory: {output_path}")
    print(f"  Number of actions: {n_actions}")

    train_frames = int(args.total_frames * args.train_ratio)
    test_frames = args.total_frames - train_frames

    for split, n_frames in [("train", train_frames), ("test", test_frames)]:
        split_path = output_path / split
        split_path.mkdir(exist_ok=True)

        frames_path = split_path / "frames.npy"
        actions_path = split_path / "actions.npy"

        frames_mmap = np.memmap(
            frames_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_frames, 3, args.img_size, args.img_size),
        )
        actions_mmap = np.memmap(
            actions_path, dtype=np.int32, mode="w+", shape=(n_frames,)
        )

        frame_idx = 0
        rng = jax.random.PRNGKey(args.seed if split == "train" else args.seed + 1)
        with tqdm(total=n_frames, desc=f"Generating {split} data") as pbar:
            while frame_idx < n_frames:
                rng, episode_key = jax.random.split(rng)

                frames, actions = generate_episode(
                    env, args.img_size, args.episode_length, episode_key
                )

                to_copy = min(len(frames), n_frames - frame_idx)

                frames_mmap[frame_idx : frame_idx + to_copy] = frames[:to_copy]
                actions_mmap[frame_idx : frame_idx + to_copy] = actions[:to_copy]

                frame_idx = frame_idx + to_copy
                pbar.update(to_copy)

                if frame_idx % args.chunk_size == 0:
                    frames_mmap.flush()
                    actions_mmap.flush()

        frames_mmap.flush()
        actions_mmap.flush()

        print(f"  Saved frames: {frames_path} ({frames_mmap.nbytes / 1e9:.2f} GB)")
        print(f"  Saved actions: {actions_path} ({actions_mmap.nbytes / 1e6:.2f} MB)")

    metadata = {
        "total_frames": args.total_frames,
        "train_frames": train_frames,
        "test_frames": test_frames,
        "img_size": args.img_size,
        "n_actions": n_actions,
        "episode_length": args.episode_length,
    }
    np.save(output_path / "metadata.npy", metadata)


if __name__ == "__main__":
    tyro.cli(main)
