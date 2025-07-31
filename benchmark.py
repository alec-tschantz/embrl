import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from tokenizer import Tokenizer
from transformer import Transformer


@dataclass
class Args:
    data_dir: str = "data/craftax"
    tokenizer_path: str = "data/checkpoints/tokenizer.eqx"

    batch_size: int = 16
    seq_len: int = 30
    patch_size: int = 7

    codebook_size: int = 512
    threshold: float = 0.75

    embed_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: float = 4.0

    dropout_embed: float = 0.0
    dropout_attn: float = 0.0
    dropout_ff: float = 0.0

    seed: int = 0


def get_memory_stats():
    devices = jax.devices()
    stats = []

    for device in devices:
        if hasattr(device, "memory_stats"):
            mem = device.memory_stats()
            stats.append(
                {
                    "device": str(device),
                    "used_gb": mem["bytes_in_use"] / 1e9,
                    "peak_gb": mem.get("peak_bytes_in_use", 0) / 1e9,
                    "limit_gb": mem.get("bytes_limit", 0) / 1e9,
                }
            )

    return stats


def print_memory_stats(label: str):
    print(f"\n{label}:")
    stats = get_memory_stats()

    for stat in stats:
        used = stat["used_gb"]
        peak = stat["peak_gb"]
        limit = stat["limit_gb"]

        used_pct = (used / limit * 100) if limit > 0 else 0
        peak_pct = (peak / limit * 100) if limit > 0 else 0

        print(f"  {stat['device']}")
        print(f"    Used:  {used:.2f} GB ({used_pct:.1f}%)")
        print(f"    Peak:  {peak:.2f} GB ({peak_pct:.1f}%)")
        print(f"    Limit: {limit:.2f} GB")


def count_parameters(model):
    leaves, _ = jax.tree_util.tree_flatten(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)


def loss_fn(
    model: Transformer,
    x: jnp.ndarray,
    y: jnp.ndarray,
    actions: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model)(x, actions, keys)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    ).mean()


@eqx.filter_jit
def forward_backward(
    model: Transformer,
    x: jnp.ndarray,
    y: jnp.ndarray,
    actions: jnp.ndarray,
    key: jax.Array,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, actions, key)
    grad_norm = optax.global_norm(grads)
    return loss, grad_norm, grads


def main(args: Args):
    print("=" * 80)
    print("Memory Benchmark")
    print("=" * 80)

    gc.collect()
    jax.clear_caches()

    print("\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Heads: {args.n_heads}")
    print(f"  FF multiplier: {args.ff_mult}")

    data_path = Path(args.data_dir)
    metadata = np.load(data_path / "metadata.npy", allow_pickle=True).item()
    img_size = metadata["img_size"]
    n_actions = metadata["n_actions"]

    print(f"\nDataset info:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Actions: {n_actions}")

    print_memory_stats("Initial memory")

    print("\nLoading tokenizer...")
    dim = args.patch_size * args.patch_size * 3
    tokenizer = eqx.tree_deserialise_leaves(
        Path(args.tokenizer_path),
        Tokenizer(
            dim,
            thr=args.threshold,
            max_codes=args.codebook_size,
            key=jax.random.PRNGKey(0),
        ),
    )
    vocab_size = tokenizer.max
    block_size = (img_size // args.patch_size) ** 2

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Block size: {block_size}")

    print_memory_stats("After tokenizer load")

    print("\nCreating model...")
    rng = jax.random.PRNGKey(args.seed)
    model_key, data_key, step_key = jax.random.split(rng, 3)

    model = Transformer(
        dim=args.embed_dim,
        depth=args.n_layers,
        block=block_size,
        heads=args.n_heads,
        hdim=args.embed_dim // args.n_heads,
        ff=args.ff_mult,
        drop_e=args.dropout_embed,
        drop_a=args.dropout_attn,
        drop_f=args.dropout_ff,
        vocab=vocab_size,
        n_actions=n_actions,
        k=model_key,
    )

    n_params = count_parameters(model)
    param_size_gb = n_params * 4 / 1e9  # float32

    print(f"\nModel stats:")
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  Parameter memory: {param_size_gb:.3f} GB")

    print_memory_stats("After model creation")

    print("\nCreating dummy data...")
    frames = jax.random.uniform(
        data_key,
        (args.batch_size, 3, args.seq_len, img_size, img_size),
        minval=0,
        maxval=1,
    )
    actions = jax.random.randint(
        data_key, (args.batch_size, args.seq_len), 0, n_actions
    )

    codes = tokenizer(frames, args.patch_size)
    x = codes[:, :-1].reshape(args.batch_size, -1)
    y = codes[:, 1:].reshape(args.batch_size, -1)
    acts = actions[:, :-1]

    total_tokens = x.size
    sequence_memory_gb = total_tokens * 4 / 1e9  # int32 tokens

    print(f"\nSequence stats:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens per batch: {x.shape[1]}")
    print(f"  Token memory: {sequence_memory_gb:.3f} GB")

    print_memory_stats("After data creation")

    print("\nRunning forward pass...")
    loss, grad_norm, grads = forward_backward(model, x, y, acts, step_key)
    loss.block_until_ready()

    print(f"\nResults:")
    print(f"  Loss: {float(loss):.4f}")
    print(f"  Grad norm: {float(grad_norm):.2f}")

    print_memory_stats("After forward/backward pass")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    final_stats = get_memory_stats()
    if final_stats:
        stat = final_stats[0]
        used = stat["used_gb"]
        peak = stat["peak_gb"]
        limit = stat["limit_gb"]

        print(f"\nMemory usage:")
        print(f"  Peak memory: {peak:.2f} GB / {limit:.2f} GB ({peak/limit*100:.1f}%)")
        print(f"  Headroom: {limit - peak:.2f} GB")

        print(f"\nEstimated limits (keeping 10% headroom):")
        target_memory = limit * 0.9
        scale_factor = target_memory / peak

        if scale_factor > 1:
            print(
                f"  Could increase batch size to: ~{int(args.batch_size * scale_factor)}"
            )
            print(f"  OR sequence length to: ~{int(args.seq_len * scale_factor)}")
            print(f"  OR model dim to: ~{int(args.embed_dim * (scale_factor ** 0.5))}")
        else:
            print(f"  ⚠️  Current config uses {peak/limit*100:.1f}% of memory!")
            print(f"  Need to reduce by factor of {1/scale_factor:.2f}")

    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    tyro.cli(main)
