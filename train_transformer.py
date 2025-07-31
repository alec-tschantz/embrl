import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from tqdm import tqdm

from tokenizer import Tokenizer
from transformer import Transformer


@dataclass
class Args:
    data_dir: str = "data/craftax"
    tokenizer_path: str = "data/checkpoints/tokenizer.eqx"
    output_dir: str = "data/checkpoints"
    checkpoint_path: Optional[str] = None

    seq_len: int = 30
    burn_in: int = 10
    batch_size: int = 16
    patch_size: int = 7

    codebook_size: int = 512
    threshold: float = 0.75

    embed_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: float = 4.0
    dropout_embed: float = 0.1
    dropout_attn: float = 0.1
    dropout_ff: float = 0.1

    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 0.5
    n_updates: int = 50_000

    eval_every: int = 500
    eval_batches: int = 10
    save_every: int = 5000
    rollout_len: int = 40
    gif_fps: int = 4

    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    seed: int = 0


def print_system_info():
    print("=" * 80)
    print("System Information")
    print("=" * 80)

    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Number of devices: {len(devices)}")

    for i, device in enumerate(devices):
        print(f"\nDevice {i}: {device}")
        if hasattr(device, "memory_stats"):
            stats = device.memory_stats()
            print(f"  Memory: {stats['bytes_in_use'] / 1e9:.2f} GB used")
            print(f"  Peak: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")

    print("=" * 80)


def count_parameters(model):
    leaves, _ = jax.tree_util.tree_flatten(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)


def load_data(data_dir: Path):
    metadata = np.load(data_dir / "metadata.npy", allow_pickle=True).item()

    train_frames = np.memmap(
        data_dir / "train" / "frames.npy",
        dtype=np.float32,
        mode="r",
        shape=(metadata["train_frames"], 3, metadata["img_size"], metadata["img_size"]),
    )
    train_actions = np.memmap(
        data_dir / "train" / "actions.npy",
        dtype=np.int32,
        mode="r",
        shape=(metadata["train_frames"],),
    )

    test_frames = np.memmap(
        data_dir / "test" / "frames.npy",
        dtype=np.float32,
        mode="r",
        shape=(metadata["test_frames"], 3, metadata["img_size"], metadata["img_size"]),
    )
    test_actions = np.memmap(
        data_dir / "test" / "actions.npy",
        dtype=np.int32,
        mode="r",
        shape=(metadata["test_frames"],),
    )

    return (train_frames, train_actions), (test_frames, test_actions), metadata


def sample_sequences(
    frames: np.memmap, actions: np.memmap, batch_size: int, seq_len: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_frames = frames.shape[0]
    starts = jax.random.randint(key, (batch_size,), 0, n_frames - seq_len + 1)

    frame_batch = []
    action_batch = []

    for start in starts:
        frame_seq = frames[start : start + seq_len]
        action_seq = actions[start : start + seq_len]
        frame_batch.append(frame_seq)
        action_batch.append(action_seq)

    return jnp.array(frame_batch), jnp.array(action_batch)


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
def train_step(
    model: Transformer,
    opt_state: optax.OptState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    actions: jnp.ndarray,
    opt: optax.GradientTransformation,
    key: jax.Array,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, actions, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, optax.global_norm(grads), model, opt_state


def log_rollout(
    model: Transformer,
    tokenizer: Tokenizer,
    test_data: Tuple[np.memmap, np.memmap],
    args: Args,
    step: int,
    run: Optional[wandb.run],
    key: jax.Array,
):
    roll_frames, roll_actions = sample_sequences(
        test_data[0], test_data[1], 1, args.rollout_len, key
    )

    obs_burn = roll_frames[:, :, : args.burn_in]
    init_seq = tokenizer(obs_burn, args.patch_size)
    preds = jax.vmap(model.generate)(init_seq, roll_actions)
    pred_pix = tokenizer.decode(preds, args.patch_size, 3)

    gt_video = np.array(roll_frames[0].transpose(1, 0, 2, 3))
    pred_video = np.array(pred_pix[0].transpose(1, 0, 2, 3))

    frames = []
    for t in range(pred_video.shape[0]):
        top = gt_video[t].transpose(1, 2, 0)
        bottom = pred_video[t].transpose(1, 2, 0)
        frame = np.concatenate([top, bottom], axis=0)
        frames.append((frame.clip(0, 1) * 255).astype(np.uint8))

    video = np.stack(frames).transpose(0, 3, 2, 1)

    if run is not None:
        run.log(
            {"rollout": wandb.Video(video, fps=args.gif_fps, format="gif")}, step=step
        )


def evaluate(
    model: Transformer,
    tokenizer: Tokenizer,
    test_data: Tuple[np.memmap, np.memmap],
    batch_size: int,
    seq_len: int,
    patch_size: int,
    eval_batches: int,
    key: jax.Array,
) -> float:
    frames, actions = test_data
    losses = []

    for i in range(eval_batches):
        eval_key = jax.random.fold_in(key, i)
        sample_key, loss_key = jax.random.split(eval_key)

        frame_batch, action_batch = sample_sequences(
            frames, actions, batch_size, seq_len, sample_key
        )

        codes = tokenizer(frame_batch, patch_size)
        x = codes[:, :-1].reshape(batch_size, -1)
        y = codes[:, 1:].reshape(batch_size, -1)
        acts = action_batch[:, :-1]

        loss = loss_fn(model, x, y, acts, loss_key)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))


def save_checkpoint(path: Path, model, opt_state, step: int):
    checkpoint = {"model": model, "opt_state": opt_state, "step": step}
    eqx.tree_serialise_leaves(path, checkpoint)


def load_checkpoint(path: Path, model, opt_state):
    checkpoint = eqx.tree_deserialise_leaves(
        path, {"model": model, "opt_state": opt_state}
    )
    return checkpoint["model"], checkpoint["opt_state"], checkpoint["step"]


def main(args: Args):
    print_system_info()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    train_data, test_data, metadata = load_data(Path(args.data_dir))
    img_size = metadata["img_size"]
    n_actions = metadata["n_actions"]

    print(f"  Train frames: {train_data[0].shape[0]:,}")
    print(f"  Test frames: {test_data[0].shape[0]:,}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of actions: {n_actions}")

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
    active_codes = int(tokenizer.active.sum())
    print(f"  Active codes: {active_codes}/{tokenizer.max}")

    block_size = (img_size // args.patch_size) ** 2
    print(f"\nModel configuration:")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Heads: {args.n_heads}")
    print(f"  Block size: {block_size}")
    print(f"  Sequence length: {args.seq_len}")

    rng = jax.random.PRNGKey(args.seed)
    model_key, train_key = jax.random.split(rng)

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
        vocab=tokenizer.max,
        n_actions=n_actions,
        k=model_key,
    )

    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    schedule = optax.cosine_decay_schedule(
        init_value=args.learning_rate, decay_steps=args.n_updates
    )
    opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    start_step = 0
    if args.checkpoint_path:
        print(f"\nLoading checkpoint from {args.checkpoint_path}")
        model, opt_state, start_step = load_checkpoint(
            Path(args.checkpoint_path), model, opt_state
        )
        print(f"  Resuming from step {start_step}")

    if args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "n_params": n_params,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "n_layers": args.n_layers,
                "embed_dim": args.embed_dim,
                "n_heads": args.n_heads,
            },
        )
    else:
        run = None

    print(f"\nTraining for {args.n_updates} steps...")

    with tqdm(total=args.n_updates - start_step, initial=start_step) as pbar:
        for step in range(start_step, args.n_updates):
            train_key, sample_key, step_key = jax.random.split(train_key, 3)

            frame_batch, action_batch = sample_sequences(
                train_data[0], train_data[1], args.batch_size, args.seq_len, sample_key
            )

            codes = tokenizer(frame_batch, args.patch_size)
            x = codes[:, :-1].reshape(args.batch_size, -1)
            y = codes[:, 1:].reshape(args.batch_size, -1)
            acts = action_batch[:, :-1]

            loss, grad_norm, model, opt_state = train_step(
                model, opt_state, x, y, acts, opt, step_key
            )

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")

            if run and step % 10 == 0:
                weight_norm = optax.global_norm(eqx.filter(model, eqx.is_array))
                run.log(
                    {
                        "train_loss": float(loss),
                        "grad_norm": float(grad_norm),
                        "weight_norm": float(weight_norm),
                        "learning_rate": schedule(step),
                    },
                    step=step,
                )

            if step % args.eval_every == 0:
                eval_key = jax.random.fold_in(train_key, step)
                test_loss = evaluate(
                    model,
                    tokenizer,
                    test_data,
                    args.batch_size,
                    args.seq_len,
                    args.patch_size,
                    args.eval_batches,
                    eval_key,
                )

                print(f"\n[Step {step}] Test loss: {test_loss:.4f}")

                if run:
                    run.log({"test_loss": float(test_loss)}, step=step)

                    if step % (args.eval_every * 2) == 0:
                        roll_key = jax.random.fold_in(eval_key, 1000)
                        log_rollout(
                            model, tokenizer, test_data, args, step, run, roll_key
                        )

            if step % args.save_every == 0:
                checkpoint_file = output_path / f"checkpoint_{step}.eqx"
                save_checkpoint(checkpoint_file, model, opt_state, step)
                print(f"\n✓ Saved checkpoint: {checkpoint_file}")

    final_checkpoint = output_path / "final_model.eqx"
    save_checkpoint(final_checkpoint, model, opt_state, args.n_updates)
    print(f"\nFinal model saved to {final_checkpoint}")

    if run:
        run.finish()


if __name__ == "__main__":
    tyro.cli(main)
