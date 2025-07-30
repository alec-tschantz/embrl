import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Optional, Tuple, Any

import equinox as eqx
import jax
import jax.image
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb

from craftax.craftax_env import make_craftax_env_from_name
from tokenizer import Tokenizer
from transformer import Transformer


@dataclass
class Args:
    seed: int = 0
    img_size: int = 63
    patch: int = 7
    seq_len: int = 20
    burn_in: int = 5
    batch: int = 16
    max_buffer: int = 10_000
    updates: int = 10_000
    eval_every: int = 200
    lr: float = 3e-4
    codebook: int = 256
    threshold: float = 0.75
    embed_dim: int = 256
    layers: int = 6
    heads: int = 8
    grad_clip: float = 1.0
    use_wandb: bool = True
    project: str = "itwm"


def _preprocess(rgb: jnp.ndarray, size: int) -> jnp.ndarray:
    img = rgb.astype(jnp.float32)
    img = jnp.transpose(img, (2, 0, 1))
    img = jax.image.resize(img, (3, size, size), method="bilinear")
    return img


def _collect_frames_and_actions(
    env, n: int, size: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    params = env.default_params
    rng, _ = jax.random.split(key)
    obs, state = env.reset(rng, params)
    frames = [_preprocess(obs, size)]
    actions = []
    for _ in range(n):
        rng, akey, skey = jax.random.split(rng, 3)
        act = env.action_space(params).sample(akey)
        actions.append(act)
        obs, state, *_ = env.step(skey, state, act, params)
        frames.append(_preprocess(obs, size))
    return jnp.stack(frames[:-1]), jnp.array(actions)


def _sample(
    frames: jnp.ndarray, actions: jnp.ndarray, b: int, t: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    total = frames.shape[0]
    starts = jax.random.randint(key, (b,), 0, total - t + 1)

    def _clip(s):
        frame_win = jax.lax.dynamic_slice(frames, (s, 0, 0, 0), (t, *frames.shape[1:]))
        act_win = jax.lax.dynamic_slice(actions, (s,), (t,))
        return jnp.transpose(frame_win, (1, 0, 2, 3)), act_win

    batch_frames, batch_actions = jax.vmap(_clip)(starts)
    return batch_frames, batch_actions


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
) -> Tuple[jnp.ndarray, Transformer, optax.OptState]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, actions, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def evaluate(
    ground_truth: jnp.ndarray,
    predicted: jnp.ndarray,
    step: int,
    run: Optional[Any],
) -> None:
    gt = np.array(ground_truth[0].transpose(1, 0, 2, 3))
    pr = np.array(predicted[0].transpose(1, 0, 2, 3))
    n_frames = gt.shape[0]
    rows, cols = ceil(n_frames / 5), 5
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.5, rows * 2.5))
    axes = axes.reshape(rows * 2, cols)
    for i in range(n_frames):
        r, c = i // cols, i % cols
        axes[r * 2, c].imshow(gt[i].transpose(1, 2, 0).clip(0, 1))
        axes[r * 2, c].axis("off")
        axes[r * 2, c].set_title("GT", fontsize=8)
        axes[r * 2 + 1, c].imshow(pr[i].transpose(1, 2, 0).clip(0, 1))
        axes[r * 2 + 1, c].axis("off")
        axes[r * 2 + 1, c].set_title("Pred", fontsize=8)
    flat, used = axes.flatten(), n_frames * 2
    for ax in flat[used:]:
        ax.axis("off")
    plt.tight_layout()
    if run is not None:
        run.log({"rollout": wandb.Image(fig)}, step=step)
    else:
        plt.show()
    plt.close(fig)


def main(args) -> None:
    run = (
        wandb.init(project=args.project, config=args.__dict__, save_code=True)
        if args.use_wandb
        else None
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, data_k, tok_k, tr_k, train_k = jax.random.split(rng, 5)

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    frames, actions = _collect_frames_and_actions(
        env, args.max_buffer, args.img_size, data_k
    )

    tokenizer = Tokenizer(
        dim=args.patch * args.patch * 3,
        thr=args.threshold,
        max_codes=args.codebook,
        key=tok_k,
    )

    for i in range(frames.shape[0]):
        tokenizer = tokenizer.update(frames[i : i + 1, :, None, :, :], args.patch)
    print(f"tokenizer ready {int(tokenizer.active.sum())}/{args.codebook}")

    n_actions = env.action_space(env.default_params).n
    transformer = Transformer(
        dim=args.embed_dim,
        depth=args.layers,
        block=(args.img_size // args.patch) ** 2,
        heads=args.heads,
        hdim=args.embed_dim // args.heads,
        ff=4.0,
        drop_e=0.1,
        drop_a=0.1,
        drop_f=0.1,
        vocab=args.codebook,
        n_actions=n_actions,
        k=tr_k,
    )

    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr))
    opt_state = opt.init(eqx.filter(transformer, eqx.is_array))

    for step in range(args.updates):
        train_k, bk, lk = jax.random.split(train_k, 3)
        batch_data, batch_actions = _sample(
            frames, actions, args.batch, args.seq_len, bk
        )

        codes = tokenizer(batch_data, args.patch)
        inp = codes[:, :-1].reshape(args.batch, -1)
        tgt = codes[:, 1:].reshape(args.batch, -1)
        acts = batch_actions[:, :-1]

        loss, transformer, opt_state = train_step(
            transformer, opt_state, inp, tgt, acts, opt, lk
        )
        if run is not None and step % 10 == 0:
            run.log({"loss": float(loss)}, step=step)

        if step % args.eval_every == 0:
            obs = batch_data[:, :, : args.burn_in]
            init_seq = tokenizer(obs, args.patch)
            pred_seq = jax.vmap(transformer.generate)(init_seq, batch_actions)
            pred_data = tokenizer.decode(pred_seq, args.patch, 3)
            evaluate(
                batch_data,
                pred_data,
                step,
                run,
            )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(Args())
