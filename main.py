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
    batch: int = 32
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


def _collect_frames(n: int, size: int, key: jax.Array) -> jnp.ndarray:
    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    params = env.default_params
    rng, _ = jax.random.split(key)
    obs, state = env.reset(rng, params)
    frames = []
    for _ in range(n):
        rng, akey, skey = jax.random.split(rng, 3)
        act = env.action_space(params).sample(akey)
        obs, state, *_ = env.step(skey, state, act, params)
        frames.append(_preprocess(obs, size))
    return jnp.stack(frames)


def _sample(buf: jnp.ndarray, b: int, t: int, key: jax.Array) -> jnp.ndarray:
    total = buf.shape[0]
    starts = jax.random.randint(key, (b,), 0, total - t + 1)

    def _clip(s):
        win = jax.lax.dynamic_slice(buf, (s, 0, 0, 0), (t, *buf.shape[1:]))
        return jnp.transpose(win, (1, 0, 2, 3))

    return jax.vmap(_clip)(starts)


def train_tokenizer(
    frames: jnp.ndarray, tok: Tokenizer, patch: int, codebook: int
) -> Tokenizer:
    for i in range(frames.shape[0]):
        tok = tok.update(frames[i : i + 1, :, None, :, :], patch)
        if i % 100 == 0:
            print(
                f"frame {i}/{frames.shape[0]} | active {int(tok.active.sum())}/{codebook}"
            )
    print(f"tokenizer ready {int(tok.active.sum())}/{codebook}")
    return tok


def loss_fn(
    model: Transformer, x: jnp.ndarray, y: jnp.ndarray, key: jax.Array
) -> jnp.ndarray:
    logits = jax.vmap(lambda s, k: model(s, key=k))(
        x, jax.random.split(key, x.shape[0])
    )
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    ).mean()


@eqx.filter_jit
def train_step(
    model: Transformer,
    opt_state: optax.OptState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.Array,
    opt: optax.GradientTransformation,
) -> Tuple[jnp.ndarray, Transformer, optax.OptState]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def _generate(
    tok: Tokenizer,
    tr: Transformer,
    prompt: jnp.ndarray,
    steps: int,
    temp: float,
    patch: int,
    key: jax.Array,
) -> jnp.ndarray:
    b, c, k, h, w = prompt.shape
    hp, wp = h // patch, w // patch
    ppf = hp * wp
    seq = tok(prompt, patch).reshape(b, -1)
    rng = key
    for _ in range(steps - k):
        rng, sub = jax.random.split(rng)
        logits = jax.vmap(lambda s, kk: tr(s, key=kk))(seq, jax.random.split(sub, b))
        next_logits = logits[:, -ppf:]
        rng, sub = jax.random.split(rng)
        nxt = jax.random.categorical(sub, next_logits / temp, axis=-1)
        seq = jnp.concatenate([seq, nxt], axis=1)
    seq = seq.reshape(b, steps, hp, wp)
    return tok.decode(seq, patch, c)


def evaluate(
    tokenizer: Tokenizer,
    transformer: Transformer,
    batch: jnp.ndarray,
    seq_len: int,
    burn_in: int,
    patch: int,
    step: int,
    run: Optional[Any],
    key: jax.Array,
) -> None:
    rollout = _generate(
        tokenizer, transformer, batch[:, :, :burn_in], seq_len, 1.0, patch, key
    )
    gt = np.array(batch[0].transpose(1, 0, 2, 3))
    pr = np.array(rollout[0].transpose(1, 0, 2, 3))
    n_show = seq_len - burn_in
    cols = 5
    rows = ceil(n_show / cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.5, rows * 2.5))
    axes = axes.reshape(rows * 2, cols)
    for i in range(n_show):
        r = i // cols
        c = i % cols
        axes[r * 2, c].imshow(gt[burn_in + i].transpose(1, 2, 0).clip(0, 1))
        axes[r * 2, c].axis("off")
        axes[r * 2 + 1, c].imshow(pr[burn_in + i].transpose(1, 2, 0).clip(0, 1))
        axes[r * 2 + 1, c].axis("off")
    flat = axes.flatten()
    used = n_show * 2
    for ax in flat[used:]:
        ax.axis("off")
    plt.tight_layout()
    if run is not None:
        run.log({"rollout": wandb.Image(fig)}, step=step)
    else:
        plt.show()
    plt.close(fig)


def train_transformer(
    frames: jnp.ndarray,
    tokenizer: Tokenizer,
    transformer: Transformer,
    batch: int,
    seq_len: int,
    burn_in: int,
    patch: int,
    updates: int,
    eval_every: int,
    lr: float,
    grad_clip: float,
    run: Optional[Any],
    key: jax.Array,
) -> None:
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(lr))
    opt_state = opt.init(eqx.filter(transformer, eqx.is_array))
    rng = key
    for step in range(updates):
        key, bk, lk = jax.random.split(key, 3)
        batch_data = _sample(frames, batch, seq_len, bk)

        codes = tokenizer(batch_data, patch)
        inp = codes[:, :-1].reshape(batch, -1)
        tgt = codes[:, 1:].reshape(batch, -1)

        loss, transformer, opt_state = train_step(
            transformer, opt_state, inp, tgt, lk, opt
        )
        if run is not None and step % 10 == 0:
            run.log({"loss": float(loss)}, step=step)
        if step % eval_every == 0:
            rng, evk = jax.random.split(rng)
            evaluate(
                tokenizer,
                transformer,
                batch_data,
                seq_len,
                burn_in,
                patch,
                step,
                run,
                evk,
            )


def main(args) -> None:
    run = (
        wandb.init(project=args.project, config=args.__dict__, save_code=True)
        if args.use_wandb
        else None
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, data_k, tok_k, tr_k, train_k = jax.random.split(rng, 5)
    frames = _collect_frames(args.max_buffer, args.img_size, data_k)
    tokenizer = Tokenizer(
        dim=args.patch * args.patch * 3,
        thr=args.threshold,
        max_codes=args.codebook,
        key=tok_k,
    )
    tokenizer = train_tokenizer(frames, tokenizer, args.patch, args.codebook)

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
        k=tr_k,
    )
    train_transformer(
        frames,
        tokenizer,
        transformer,
        args.batch,
        args.seq_len,
        args.burn_in,
        args.patch,
        args.updates,
        args.eval_every,
        args.lr,
        args.grad_clip,
        run,
        train_k,
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(Args())
