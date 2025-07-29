# main.py
from pathlib import Path
from typing import Tuple

import equinox as eqx
import jax
import jax.image
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from tokenizer import Tokenizer, reconstruct_from_patches
from transformer import Transformer

IMG_SIZE, PATCH, FRAMES_T, BURN_IN = 64, 8, 20, 10
BATCH, BUFFER_SZ, UPDATES, EVAL_EVERY = 16, 10_000, 10_000, 100
LR, CODEBOOK = 3e-4, 1_024


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
        px = obs["pixels"] if isinstance(obs, dict) else obs
        frames.append(_preprocess(px, size))
    return jnp.stack(frames)


def _sample(buf: jnp.ndarray, b: int, t: int, key: jax.Array) -> jnp.ndarray:
    total, c, h, w = buf.shape
    starts = jax.random.randint(key, (b,), 0, total - t + 1)

    def _clip(s):
        win = jax.lax.dynamic_slice(buf, (s, 0, 0, 0), (t, c, h, w))
        return jnp.transpose(win, (1, 0, 2, 3))

    return jax.vmap(_clip)(starts)


def _generate(
    tok: Tokenizer,
    tr: Transformer,
    prompt: jnp.ndarray,
    steps: int,
    temp: float,
    key: jax.Array,
) -> jnp.ndarray:
    B, C, K, H, W = prompt.shape
    hp, wp = H // PATCH, W // PATCH
    ppf = hp * wp
    seq, _ = tok.forward_tokenize(prompt, PATCH, train=False)
    seq = seq.reshape(B, -1)
    rng = key
    for _ in range(steps - K):
        rng, sub = jax.random.split(rng)
        logits = jax.vmap(lambda s, k: tr(s, key=k))(seq, jax.random.split(sub, B))
        next_log = logits[:, -ppf:]
        rng, sub = jax.random.split(rng)
        new_tok = jax.random.categorical(sub, next_log / temp, axis=-1)
        seq = jnp.concatenate([seq, new_tok], axis=1)
    decoded = tok.decode(seq).reshape(B, steps, hp, wp, PATCH * PATCH * 3)
    return reconstruct_from_patches(decoded, PATCH, C)


def main() -> None:
    wandb.init(project="embrl")
    rng = jax.random.PRNGKey(0)
    rng, dk = jax.random.split(rng)
    frames = _collect_frames(BUFFER_SZ, IMG_SIZE, dk)
    rng, ktok, ktr = jax.random.split(rng, 3)
    tok = Tokenizer(dim=PATCH * PATCH * 3, thr=0.5, max_codes=CODEBOOK, key=ktok)
    tr = Transformer(
        dim=256,
        depth=6,
        block=(IMG_SIZE // PATCH) ** 2,
        heads=8,
        hdim=32,
        ff=4.0,
        drop_e=0.1,
        drop_a=0.1,
        drop_f=0.1,
        vocab=CODEBOOK,
        k=ktr,
    )
    opt = optax.adamw(LR)
    opt_state = opt.init(eqx.filter(tr, eqx.is_array))

    @eqx.filter_value_and_grad
    def _loss(model, x, y, key):
        logits = jax.vmap(lambda s, k: model(s, key=k))(
            x, jax.random.split(key, x.shape[0])
        )
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        ).mean()

    @eqx.filter_jit
    def _step(model, opt_state, x, y, key):
        loss, grads = _loss(model, x, y, key)
        updates, o_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return loss, model, o_state

    for step in range(UPDATES):
        rng, bk, lk = jax.random.split(rng, 3)
        batch = _sample(frames, BATCH, FRAMES_T, bk)
        codes, tok = tok.forward_tokenize(batch, PATCH, train=True)
        inp = codes[:, :-1].reshape(BATCH, -1)
        tgt = codes[:, 1:].reshape(BATCH, -1)
        loss, tr, opt_state = _step(tr, opt_state, inp, tgt, lk)
        if step % 10 == 0:
            wandb.log({"loss": float(loss)}, step=step)
        if step % EVAL_EVERY == 0:
            rng, gk = jax.random.split(rng)
            rollout = _generate(tok, tr, batch[:, :, :BURN_IN], FRAMES_T, 1.0, gk)
            gt = np.array(batch[0].transpose(1, 0, 2, 3))
            pr = np.array(rollout[0].transpose(1, 0, 2, 3))

            T_show = FRAMES_T - BURN_IN
            fig, ax = plt.subplots(2, T_show, figsize=(T_show * 1.4, 3))
            for i, t in enumerate(range(BURN_IN, FRAMES_T)):
                gt_img = gt[t].transpose(1, 2, 0)
                pr_img = pr[t].transpose(1, 2, 0)
                ax[0, i].imshow(gt_img.clip(0, 1))
                ax[0, i].axis("off")
                ax[1, i].imshow(pr_img.clip(0, 1))
                ax[1, i].axis("off")
            plt.tight_layout()
            wandb.log({"rollout": wandb.Image(fig)}, step=step)
            plt.close(fig)


if __name__ == "__main__":
    main()
