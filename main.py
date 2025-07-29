# main.py
from pathlib import Path
from typing import Tuple

import equinox as eqx
import jax
import jax.image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from craftax.craftax_env import make_craftax_env_from_name


from tokenizer import Tokenizer, extract_patches, reconstruct_from_patches
from transformer import Transformer


# ──────────────── hyper‑parameters ─────────────────
IMG_SIZE = 64
PATCH = 8
FRAMES_T = 6  # sequence length
BATCH = 16
BUFFER_SZ = 5_000
UPDATES = 1_000
EVAL_EVERY = 100
LR = 3e-4
BURN_IN = 2  # frames fed to generator
CODEBOOK = 1024  # max codes


# ──────────────── data utils ───────────────────────
def _preprocess(rgb: jnp.ndarray, size: int) -> jnp.ndarray:
    rgb = rgb.astype(jnp.float32) / 255.0
    if rgb.shape[-1] == 3:
        rgb = jnp.transpose(rgb, (2, 0, 1))
    if rgb.shape[1:] != (size, size):
        rgb = jax.image.resize(rgb, (3, size, size), method="bilinear")
    return rgb


def _collect_frames(n: int, size: int, key: jax.Array) -> jnp.ndarray:
    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    params = env.default_params
    rng = key
    obs, state = env.reset(rng, params)
    buf = []
    for _ in range(n):
        rng, ak, sk = jax.random.split(rng, 3)
        act = env.action_space(params).sample(ak)
        obs, state, *_ = env.step(sk, state, act, params)
        px = obs["pixels"] if isinstance(obs, dict) else obs
        buf.append(_preprocess(px, size))
    return jnp.stack(buf)  # (T,3,H,W)


def _sample(frames: jnp.ndarray, b: int, t: int, key: jax.Array) -> jnp.ndarray:
    total, c, h, w = frames.shape
    starts = jax.random.randint(key, (b,), 0, total - t + 1)

    def _clip(s):
        win = jax.lax.dynamic_slice(frames, (s, 0, 0, 0), (t, c, h, w))
        return jnp.transpose(win, (1, 0, 2, 3))  # (C,T,H,W)

    return jax.vmap(_clip)(starts)


# ──────────────── generation ───────────────────────
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
    seq, _ = tok.forward_tokenize(prompt, False)  # (B,K,hp,wp)
    seq = seq.reshape(B, -1)
    rng = key
    for _ in range(steps - K):
        rng, sub = jax.random.split(rng)
        logits = tr(seq, sub)  # (B,S,V)
        nxt = logits[:, -ppf:]
        rng, sub = jax.random.split(rng)
        new = jax.random.categorical(sub, nxt / temp, -1)
        seq = jnp.concatenate([seq, new], 1)
    seq = seq.reshape(B, -1, hp, wp)
    return tok.decode(seq)  # (B,C,T,H,W)


# ──────────────── training loop ────────────────────
def main() -> None:
    rng = jax.random.PRNGKey(0)

    # buffer
    rng, dk = jax.random.split(rng)
    frames = _collect_frames(BUFFER_SZ, IMG_SIZE, dk)

    # model components
    rng, ktok, ktr = jax.random.split(rng, 3)
    tok = Tokenizer(PATCH * PATCH * 3, 0.5, CODEBOOK, ktok)
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

    # optimiser
    opt = optax.adamw(LR)
    opt_state = opt.init(eqx.filter(tr, eqx.is_array))

    @eqx.filter_value_and_grad
    def _loss(m, x, y, k):
        logits = m(x, k)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        ).mean()

    @eqx.filter_jit
    def _step(m, state, x, y, k):
        l, g = _loss(m, x, y, k)
        upd, state = opt.update(g, state, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, upd)
        return l, m, state

    for step in range(1, UPDATES + 1):
        rng, bk, lk = jax.random.split(rng, 3)
        batch = _sample(frames, BATCH, FRAMES_T, bk)

        # tokenizer growth outside JIT
        codes, tok = tok.forward_tokenize(batch, True)
        inp = codes[:, :-1].reshape(BATCH, -1)
        tgt = codes[:, 1:].reshape(BATCH, -1)

        loss, tr, opt_state = _step(tr, opt_state, inp, tgt, lk)

        if step % 10 == 0:
            print(f"{step:04d}  loss={loss:.4f}")

        if step % EVAL_EVERY == 0:
            rng, gk = jax.random.split(rng)
            rollout = _generate(tok, tr, batch[:, :, :BURN_IN], FRAMES_T, 1.0, gk)
            gt = batch[0].transpose(1, 0, 2, 3)
            pr = rollout[0].transpose(1, 0, 2, 3)
            fig, ax = plt.subplots(2, FRAMES_T, figsize=(FRAMES_T * 1.4, 3))
            for t in range(FRAMES_T):
                ax[0, t].imshow(gt[t])
                ax[0, t].axis("off")
                ax[1, t].imshow(pr[t])
                ax[1, t].axis("off")
            ax[0, 0].set_ylabel("GT")
            ax[1, 0].set_ylabel("Pred")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
