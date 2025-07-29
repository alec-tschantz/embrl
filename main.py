# main.py
"""
Training script for the patch‑based Transformer world‑model
implemented in JAX + Equinox.
References
──────────
• Dedieu et al., “Improving Transformer World Models for Data‑Efficient RL”
  (ICML 2025) :contentReference[oaicite:0]{index=0}
• Craftax repository README for basic environment API and usage examples
  :contentReference[oaicite:1]{index=1}
"""
from __future__ import annotations

import functools
import pathlib
from typing import Tuple

import jax
from jax import lax
import jax.image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import equinox as eqx

from model import Model


# ──────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────
def _preprocess_frame(frame: jnp.ndarray, size: int) -> jnp.ndarray:
    """Convert Craftax RGB image ➞ (C, H, W) float32 tensor in [0, 1]."""
    if frame.dtype != jnp.float32:
        frame = frame.astype(jnp.float32) / 255.0
    if frame.shape[-1] == 3:  # channels‑last → channels‑first
        frame = jnp.transpose(frame, (2, 0, 1))
    if frame.shape[1:] != (size, size):
        frame = jax.image.resize(frame, (3, size, size), method="bilinear")
    return frame


def _collect_craftax_frames(
    *, num_frames: int, image_size: int, key: jax.Array
) -> jnp.ndarray:
    """
    Roll a Craftax **pixel** environment with random actions to build an
    offline buffer of frames. Falls back to Gaussian noise if Craftax is not
    available (so CI runs don’t error out).
    """
    try:
        from craftax import make_craftax_env_from_name  # type: ignore
    except ModuleNotFoundError:
        # Fallback – still deterministic thanks to `key`
        return jax.random.normal(key, (num_frames, 3, image_size, image_size))

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    env_params = env.default_params

    rng = key
    obs, state = env.reset(rng, env_params)
    frames = []

    for _ in range(num_frames):
        rng, akey, skey = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(akey)
        obs, state, *_ = env.step(skey, state, action, env_params)

        # Pixel observation can be `obs` or `obs["pixels"]` depending on version
        frame = obs["pixels"] if isinstance(obs, dict) else obs
        frames.append(_preprocess_frame(frame, image_size))

    return jnp.stack(frames)  # [T, 3, H, W]


def _sample_batch(
    frames: jnp.ndarray,  # (TotalT, C, H, W)
    *,
    batch_size: int,
    time_steps: int,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Random contiguous snippets from the replay buffer using dynamic_slice.

    Returns array with shape (B, C, T, H, W).
    """
    total_t, channels, height, width = frames.shape
    # pick start indices [0, total_t - time_steps)
    starts = jax.random.randint(key, (batch_size,), 0, total_t - time_steps + 1)

    def _slice(start):
        # dynamic_slice requires static slice shape
        window = lax.dynamic_slice(
            frames,
            (start, 0, 0, 0),
            (time_steps, channels, height, width),
        )  # (T, C, H, W)
        # reorder to (C, T, H, W)
        return jnp.transpose(window, (1, 0, 2, 3))

    # vmap over batch of starts → (B, C, T, H, W)
    return jax.vmap(_slice)(starts)


# ──────────────────────────────────────────────────────────────────────────
# Autoregressive generation kept outside the model (clean separation)
# ──────────────────────────────────────────────────────────────────────────
def generate(
    model: Model,
    prompt: jnp.ndarray,  # (B, C, K, H, W)
    total_steps: int,
    *,
    temperature: float,
    key: jax.Array,
) -> jnp.ndarray:
    """Autoregressive rollout with block teacher forcing."""
    b, c, k, h, w = prompt.shape
    h_p, w_p = h // model.patch_size, w // model.patch_size
    pp_frame = h_p * w_p  # patches per frame

    # Tokenise prompt (no codebook growth in inference mode)
    token_ids, _ = model.forward_tokenize(prompt, training=False)  # (B, K, h_p, w_p)
    tokens = token_ids.reshape(b, -1)  # (B, K·ppf)

    rng = key
    for _ in range(total_steps - k):
        rng, sub = jax.random.split(rng)

        logits = model(tokens, key=sub)  # (B, seq, V)
        next_logits = logits[:, -pp_frame:]  # last frame

        rng, sub = jax.random.split(rng)
        new_tok = jax.random.categorical(sub, next_logits / temperature, axis=-1)

        tokens = jnp.concatenate([tokens, new_tok], axis=1)  # grow sequence

    # Decode
    total_t = tokens.shape[1] // pp_frame
    tokens = tokens.reshape(b, total_t, h_p, w_p)
    return model.decode_tokens(tokens)  # (B, C, T, H, W)


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def compute_loss(
    model: Model, batch: jnp.ndarray, key: jax.Array
) -> Tuple[jnp.ndarray, Model]:
    """
    X ▷ (B, C, T, H, W)

    Block teacher forcing – predict patches of **t+1** given tokens for ≤ t.
    """
    b, c, t, h, w = batch.shape
    tokens, model = model.forward_tokenize(batch, training=True)  # (B, T, h_p, w_p)

    inp, tgt = tokens[:, :-1], tokens[:, 1:]  # split frames
    inp_flat = inp.reshape(b, -1)  # (B, seq)
    tgt_flat = tgt.reshape(b, -1)  # (B, seq)

    logits = model(inp_flat, key=key)  # (B, seq, V)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), tgt_flat.reshape(-1)
    ).mean()

    return loss, model


def main() -> None:
    # ─── Hyper‑parameters ───────────────────────────────────────────────
    IMAGE_SIZE = 64
    PATCH_SIZE = 8
    CHANNELS = 3
    TIME_STEPS = 6  # training sequence length
    LR = 3e-4
    BATCH_SIZE = 16
    NUM_FRAMES_BUF = 5_000
    NUM_UPDATES = 1_000
    EVAL_INTERVAL = 100
    BURN_IN = 2  # initial frames fed to generator

    # ─── RNG ────────────────────────────────────────────────────────────
    master_key = jax.random.PRNGKey(42)

    # ─── Dataset ────────────────────────────────────────────────────────
    master_key, data_key = jax.random.split(master_key)
    buffer = _collect_craftax_frames(
        num_frames=NUM_FRAMES_BUF, image_size=IMAGE_SIZE, key=data_key
    )  # (T, 3, H, W)

    # ─── Model ──────────────────────────────────────────────────────────
    master_key, model_key = jax.random.split(master_key)
    model = Model(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        tokenizer_kwargs=dict(distance_threshold=0.5, max_codes=1024),
        transformer_kwargs=dict(
            dim=256,
            depth=6,
            num_heads=8,
            head_dim=32,
            ff_expand_factor=4.0,
            dropout_attn=0.1,
            dropout_ff=0.1,
            dropout_embed=0.1,
            block_size=(IMAGE_SIZE // PATCH_SIZE) ** 2,  # full image
            max_seq_len=TIME_STEPS * (IMAGE_SIZE // PATCH_SIZE) ** 2,
        ),
        key=model_key,
    )

    # ─── Optimiser ──────────────────────────────────────────────────────
    optim = optax.adamw(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Jitted step
    compute_loss_grad = eqx.filter_value_and_grad(compute_loss, has_aux=True)

    @eqx.filter_jit
    def _train_step(
        mdl: Model, state, batch, key
    ) -> Tuple[jnp.ndarray, Model, optax.OptState]:
        (loss, mdl2), grads = compute_loss_grad(mdl, batch, key)
        upd, state = optim.update(grads, state, mdl2)
        mdl2 = eqx.apply_updates(mdl2, upd)
        return loss, mdl2, state

    # ─── Training loop ──────────────────────────────────────────────────
    for step in range(1, NUM_UPDATES + 1):
        master_key, batch_key, loss_key = jax.random.split(master_key, 3)
        batch = _sample_batch(
            buffer, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, key=batch_key
        )

        loss, model, opt_state = _train_step(model, opt_state, batch, loss_key)

        if step % 10 == 0:
            print(f"[{step:04d}]   loss = {loss.item():8.4f}")

        # periodic rollout visualisation
        if step % EVAL_INTERVAL == 0:
            master_key, g_key = jax.random.split(master_key)
            prompt = batch[:, :, :BURN_IN]  # (B,C,K,H,W)
            rollout = generate(
                model, prompt, total_steps=TIME_STEPS, temperature=1.0, key=g_key
            )

            # (optional) quick visual – first sample only
            gt, pred = batch[0].transpose(1, 0, 2, 3), rollout[0].transpose(1, 0, 2, 3)
            fig, axs = plt.subplots(2, TIME_STEPS, figsize=(TIME_STEPS * 1.5, 3))
            for t in range(TIME_STEPS):
                axs[0, t].imshow(gt[t])
                axs[1, t].imshow(pred[t])
                axs[0, t].axis("off")
                axs[1, t].axis("off")
            axs[0, 0].set_ylabel("GT")
            axs[1, 0].set_ylabel("Model")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
