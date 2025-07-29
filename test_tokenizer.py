# test_tokenizer.py
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.image
import numpy as np
import matplotlib.pyplot as plt
from craftax.craftax_env import make_craftax_env_from_name
from tokenizer import Tokenizer, reconstruct_from_patches

IMG_SIZE = 64
PATCH = 8
N_FRAMES = 5000
EVAL_EVERY = 100
BATCH_SIZE = 8
CODEBOOK_SIZE = 4096
THRESHOLD = 0.5


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
    
    print(f"Collecting {n} frames...")
    for i in range(n):
        rng, akey, skey = jax.random.split(rng, 3)
        act = env.action_space(params).sample(akey)
        obs, state, *_ = env.step(skey, state, act, params)
        px = obs["pixels"] if isinstance(obs, dict) else obs
        frames.append(_preprocess(px, size))
    
    print(f"Finished collecting {n} frames")
    return jnp.stack(frames)


def _sample_batch(frames: jnp.ndarray, batch_size: int, max_idx: int, key: jax.Array) -> jnp.ndarray:
    indices = jax.random.choice(key, max_idx, shape=(batch_size,), replace=False)
    return frames[indices]


def visualize_reconstruction(original, reconstructed, step, frames_processed, codes_used):
    n_samples = min(4, original.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    fig.suptitle(f'Step {step} | Frames: {frames_processed} | Active Codes: {codes_used}/{CODEBOOK_SIZE}')
    
    for i in range(n_samples):
        orig_img = np.array(original[i].transpose(1, 2, 0))
        axes[0, i].imshow(orig_img.clip(0, 1))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        recon_img = np.array(reconstructed[i].transpose(1, 2, 0))
        axes[1, i].imshow(recon_img.clip(0, 1))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    mse = jnp.mean((original - reconstructed) ** 2)
    print(f"  Reconstruction MSE: {float(mse):.6f}")


def main():
    key = jax.random.PRNGKey(42)
    key, collect_key, tok_key = jax.random.split(key, 3)
    
    frames = _collect_frames(N_FRAMES, IMG_SIZE, collect_key)
    
    tokenizer = Tokenizer(
        dim=PATCH * PATCH * 3,
        thr=THRESHOLD,
        max_codes=CODEBOOK_SIZE,
        key=tok_key
    )
    
    print(f"\nStarting tokenizer evaluation...")
    print(f"Patch size: {PATCH}x{PATCH}")
    print(f"Codebook size: {CODEBOOK_SIZE}")
    print(f"Threshold: {THRESHOLD}")
    print("-" * 50)
    
    for i in range(N_FRAMES):
        frame = frames[i:i+1]
        frame_with_time = frame[:, :, None, :, :]
        
        _, tokenizer = tokenizer.forward_tokenize(frame_with_time, PATCH, train=True)
        
        if i > 0 and i % EVAL_EVERY == 0:
            print(f"\nStep {i}:")
            
            key, sample_key = jax.random.split(key)
            batch = _sample_batch(frames, BATCH_SIZE, i, sample_key)
            
            batch_with_time = batch[:, :, None, :, :]
            
            codes, _ = tokenizer.forward_tokenize(batch_with_time, PATCH, train=False)
            
            B, T, Hp, Wp = codes.shape
            codes_flat = codes.reshape(B, -1)
            decoded = tokenizer.decode(codes_flat)
            decoded = decoded.reshape(B, T, Hp, Wp, PATCH * PATCH * 3)
            
            reconstructed = reconstruct_from_patches(decoded, PATCH, 3)
            reconstructed = reconstructed.squeeze(2)
            
            active_codes = int(tokenizer.active.sum())
            
            visualize_reconstruction(batch, reconstructed, i, i+1, active_codes)
            
            print(f"  Active codes: {active_codes}/{CODEBOOK_SIZE} ({100*active_codes/CODEBOOK_SIZE:.1f}%)")
            print(f"  Frames processed: {i+1}")


if __name__ == "__main__":
    main()