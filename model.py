# model.py
from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32

from tokenizer import NearestNeighborTokenizer
from transformer import BlockTransformer


# ──────────────────────────────────────────────────────────────────────────
# Patch helpers
# ──────────────────────────────────────────────────────────────────────────
def _extract_patches(
    imgs: Float[Array, "B C T H W"], patch_size: int
) -> Float[Array, "B T Hp Wp D"]:
    b, c, t, h, w = imgs.shape
    hp, wp = h // patch_size, w // patch_size
    d = patch_size * patch_size * c

    imgs = imgs.reshape(b, c, t, hp, patch_size, wp, patch_size)
    imgs = jnp.transpose(imgs, (0, 2, 3, 5, 4, 6, 1))                   # B T Hp Wp p p C
    return imgs.reshape(b, t, hp, wp, d)


def _reconstruct_from_patches(
    patches: Float[Array, "B T Hp Wp D"], patch_size: int, channels: int
) -> Float[Array, "B C T H W"]:
    b, t, hp, wp, d = patches.shape
    h, w = hp * patch_size, wp * patch_size
    patches = patches.reshape(b, t, hp, wp, patch_size, patch_size, channels)
    patches = jnp.transpose(patches, (0, 6, 1, 2, 4, 3, 5))
    return patches.reshape(b, channels, t, h, w)


# ──────────────────────────────────────────────────────────────────────────
# World‑model
# ──────────────────────────────────────────────────────────────────────────
class Model(eqx.Module):
    """
    A *stateless* (pure‑function) module apart from its tokenizer codebook,
    which grows online as in Dedieu et al. (2025).  The class now provides
    only **inference utilities**; training logic & loss live in `main.py`.
    """

    tokenizer: NearestNeighborTokenizer
    transformer: BlockTransformer
    to_logits: eqx.nn.Linear

    # misc
    image_size: int
    patch_size: int
    channels: int
    patches_per_image: int

    # ─── Construction ────────────────────────────────────────────────
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        channels: int,
        tokenizer_kwargs: dict,
        transformer_kwargs: dict,
        key: jax.Array,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels

        patches_per_axis = image_size // patch_size
        self.patches_per_image = patches_per_axis ** 2
        patch_dim = patch_size * patch_size * channels

        # force transformer block‑size = num patches in single frame
        transformer_kwargs = transformer_kwargs.copy()
        transformer_kwargs["block_size"] = self.patches_per_image

        # ─ Tokeniser
        key, sub = jax.random.split(key)
        self.tokenizer = NearestNeighborTokenizer(
            dim=patch_dim, **tokenizer_kwargs, key=sub
        )

        # ─ Transformer
        transformer_kwargs["vocab_size"] = self.tokenizer.max_codes
        key, sub = jax.random.split(key)
        self.transformer = BlockTransformer(key=sub, **transformer_kwargs)

        # ─ Projection
        key, sub = jax.random.split(key)
        self.to_logits = eqx.nn.Linear(
            transformer_kwargs["dim"], self.tokenizer.max_codes, key=sub
        )

    # ─── Public helpers ───────────────────────────────────────────────
    def forward_tokenize(
        self,
        imgs: Float[Array, "B C T H W"],
        *,
        training: bool,
    ) -> Tuple[Int32[Array, "B T Hp Wp"], "Model"]:
        """Patchify + tokenise – returns (codes, possibly‑updated model)."""
        patches = _extract_patches(imgs, self.patch_size)               # (B,T,Hp,Wp,D)
        b, t, hp, wp, d = patches.shape
        flat = patches.reshape(b, t * hp * wp, d)                       # (B,seq,D)

        indices, new_tok = self.tokenizer.tokenize(flat, training)
        indices = indices.reshape(b, t, hp, wp)

        return indices, eqx.tree_at(lambda m: m.tokenizer, self, new_tok)

    def decode_tokens(
        self, indices: Int32[Array, "B T Hp Wp"]
    ) -> Float[Array, "B C T H W"]:
        """Vector‑quantised codes → reconstructed images."""
        b, t, hp, wp = indices.shape
        flat = indices.reshape(b, t * hp * wp)
        patches = self.tokenizer.decode(flat)                           # (B,seq,D)
        patches = patches.reshape(b, t, hp, wp, -1)
        return _reconstruct_from_patches(patches, self.patch_size, self.channels)

    # ─── Core forward (no loss) ───────────────────────────────────────
    def __call__(
        self,
        token_seq: Int32[Array, "B S"],          # flattened sequence
        *,
        key: Optional[jax.Array] = None,
    ) -> Float[Array, "B S V"]:
        """
        Batched autoregressive forward pass.
        token_seq: (B, S) integer indices  
        Returns logits with identical shape except final vocab dimension.
        """
        def _f(seq, k):
            embeds = self.transformer(seq, key=k)                       # (S,D)
            return jax.vmap(self.to_logits)(embeds)                     # (S,V)

        keys = (
            jax.random.split(key, token_seq.shape[0]) if key is not None
            else [None] * token_seq.shape[0]
        )
        return jax.vmap(_f)(token_seq, jnp.stack(keys))                 # (B,S,V)
