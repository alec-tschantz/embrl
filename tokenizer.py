"""
Nearest‑Neighbour Tokeniser (grows online).  Fully JIT‑safe implementation using only pure JAX primitives and `lax` loops.
"""
from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int32


class NearestNeighborTokenizer(eqx.Module):
    # Persistent codebook state
    codes: Float[Array, "max_codes dim"]
    is_active: Bool[Array, "max_codes"]

    # Hyper-parameters (frozen)
    max_codes: int
    no_code_id: int
    distance_threshold: float
    dim: int

    def __init__(
        self,
        *,
        dim: int,
        distance_threshold: float,
        max_codes: int = 16_384,
        no_code_id: int = -1,
        key: jax.Array,
    ):
        """
        Initialize an empty codebook with all codes inactive.
        """
        self.dim = dim
        self.distance_threshold = distance_threshold
        self.max_codes = max_codes
        self.no_code_id = no_code_id

        # Zeroed code vectors and inactive mask
        self.codes = jnp.zeros((max_codes, dim), dtype=jnp.float32)
        self.is_active = jnp.zeros(max_codes, dtype=bool)

    def _l2_sq(
        self,
        x: Float[Array, "B S D"],
        codes: Float[Array, "N D"],
        active: Bool[Array, "N"],
    ) -> Float[Array, "B S N"]:
        """
        Squared L2 distance between each x[b,s] and each active code.
        Inactive codes get "+inf" distance.
        """
        x2 = (x ** 2).sum(-1, keepdims=True)           # (B, S, 1)
        c2 = (codes ** 2).sum(-1)                      # (N,)
        dot = jnp.einsum("bsd,nd->bsn", x, codes)    # (B, S, N)
        dist = x2 + c2[None, None, :] - 2.0 * dot
        return jnp.where(active[None, None, :], dist, jnp.inf)

    def _add_codes(
        self,
        x: Float[Array, "batch seq dim"],
        should_add: Bool[Array, "batch seq"],
        codes: Float[Array, "max_codes dim"],
        is_active: Bool[Array, "max_codes"],
    ) -> Tuple[Float[Array, "max_codes dim"], Bool[Array, "max_codes"]]:
        """
        Add new vectors from x where `should_add` is True into the next
        available slots of the codebook.  Uses only JAX primitives + lax.fori_loop.
        """
        batch, seq, D = x.shape
        flat_x = x.reshape(-1, D)                      # (batch*seq, dim)
        mask_flat = should_add.reshape(-1)             # (batch*seq,)

        # cumulative count to assign positions
        cumsum = jnp.cumsum(mask_flat)
        n_active = jnp.sum(is_active)
        add_indices = cumsum - 1 + n_active            # (batch*seq,)
        add_indices = jnp.where(mask_flat, add_indices, self.max_codes)

        # safe patches array for fallback
        safe_patches = jnp.concatenate([flat_x, jnp.zeros((1, D), flat_x.dtype)], axis=0)
        scatter_src = jnp.where(mask_flat, jnp.arange(flat_x.shape[0], dtype=jnp.int32), flat_x.shape[0])

        def _update(i, cb):
            mask_i = (add_indices == i) & (i < self.max_codes)
            any_i = jnp.any(mask_i)
            sel = jnp.argmax(mask_i)
            vec = safe_patches[scatter_src[sel]]
            return cb.at[i].set(jnp.where(any_i, vec, cb[i]))

        # only loop over new slots
        n_to_add = jnp.sum(mask_flat)
        start = n_active
        end = jnp.minimum(n_active + n_to_add, self.max_codes)
        new_codes = lax.fori_loop(start, end, _update, codes)

        # scatter update is_active mask at dynamic positions
        idxs_to_activate = add_indices[add_indices < self.max_codes]
        new_active = is_active.at[idxs_to_activate].set(True)

        return new_codes, new_active

    def tokenize(
        self,
        x: Float[Array, "B S D"],
        training: bool,
    ) -> Tuple[Int32[Array, "B S"], "NearestNeighborTokenizer"]:
        """
        Map each patch in x to its nearest code.  Grows codebook online when
        training and distances exceed threshold.
        Returns (indices, updated_tokenizer).
        """
        has_codes = self.is_active.any()

        def _init():
            full = jnp.ones(x.shape[:2], bool)
            codes2, act2 = self._add_codes(x, full, self.codes, self.is_active)
            tok2 = eqx.tree_at(lambda m: (m.codes, m.is_active), self, (codes2, act2))
            B, S = x.shape[:2]
            idxs = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
            return idxs, tok2

        def _step():
            dist = self._l2_sq(x, self.codes, self.is_active)
            idxs = dist.argmin(-1).astype(jnp.int32)
            mind = jnp.take_along_axis(dist, idxs[..., None], axis=-1).squeeze(-1)
            ok = mind <= self.distance_threshold

            if training:
                codes2, act2 = lax.cond(
                    (~ok).any(),
                    lambda: self._add_codes(x, ~ok, self.codes, self.is_active),
                    lambda: (self.codes, self.is_active),
                )
                tok2 = eqx.tree_at(lambda m: (m.codes, m.is_active), self, (codes2, act2))
                return idxs, tok2
            else:
                idxs2 = jnp.where(ok, idxs, self.no_code_id)
                return idxs2, self

        return jax.lax.cond(has_codes, _step, _init)

    def decode(self, ids: Int32[Array, "B S"]) -> Float[Array, "B S D"]:
        """
        Convert token indices back to patch vectors; unseen codes → zeros.
        """
        valid = ids != self.no_code_id
        safe = jnp.clip(ids, 0, self.max_codes - 1)
        vecs = self.codes[safe]
        return jnp.where(valid[..., None], vecs, 0.0)
