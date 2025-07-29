# transformer.py
"""
Causal **block** transformer (single‑example forward; batching is handled
via `jax.vmap` in the caller).  Architecture unchanged except for minor
stylistic clean‑ups and exhaustive type hints.
"""
from __future__ import annotations

import math
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int32


# ──────────────────────────────────────────────────────────────────────────
# Positional + block masks
# ──────────────────────────────────────────────────────────────────────────
def _pos_embed(n: int, d: int) -> Float[Array, "n d"]:
    position = jnp.arange(n, dtype=jnp.float32)[:, None]
    div = jnp.exp(jnp.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe = jnp.zeros((n, d), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div))
    return pe


def _block_causal(n: int, block: int) -> Bool[Array, "n n"]:
    blk = jnp.tril(jnp.ones(( (n + block - 1) // block, ) * 2, dtype=bool))
    idx = jnp.arange(n) // block
    return blk[idx[:, None], idx[None, :]]


# ──────────────────────────────────────────────────────────────────────────
# Core components
# ──────────────────────────────────────────────────────────────────────────
class _Attention(eqx.Module):
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    n_heads: int
    d_head: int
    block: int

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        block: int,
        dropout_rate: float,
        *,
        key: jax.Array,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.wq = eqx.nn.Linear(dim, n_heads * d_head, key=k1, use_bias=False)
        self.wk = eqx.nn.Linear(dim, n_heads * d_head, key=k2, use_bias=False)
        self.wv = eqx.nn.Linear(dim, n_heads * d_head, key=k3, use_bias=False)
        self.wo = eqx.nn.Linear(n_heads * d_head, dim, key=k4, use_bias=False)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.n_heads, self.d_head, self.block = n_heads, d_head, block

    def __call__(
        self, x: Float[Array, "S D"], mask: Bool[Array, "S S"], *, key: Optional[jax.Array]
    ) -> Float[Array, "S D"]:
        s = x.shape[0]
        q = jax.vmap(self.wq)(x).reshape(s, self.n_heads, self.d_head)
        k = jax.vmap(self.wk)(x).reshape(s, self.n_heads, self.d_head)
        v = jax.vmap(self.wv)(x).reshape(s, self.n_heads, self.d_head)

        q = jnp.transpose(q, (1, 0, 2))                                  # H S Dh
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / math.sqrt(self.d_head)
        scores = jnp.where(mask[None], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, -1)

        if key is not None:
            attn = jax.vmap(lambda a: self.dropout(a, key=key))(attn)

        out = jnp.matmul(attn, v)                                         # H S Dh
        out = jnp.transpose(out, (1, 0, 2)).reshape(s, -1)                # S D
        return jax.vmap(self.wo)(out)                                     # S D


class _FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, mult: float, dropout: float, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        hid = int(dim * mult * 2 / 3)
        self.w1 = eqx.nn.Linear(dim, hid, key=k1, use_bias=False)
        self.w2 = eqx.nn.Linear(dim, hid, key=k2, use_bias=False)
        self.w3 = eqx.nn.Linear(hid, dim, key=k3, use_bias=False)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x: Float[Array, "S D"], *, key: Optional[jax.Array]):
        gate = jax.vmap(self.w1)(x)
        up   = jax.vmap(self.w2)(x)
        h = jax.nn.gelu(gate) * up
        if key is not None:
            h = jax.vmap(lambda _h: self.dropout(_h, key=key))(h)
        return jax.vmap(self.w3)(h)


class _Block(eqx.Module):
    norm1: eqx.nn.RMSNorm
    attn: _Attention
    norm2: eqx.nn.RMSNorm
    ffn: _FeedForward

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        block: int,
        mult: float,
        dropout_a: float,
        dropout_f: float,
        *,
        key: jax.Array,
    ):
        k_attn, k_ff = jax.random.split(key)
        self.norm1 = eqx.nn.RMSNorm(dim)
        self.attn  = _Attention(dim, n_heads, d_head, block, dropout_a, key=k_attn)
        self.norm2 = eqx.nn.RMSNorm(dim)
        self.ffn   = _FeedForward(dim, mult, dropout_f, key=k_ff)

    def __call__(
        self, x: Float[Array, "S D"], mask: Bool[Array, "S S"], *, key: Optional[jax.Array]
    ) -> Float[Array, "S D"]:
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None
        y = self.attn(jax.vmap(self.norm1)(x), mask, key=k1)
        x = x + y
        y = self.ffn(jax.vmap(self.norm2)(x), key=k2)
        return x + y


# ──────────────────────────────────────────────────────────────────────────
# Transformer wrapper
# ──────────────────────────────────────────────────────────────────────────
class BlockTransformer(eqx.Module):
    token_embed: Optional[eqx.nn.Embedding]
    pos: Float[Array, "L D"]
    blocks: List[_Block]
    norm: eqx.nn.RMSNorm
    dropout: eqx.nn.Dropout

    dim: int
    block_size: int
    max_len: int

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        ff_expand_factor: float,
        dropout_attn: float,
        dropout_ff: float,
        dropout_embed: float,
        max_seq_len: int,
        vocab_size: int,
        key: jax.Array,
    ):
        self.dim, self.block_size, self.max_len = dim, block_size, max_seq_len

        # token embeddings (shared for inputs + output head)
        key, sub = jax.random.split(key)
        self.token_embed = eqx.nn.Embedding(vocab_size, dim, key=sub)

        self.pos = _pos_embed(max_seq_len, dim)
        self.dropout = eqx.nn.Dropout(dropout_embed)

        keys = jax.random.split(key, depth)
        self.blocks = [
            _Block(
                dim,
                num_heads,
                head_dim,
                block_size,
                ff_expand_factor,
                dropout_attn,
                dropout_ff,
                key=k,
            )
            for k in keys
        ]
        self.norm = eqx.nn.RMSNorm(dim)

    # ------------------------------------------------------------------
    def __call__(self, tok: Int32[Array, "S"], *, key: Optional[jax.Array]):
        x = jax.vmap(self.token_embed)(tok)                              # (S,D)
        s = x.shape[0]
        x = x + self.pos[:s]

        if key is not None:
            key, sub = jax.random.split(key)
            x = jax.vmap(lambda v: self.dropout(v, key=sub))(x)

        mask = _block_causal(s, self.block_size)

        keys = (
            jax.random.split(key, len(self.blocks)) if key is not None
            else [None] * len(self.blocks)
        )
        for blk, k in zip(self.blocks, keys):
            x = blk(x, mask, key=k)

        return jax.vmap(self.norm)(x)                                    # (S,D)
