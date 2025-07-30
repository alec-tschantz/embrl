import math
from typing import Sequence, Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32


def _pos(L: int, D: int) -> Float[Array, "L D"]:
    pos = jnp.arange(L, dtype=jnp.float32)[:, None]
    div = jnp.exp(jnp.arange(0, D, 2, dtype=jnp.float32) * (-math.log(1e4) / D))
    pe = jnp.zeros((L, D), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
    return pe


def _block_causal_mask(N: int, B: int) -> jnp.ndarray:
    n_blk = (N + B - 1) // B
    blk = jnp.tril(jnp.ones((n_blk, n_blk), dtype=bool))
    mask = jnp.repeat(jnp.repeat(blk, B, 0), B, 1)
    return mask[:N, :N]


class Attention(eqx.Module):
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear
    heads: int
    dh: int
    drop: eqx.nn.Dropout

    def __init__(self, d: int, *, heads: int, dim_head: int, dropout: float, key):
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q = eqx.nn.Linear(d, heads * dim_head, key=kq, use_bias=False)
        self.k = eqx.nn.Linear(d, heads * dim_head, key=kk, use_bias=False)
        self.v = eqx.nn.Linear(d, heads * dim_head, key=kv, use_bias=False)
        self.o = eqx.nn.Linear(heads * dim_head, d, key=ko, use_bias=False)
        self.heads = heads
        self.dh = dim_head
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, x: Float[Array, "S D"], mask: jnp.ndarray, *, key=None):
        S, h, dh = x.shape[0], self.heads, self.dh

        q = jax.vmap(self.q)(x).reshape(S, h, dh).transpose(1, 0, 2)  # (h, S, dh)
        k = jax.vmap(self.k)(x).reshape(S, h, dh).transpose(1, 0, 2)
        v = jax.vmap(self.v)(x).reshape(S, h, dh).transpose(1, 0, 2)

        sim = jnp.einsum("h i d, h j d -> h i j", q, k) / math.sqrt(dh)
        sim = jnp.where(mask[None], sim, -jnp.inf)

        attn = jax.nn.softmax(sim, axis=-1)
        if key is not None:
            attn = self.drop(attn, key=key)

        out = jnp.einsum("h i j, h j d -> i h d", attn, v).reshape(S, h * dh)
        return jax.vmap(self.o)(out)


class FeedForward(eqx.Module):
    n: eqx.nn.RMSNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, d: int, *, mult: float, dropout: float, key):
        k1, k2 = jax.random.split(key)
        hidden = int(d * mult)
        self.n = eqx.nn.RMSNorm(d)
        self.fc1 = eqx.nn.Linear(d, hidden * 2, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, d, key=k2)
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, x: Float[Array, "S D"], *, key=None):
        y = jax.vmap(self.n)(x)
        y = jax.vmap(self.fc1)(y)
        h, g = jnp.split(y, 2, axis=-1)
        y = jax.nn.gelu(h) * g
        y = jax.vmap(self.fc2)(y)
        if key is not None:
            y = self.drop(y, key=key)
        return x + y


class Block(eqx.Module):
    n: eqx.nn.RMSNorm
    att: Attention
    ff: FeedForward

    def __init__(
        self,
        *,
        d: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        drop_a: float,
        drop_f: float,
        key,
    ):
        k_att, k_ff = jax.random.split(key)
        self.n = eqx.nn.RMSNorm(d)
        self.att = Attention(
            d, heads=heads, dim_head=dim_head, dropout=drop_a, key=k_att
        )
        self.ff = FeedForward(d, mult=ff_mult, dropout=drop_f, key=k_ff)

    def __call__(self, x: Float[Array, "S D"], mask: jnp.ndarray, *, key=None):
        k1, k2 = jax.random.split(key) if key is not None else (None, None)
        x = x + self.att(jax.vmap(self.n)(x), mask, key=k1)
        return self.ff(x, key=k2)


class Transformer(eqx.Module):
    emb: eqx.nn.Embedding
    act_emb: eqx.nn.Embedding
    pos: Float[Array, "L D"]
    blocks: Sequence[Block]
    norm: eqx.nn.RMSNorm
    head: eqx.nn.Linear
    drop_e: eqx.nn.Dropout
    block_size: int
    n_actions: int

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        block: int,
        heads: int,
        hdim: int,
        ff: float,
        drop_e: float,
        drop_a: float,
        drop_f: float,
        vocab: int,
        n_actions: int,
        k,
    ):
        k_emb, k_act, k_head, *k_blocks = jax.random.split(k, depth + 3)

        self.emb = eqx.nn.Embedding(vocab, dim, key=k_emb)
        self.act_emb = eqx.nn.Embedding(n_actions, dim, key=k_act)
        self.pos = _pos(8_192, dim)

        self.blocks = [
            Block(
                d=dim,
                heads=heads,
                dim_head=hdim,
                ff_mult=ff,
                drop_a=drop_a,
                drop_f=drop_f,
                key=kb,
            )
            for kb in k_blocks
        ]
        self.norm = eqx.nn.RMSNorm(dim)
        self.head = eqx.nn.Linear(dim, vocab, key=k_head)
        self.drop_e = eqx.nn.Dropout(drop_e)

        self.block_size = block
        self.n_actions = n_actions

    def __call__(
        self,
        seq: Int32[Array, "S"],
        actions: Int32[Array, "T"],
        key: Optional[jax.Array] = None,
    ) -> Float[Array, "S vocab"]:
        S = seq.shape[0]

        x = jax.vmap(self.emb)(seq) + self.pos[:S]
        act_per_token = jnp.repeat(actions, self.block_size)[:S]
        x = x + jax.vmap(self.act_emb)(act_per_token)

        if key is not None:
            key, ke = jax.random.split(key)
            x = self.drop_e(x, key=ke)

        mask = _block_causal_mask(S, self.block_size)
        keys = (
            jax.random.split(key, len(self.blocks))
            if key is not None
            else [None] * len(self.blocks)
        )

        for blk, k_blk in zip(self.blocks, keys):
            x = blk(x, mask, key=k_blk)

        x = jax.vmap(self.norm)(x)
        return jax.vmap(self.head)(x)

    def generate(
        self,
        tokens: Int32[Array, "L Hp Wp"],
        actions: Int32[Array, "K"],
    ) -> Int32[Array, "K Hp Wp"]:
        L, Hp, Wp = tokens.shape
        K = actions.shape[0]

        seq = tokens.reshape(-1)
        for t in range(L, K):
            logits = self(seq, actions[:t])
            next_tok = logits[-self.block_size :].argmax(axis=-1)
            seq = jnp.concatenate([seq, next_tok], axis=0)
        return seq.reshape(K, Hp, Wp)
