import math
from typing import Sequence, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32


def _block_causal_mask(N: int, B: int) -> jnp.ndarray:
    n_blk = (N + B - 1) // B
    blk = jnp.tril(jnp.ones((n_blk, n_blk), dtype=bool))
    mask = jnp.repeat(jnp.repeat(blk, B, 0), B, 1)
    return mask[:N, :N]


class RotaryEmbedding(eqx.Module):
    inv_freq: Float[Array, "d2"]
    base: float = 10000.0

    def __init__(self, dim: int, *, base: float = 10000.0):
        assert dim % 2 == 0
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        object.__setattr__(self, "base", base)

    def _cos_sin(self, seq_len: int):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        return jnp.cos(freqs), jnp.sin(freqs)

    @staticmethod
    def _apply(x, cos, sin):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).reshape(
            x.shape
        )

    def __call__(self, q, k):
        cos, sin = self._cos_sin(q.shape[1])
        cos = cos[None, ...]
        sin = sin[None, ...]
        return self._apply(q, cos, sin), self._apply(k, cos, sin)


class Attention(eqx.Module):
    proj: eqx.nn.Linear
    o: eqx.nn.Linear
    rope: RotaryEmbedding
    heads: int
    dh: int
    drop: eqx.nn.Dropout

    def __init__(self, d: int, *, heads: int, dim_head: int, dropout: float, key):
        kp, ko = jax.random.split(key, 2)
        self.proj = eqx.nn.Linear(d, 3 * heads * dim_head, key=kp, use_bias=False)
        self.o = eqx.nn.Linear(heads * dim_head, d, key=ko, use_bias=False)
        self.rope = RotaryEmbedding(dim_head)
        self.heads = heads
        self.dh = dim_head
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, x: Float[Array, "S D"], mask: jnp.ndarray, *, key=None):
        qkv = jax.vmap(self.proj)(x)
        S, h, dh = x.shape[0], self.heads, self.dh
        qkv = qkv.reshape(S, 3, h, dh).transpose(1, 2, 0, 3)
        q, k, v = qkv
        q, k = self.rope(q, k)
        sim = jnp.einsum("h s d, h t d -> h s t", q, k) / math.sqrt(dh)
        sim = jnp.where(mask[None], sim, -jnp.inf)
        attn = jax.nn.softmax(sim, axis=-1)
        if key is not None:
            attn = self.drop(attn, key=key)
        out = jnp.einsum("h s t, h t d -> s h d", attn, v).reshape(S, h * dh)
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
        ka, kf = jax.random.split(key)
        self.n = eqx.nn.RMSNorm(d)
        self.att = Attention(d, heads=heads, dim_head=dim_head, dropout=drop_a, key=ka)
        self.ff = FeedForward(d, mult=ff_mult, dropout=drop_f, key=kf)

    def __call__(self, x: Float[Array, "S D"], mask: jnp.ndarray, *, key=None):
        k1, k2 = jax.random.split(key) if key is not None else (None, None)
        x = x + self.att(jax.vmap(self.n)(x), mask, key=k1)
        return self.ff(x, key=k2)


class Transformer(eqx.Module):
    emb: eqx.nn.Embedding
    act_emb: eqx.nn.Embedding
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
        x = jax.vmap(self.emb)(seq)
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
        for blk, kb in zip(self.blocks, keys):
            x = blk(x, mask, key=kb)
        x = jax.vmap(self.norm)(x)
        return jax.vmap(self.head)(x)

    def generate(
        self, tokens: Int32[Array, "L Hp Wp"], actions: Int32[Array, "K"]
    ) -> Int32[Array, "K Hp Wp"]:
        block = self.block_size
        L, Hp, Wp = tokens.shape
        K = int(actions.shape[0])
        init = tokens.reshape(-1)
        max_len = K * block
        buf = jnp.zeros((max_len,), init.dtype).at[: init.shape[0]].set(init)
        start = init.shape[0]
        steps = K - L

        @eqx.filter_jit
        def _gen(model, act, buf, start_len):
            def body(carry, _):
                b, pos = carry
                logits = model(b, act)
                sl = jax.lax.dynamic_slice(
                    logits, (pos - block, 0), (block, logits.shape[1])
                )
                nxt = jnp.argmax(sl, 1).astype(b.dtype)
                b = jax.lax.dynamic_update_slice(b, nxt, (pos,))
                return (b, pos + block), None

            (b, _), _ = jax.lax.scan(body, (buf, start_len), None, length=steps)
            return b

        out = _gen(self, actions, buf, start)
        return out.reshape(K, Hp, Wp)
