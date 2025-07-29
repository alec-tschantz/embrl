# transformer.py
import jax, jax.numpy as jnp, equinox as eqx, math
from jaxtyping import Int32, Array, Float


def _pos(L, D):
    pos = jnp.arange(L)[:, None]
    div = jnp.exp(jnp.arange(0, D, 2) * -(math.log(10000) / D))
    pe = jnp.zeros((L, D))
    pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
    return pe


def _mask(N, B):
    blk = jnp.tril(jnp.ones(((N + B - 1) // B,) * 2, bool))
    i = jnp.arange(N) // B
    return blk[i[:, None], i]


class Attention(eqx.Module):
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear
    B: int
    h: int
    dh: int
    drop: eqx.nn.Dropout

    def __init__(self, d, h, dh, B, dr, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q, self.k, self.v, self.o = (
            eqx.nn.Linear(d, h * dh, key=k, use_bias=False) for k in (k1, k2, k3, k4)
        )
        self.h, self.dh, self.drop = h, dh, eqx.nn.Dropout(dr)
        self.B = B

    def __call__(self, x, mask, key):
        S = x.shape[0]
        q = self.q(x).reshape(S, self.h, self.dh).transpose(1, 0, 2)
        k = self.k(x).reshape(S, self.h, self.dh).transpose(1, 0, 2)
        v = self.v(x).reshape(S, self.h, self.dh).transpose(1, 0, 2)
        s = jnp.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(self.dh)
        s = jnp.where(mask[None], s, -jnp.inf)
        a = jax.nn.softmax(s, -1)
        if key is not None:
            a = jax.vmap(lambda y: self.drop(y, key=key))(a)
        out = jnp.matmul(a, v).transpose(1, 0, 2).reshape(S, -1)
        return self.o(out)


class Block(eqx.Module):
    n1: eqx.nn.RMSNorm
    att: Attention
    n2: eqx.nn.RMSNorm
    ff: eqx.nn.Linear

    def __init__(self, d, h, dh, B, ff, da, df, key):
        k1, k2 = jax.random.split(key)
        self.n1, self.n2 = eqx.nn.RMSNorm(d), eqx.nn.RMSNorm(d)
        self.att = Attention(d, h, dh, B, da, k1)
        self.ff = eqx.nn.Linear(d, int(d * ff), key=k2)

    def __call__(self, x, mask, key):
        x = x + self.att(self.n1(x), mask, key)
        return x + jax.nn.gelu(self.ff(self.n2(x)))


class Transformer(eqx.Module):
    emb: eqx.nn.Embedding
    pos: Float[Array, "L D"]
    blocks: list
    norm: eqx.nn.RMSNorm
    head: eqx.nn.Linear

    def __init__(
        self, dim, depth, block, heads, hdim, ff, drop_e, drop_a, drop_f, vocab, k
    ):
        self.emb = eqx.nn.Embedding(vocab, dim, key=k)
        self.pos = _pos(4096, dim)
        keys = jax.random.split(k, depth)
        self.blocks = [
            Block(dim, heads, hdim, block, ff, drop_a, drop_f, kk) for kk in keys
        ]
        self.norm, self.head = eqx.nn.RMSNorm(dim), eqx.nn.Linear(dim, vocab, key=k)

    def __call__(self, seq, key=None):
        x = self.emb(seq) + self.pos[: seq.shape[1]]
        mask = _mask(seq.shape[1], self.blocks[0].att.B)
        for b in self.blocks:
            x = b(x, mask, key)
        return jax.vmap(self.head)(self.norm(x))
