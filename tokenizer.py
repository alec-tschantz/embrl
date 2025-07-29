import jax
import jax.lax as lax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int32


def extract_patches(
    x: Float[Array, "B C T H W"], p: int
) -> Float[Array, "B T Hp Wp D"]:
    B, C, T, H, W = x.shape
    Hp, Wp, D = H // p, W // p, p * p * C
    x = x.reshape(B, C, T, Hp, p, Wp, p)
    x = x.transpose(0, 2, 3, 5, 4, 6, 1)
    return x.reshape(B, T, Hp, Wp, D)


def reconstruct_from_patches(
    p: Float[Array, "B T Hp Wp D"], psz: int, c: int
) -> Float[Array, "B C T H W"]:
    """Inverse of `extract_patches`."""
    B, T, Hp, Wp, _ = p.shape
    p = p.reshape(B, T, Hp, Wp, psz, psz, c)
    p = p.transpose(0, 6, 1, 2, 4, 3, 5)
    return p.reshape(B, c, T, Hp * psz, Wp * psz)


class Tokenizer(eqx.Module):

    codes: Float[Array, "N D"]
    active: Array
    max: int
    dim: int
    thr: float
    noc: int = -1

    def __init__(self, dim: int, thr: float, max_codes: int, key):
        self.dim, self.thr, self.max = dim, thr, max_codes
        self.codes = jnp.zeros((max_codes, dim))
        self.active = jnp.zeros(max_codes, dtype=bool)

    # ── internal helpers ───────────────────────────
    def _d2(self, x):
        x2 = (x**2).sum(-1, keepdims=True)
        c2 = (self.codes**2).sum(-1)
        d = x2 + c2 - 2 * jnp.einsum("bsd,nd->bsn", x, self.codes)
        return jnp.where(self.active[None, None, :], d, jnp.inf)

    def _grow(self, x, mask):
        flat, mflat = x.reshape(-1, self.dim), mask.reshape(-1)

        def body(carry, inp):
            codes, act, ptr = carry
            vec, flag = inp

            def add(carry_inner):
                c, a, p = carry_inner
                c = c.at[p].set(vec)
                a = a.at[p].set(True)
                return c, a, p + 1

            return lax.cond(flag & (ptr < self.max), add, lambda z: z, carry), None

        start_ptr = self.active.sum()
        (codes, act, _), _ = lax.scan(
            body, (self.codes, self.active, start_ptr), (flat, mflat)
        )
        return codes, act

    def tokenize(self, x, *, train: bool = True):
        """Vector‑quantise `x`: (B,S,D) → (B,S) indices."""
        if not bool(self.active.any()):
            c, a = self._grow(x, jnp.ones(x.shape[:2], dtype=bool))
            self = eqx.tree_at(lambda t: (t.codes, t.active), self, (c, a))

        d = self._d2(x)
        idx = d.argmin(-1)
        ok = jnp.take_along_axis(d, idx[..., None], -1)[..., 0] <= self.thr

        c, a = lax.cond(
            (~ok).any() & train,
            lambda: self._grow(x, ~ok),
            lambda: (self.codes, self.active),
        )
        tok = eqx.tree_at(lambda t: (t.codes, t.active), self, (c, a))
        return (idx if train else jnp.where(ok, idx, self.noc)), tok

    def forward_tokenize(
        self,
        imgs: Float[Array, "B C T H W"],
        patch_size: int,
        *,
        train: bool = True,
    ):
        """(B,C,T,H,W) → indices (B,T,Hp,Wp)."""
        patches = extract_patches(imgs, patch_size)  # B,T,Hp,Wp,D
        B, T, Hp, Wp, D = patches.shape
        flat = patches.reshape(B, T * Hp * Wp, D)  # B,S,D
        idx, new = self.tokenize(flat, train=train)
        return idx.reshape(B, T, Hp, Wp), new

    def decode(self, idx: Int32[Array, "B S"]) -> Float[Array, "B S D"]:
        """Indices → flattened patch vectors (zeros for unknown codes)."""
        valid = idx != self.noc
        idx = jnp.clip(idx, 0, self.max - 1)
        return jnp.where(valid[..., None], self.codes[idx], 0.0)
