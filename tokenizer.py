import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
from jaxtyping import Array, Float, Int32, Bool


def extract_patches(
    x: Float[Array, "B C T H W"], p: int
) -> Float[Array, "B T Hp Wp D"]:
    B, C, T, H, W = x.shape
    Hp, Wp, D = H // p, W // p, p * p * C
    x = x.reshape(B, C, T, Hp, p, Wp, p)
    x = x.transpose(0, 2, 3, 5, 4, 6, 1)
    return x.reshape(B, T, Hp, Wp, D)


def reconstruct_from_patches(
    patches: Float[Array, "B T Hp Wp D"], p: int, c: int
) -> Float[Array, "B C T H W"]:
    B, T, Hp, Wp, _ = patches.shape
    patches = patches.reshape(B, T, Hp, Wp, p, p, c)
    patches = patches.transpose(0, 6, 1, 2, 4, 3, 5)
    return patches.reshape(B, c, T, Hp * p, Wp * p)


class Tokenizer(eqx.Module):
    codes: Float[Array, "N D"]
    active: Bool[Array, "N"]
    max: int
    dim: int
    thr: float
    noc: int = -1

    def __init__(self, dim: int, thr: float = 0.75, max_codes: int = 4096, *, key):
        self.dim = dim
        self.thr = thr
        self.max = max_codes
        self.codes = jnp.zeros((max_codes, dim), dtype=jnp.float32)
        self.active = jnp.zeros(max_codes, dtype=bool)

    def _sqdists(self, x: Float[Array, "... D"]) -> Float[Array, "... N"]:
        x2 = jnp.sum(x**2, axis=-1, keepdims=True)
        c2 = jnp.sum(self.codes**2, axis=-1)
        d = x2 + c2 - 2 * jnp.einsum("...d,nd->...n", x, self.codes)
        return jnp.where(self.active, d, jnp.inf)

    @eqx.filter_jit
    def __call__(
        self, imgs: Float[Array, "B C T H W"], patch_size: int
    ) -> Int32[Array, "B T Hp Wp"]:
        patches = extract_patches(imgs, patch_size)
        B, T, Hp, Wp, _ = patches.shape
        flat = patches.reshape(-1, self.dim)
        d = self._sqdists(flat)
        best = d.argmin(-1)
        keep = jnp.take_along_axis(d, best[..., None], -1)[..., 0] <= self.thr
        idx = jnp.where(keep, best, self.noc)
        return idx.reshape(B, T, Hp, Wp)

    @eqx.filter_jit
    def update(self, imgs: Float[Array, "B C T H W"], patch_size: int) -> "Tokenizer":
        patches = extract_patches(imgs, patch_size)
        flat = patches.reshape(-1, self.dim)

        def body(carry, vec):
            codes, active, ptr = carry
            d = jnp.sum(vec**2) + jnp.sum(codes**2, axis=1) - 2 * jnp.dot(codes, vec)
            d = jnp.where(active, d, jnp.inf)
            add = (d.min() > self.thr) & (ptr < self.max)
            codes = lax.cond(add, lambda c: c.at[ptr].set(vec), lambda c: c, codes)
            active = lax.cond(add, lambda a: a.at[ptr].set(True), lambda a: a, active)
            ptr = ptr + lax.convert_element_type(add, jnp.int32)
            return (codes, active, ptr), None

        ptr0 = jnp.sum(self.active, dtype=jnp.int32)
        (codes, active, _), _ = lax.scan(body, (self.codes, self.active, ptr0), flat)
        return eqx.tree_at(lambda t: (t.codes, t.active), self, (codes, active))

    def decode(
        self,
        idx: Int32[Array, "B T Hp Wp"],
        patch_size: int,
        channels: int,
    ) -> Float[Array, "B C T H W"]:
        B, T, Hp, Wp = idx.shape
        flat = idx.reshape(-1)
        valid = flat != self.noc
        flat = jnp.clip(flat, 0, self.max - 1)
        vecs = jnp.where(valid[:, None], self.codes[flat], 0.0)
        patches = vecs.reshape(B, T, Hp, Wp, self.dim)
        return reconstruct_from_patches(patches, patch_size, channels)
