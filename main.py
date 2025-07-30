from dataclasses import dataclass
from typing import Tuple, Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.image
import jax.lax as lax
import numpy as np
import optax
import wandb

from craftax.craftax_env import make_craftax_env_from_name
from tokenizer import Tokenizer
from transformer import Transformer


@dataclass
class Args:
    seed: int = 0
    img_size: int = 63
    patch: int = 7
    seq_len: int = 20
    burn_in: int = 5
    batch: int = 8
    max_buffer: int = 100_000
    test_frac: float = 0.2

    updates: int = 50_000
    eval_every: int = 500
    lr: float = 3e-4
    grad_clip: float = 0.5
    weight_decay: float = 1e-3

    codebook: int = 256
    threshold: float = 0.75
    embed_dim: int = 256
    layers: int = 4
    heads: int = 8

    eval_batches: int = 10
    rollout_len: int = 30
    gif_fps: int = 4

    use_wandb: bool = True
    project: str = "itwm"


def _preprocess(rgb: jnp.ndarray, size: int) -> jnp.ndarray:
    img = rgb.astype(jnp.float32)
    img = jnp.transpose(img, (2, 0, 1))
    return jax.image.resize(img, (3, size, size), method="bilinear")


def _collect_frames_and_actions(
    env, n: int, size: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    params = env.default_params
    rng, _ = jax.random.split(key)
    obs, state = env.reset(rng, params)
    frames, actions = [_preprocess(obs, size)], []
    for _ in range(n):
        rng, akey, skey = jax.random.split(rng, 3)
        act = env.action_space(params).sample(akey)
        actions.append(act)
        obs, state, *_ = env.step(skey, state, act, params)
        frames.append(_preprocess(obs, size))
    return jnp.stack(frames[:-1]), jnp.array(actions)


def _sample(
    frames: jnp.ndarray, actions: jnp.ndarray, b: int, t: int, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    total = frames.shape[0]
    starts = jax.random.randint(key, (b,), 0, total - t + 1)

    def _clip(s):
        frm = lax.dynamic_slice(frames, (s, 0, 0, 0), (t, *frames.shape[1:]))
        act = lax.dynamic_slice(actions, (s,), (t,))
        return jnp.transpose(frm, (1, 0, 2, 3)), act

    return jax.vmap(_clip)(starts)


def loss_fn(
    model: Transformer,
    x: jnp.ndarray,
    y: jnp.ndarray,
    actions: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model)(x, actions, keys)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    ).mean()


@eqx.filter_jit
def _train_step(
    model: Transformer,
    opt_state: optax.OptState,
    x,
    y,
    actions,
    opt: optax.GradientTransformation,
    key: jax.Array,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, actions, key)
    grad_norm = optax.global_norm(grads)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, grad_norm, model, opt_state


def _log_rollout(
    gt: jnp.ndarray, pr: jnp.ndarray, step: int, run: Optional[Any], fps: int
):
    g = np.array(gt[0].transpose(1, 0, 2, 3))
    p = np.array(pr[0].transpose(1, 0, 2, 3))
    frames = []
    for t in range(p.shape[0]):
        top = g[t].transpose(1, 2, 0)
        bottom = p[t].transpose(1, 2, 0)
        frm = np.concatenate([top, bottom], axis=0)
        frames.append((frm.clip(0, 1) * 255).astype(np.uint8))
    video = np.stack(frames).transpose(0, 3, 2, 1)
    if run is not None:
        run.log({"rollout": wandb.Video(video, fps=fps, format="gif")}, step=step)


def evaluate_and_log(
    model: Transformer,
    tokenizer: Tokenizer,
    frames: jnp.ndarray,
    actions: jnp.ndarray,
    args: Args,
    run: Optional[Any],
    step: int,
    rng: jax.Array,
):
    rngs = jax.random.split(rng, args.eval_batches + 2)
    losses = []
    for i in range(args.eval_batches):
        k = rngs[i]
        te_data, te_acts = _sample(frames, actions, args.batch, args.seq_len, k)
        te_codes = tokenizer(te_data, args.patch)
        te_inp = te_codes[:, :-1].reshape(args.batch, -1)
        te_tgt = te_codes[:, 1:].reshape(args.batch, -1)
        te_acts = te_acts[:, :-1]
        losses.append(loss_fn(model, te_inp, te_tgt, te_acts, k))
    test_loss = jnp.stack(losses).mean()
    roll_key = rngs[-2]
    roll_data, roll_acts = _sample(
        frames, actions, args.batch, args.rollout_len, roll_key
    )
    obs_burn = roll_data[:, :, : args.burn_in]
    init_seq = tokenizer(obs_burn, args.patch)
    preds = jax.vmap(model.generate)(init_seq, roll_acts)
    pred_pix = tokenizer.decode(preds, args.patch, 3)
    _log_rollout(roll_data, pred_pix, step, run, args.gif_fps)
    if run is not None:
        run.log({"test_loss": float(test_loss)}, step=step)


def main(args: Args):
    run = (
        wandb.init(project=args.project, config=args.__dict__, save_code=True)
        if args.use_wandb
        else None
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, data_k, tok_k, tr_k, loop_k = jax.random.split(rng, 5)
    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
    frames, actions = _collect_frames_and_actions(
        env, args.max_buffer, args.img_size, data_k
    )
    split = int(frames.shape[0] * (1.0 - args.test_frac))
    tr_frames, te_frames = frames[:split], frames[split:]
    tr_actions, te_actions = actions[:split], actions[split:]
    tokenizer = Tokenizer(
        dim=args.patch * args.patch * 3,
        thr=args.threshold,
        max_codes=args.codebook,
        key=tok_k,
    )
    for i in range(frames.shape[0]):
        tokenizer = tokenizer.update(frames[i : i + 1, :, None, :, :], args.patch)
    print(f"tokenizer ready {int(tokenizer.active.sum())}/{args.codebook}")
    n_actions = env.action_space(env.default_params).n
    model = Transformer(
        dim=args.embed_dim,
        depth=args.layers,
        block=(args.img_size // args.patch) ** 2,
        heads=args.heads,
        hdim=args.embed_dim // args.heads,
        ff=4.0,
        drop_e=0.2,
        drop_a=0.2,
        drop_f=0.2,
        vocab=args.codebook,
        n_actions=n_actions,
        k=tr_k,
    )
    sched = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.updates)
    opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=sched, weight_decay=args.weight_decay),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for step in range(args.updates):
        loop_k, trk, lk, evk = jax.random.split(loop_k, 4)
        tr_data, tr_acts = _sample(tr_frames, tr_actions, args.batch, args.seq_len, trk)
        tr_codes = tokenizer(tr_data, args.patch)
        tr_inp = tr_codes[:, :-1].reshape(args.batch, -1)
        tr_tgt = tr_codes[:, 1:].reshape(args.batch, -1)
        tr_acts = tr_acts[:, :-1]
        train_loss, grad_norm, model, opt_state = _train_step(
            model, opt_state, tr_inp, tr_tgt, tr_acts, opt, lk
        )
        if run and step % 10 == 0:
            weight_norm = optax.global_norm(eqx.filter(model, eqx.is_array))
            run.log(
                {
                    "train_loss": float(train_loss),
                    "grad_norm": float(grad_norm),
                    "weight_norm": float(weight_norm),
                },
                step=step,
            )
        if step % args.eval_every == 0:
            evaluate_and_log(
                model=model,
                tokenizer=tokenizer,
                frames=te_frames,
                actions=te_actions,
                args=args,
                run=run,
                step=step,
                rng=evk,
            )
    if run:
        run.finish()


if __name__ == "__main__":
    main(Args())
