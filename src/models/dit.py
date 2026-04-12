"""
DiT (Diffusion Transformer) ported to JAX / Equinox.

Exact port of Meta's DiT architecture (github.com/facebookresearch/DiT)
for use with BroDeer's DEER and TAA algorithms.  Loads pretrained PyTorch
weights (e.g. DiT-XL-2-256x256.pt) and runs inference in JAX via the
DDIMDynamics interface.

Architecture reference:  models.py  in ParaTAA-Diffusion
Pipeline reference:      parallel_dit.py  in ParaTAA-Diffusion
"""

import math
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


# ═══════════════════════════════════════════════════════════════════════
#  Positional-embedding utilities  (numpy, computed once at init)
# ═══════════════════════════════════════════════════════════════════════

def _get_1d_sincos_pos_embed(embed_dim, pos):
    """pos: (M,) float array → (M, embed_dim) sin-cos embedding."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega                         # (D/2,)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def _get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Return (grid_size*grid_size, embed_dim) sin-cos positional embedding."""
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)                  # w first
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  Small helpers
# ═══════════════════════════════════════════════════════════════════════

def _layer_norm(x, eps=1e-6):
    """LayerNorm without learnable affine (elementwise_affine=False)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _modulate(x, shift, scale):
    """x: (N, C), shift/scale: (C,) — adaLN modulation."""
    return x * (1.0 + scale[None, :]) + shift[None, :]


# ═══════════════════════════════════════════════════════════════════════
#  Model components
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbed(eqx.Module):
    """Conv2d patch projection  (timm.PatchEmbed equivalent)."""
    proj: eqx.nn.Conv2d
    num_patches: int = eqx.field(static=True)

    def __init__(self, input_size, patch_size, in_channels, embed_dim, *, key):
        self.proj = eqx.nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            use_bias=True, key=key,
        )
        self.num_patches = (input_size // patch_size) ** 2

    def __call__(self, x):
        # x: (C, H, W) → (embed_dim, H/P, W/P)
        x = self.proj(x)
        # flatten spatial → (num_patches, embed_dim)
        return x.reshape(x.shape[0], -1).T


class TimestepEmbedder(eqx.Module):
    """Sinusoidal timestep → MLP embedding."""
    mlp_0: eqx.nn.Linear
    mlp_2: eqx.nn.Linear
    frequency_embedding_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, frequency_embedding_size=256, *, key):
        k1, k2 = jr.split(key)
        self.mlp_0 = eqx.nn.Linear(frequency_embedding_size, hidden_size, key=k1)
        self.mlp_2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t.astype(jnp.float32) * freqs
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)])
        if dim % 2:
            emb = jnp.concatenate([emb, jnp.zeros(1)])
        return emb

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.astype(self.mlp_0.weight.dtype)
        return self.mlp_2(jax.nn.silu(self.mlp_0(t_freq)))


class LabelEmbedder(eqx.Module):
    """Class-label embedding table (with CFG null-class slot)."""
    embedding_table: eqx.nn.Embedding
    num_classes: int = eqx.field(static=True)
    dropout_prob: float = eqx.field(static=True)

    def __init__(self, num_classes, hidden_size, dropout_prob, *, key):
        use_cfg = int(dropout_prob > 0)
        self.embedding_table = eqx.nn.Embedding(
            num_classes + use_cfg, hidden_size, key=key,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def __call__(self, label):
        # Inference only — no label dropout.
        return self.embedding_table(label)


class Attention(eqx.Module):
    """Multi-head self-attention with fused QKV  (timm.Attention equivalent)."""
    qkv:  eqx.nn.Linear
    proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim:  int = eqx.field(static=True)
    scale:   float = eqx.field(static=True)

    def __init__(self, dim, num_heads, *, key):
        k1, k2 = jr.split(key)
        self.qkv  = eqx.nn.Linear(dim, dim * 3, key=k1)
        self.proj = eqx.nn.Linear(dim, dim,     key=k2)
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = (dim // num_heads) ** -0.5

    def __call__(self, x):
        # x: (N, C)
        N, C = x.shape
        qkv = jax.vmap(self.qkv)(x)                          # (N, 3C)
        qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)
        q = qkv[:, 0].transpose(1, 0, 2)                     # (H, N, d)
        k = qkv[:, 1].transpose(1, 0, 2)
        v = qkv[:, 2].transpose(1, 0, 2)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale         # (H, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        out  = attn @ v                                       # (H, N, d)
        out  = out.transpose(1, 0, 2).reshape(N, C)           # (N, C)
        return jax.vmap(self.proj)(out)


class Mlp(eqx.Module):
    """Two-layer MLP with approximate GELU  (timm.Mlp equivalent)."""
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, in_features, hidden_features, *, key):
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(in_features,     hidden_features, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_features,  in_features,    key=k2)

    def __call__(self, x):
        # x: (C,) — call per-token via vmap
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=True))


class DiTBlock(eqx.Module):
    """DiT block with adaLN-Zero conditioning."""
    attn: Attention
    mlp:  Mlp
    adaLN_modulation: eqx.nn.Linear   # SiLU applied before this linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.attn = Attention(hidden_size, num_heads, key=k1)
        self.mlp  = Mlp(hidden_size, int(hidden_size * mlp_ratio), key=k2)
        self.adaLN_modulation = eqx.nn.Linear(
            hidden_size, 6 * hidden_size, key=k3,
        )
        self.hidden_size = hidden_size

    def __call__(self, x, c):
        # x: (N, C), c: (C,) conditioning vector
        ada = self.adaLN_modulation(jax.nn.silu(c))           # (6C,)
        ada = ada.reshape(6, self.hidden_size)
        shift_msa, scale_msa, gate_msa = ada[0], ada[1], ada[2]
        shift_mlp, scale_mlp, gate_mlp = ada[3], ada[4], ada[5]

        # --- self-attention ---
        h = _modulate(_layer_norm(x), shift_msa, scale_msa)
        h = self.attn(h)
        x = x + gate_msa[None, :] * h

        # --- feed-forward ---
        h = _modulate(_layer_norm(x), shift_mlp, scale_mlp)
        h = jax.vmap(self.mlp)(h)
        x = x + gate_mlp[None, :] * h
        return x


class FinalLayer(eqx.Module):
    """DiT final projection with adaLN-Zero."""
    linear: eqx.nn.Linear
    adaLN_modulation: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, patch_size, out_channels, *, key):
        k1, k2 = jr.split(key)
        self.linear = eqx.nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, key=k1,
        )
        self.adaLN_modulation = eqx.nn.Linear(
            hidden_size, 2 * hidden_size, key=k2,
        )
        self.hidden_size = hidden_size

    def __call__(self, x, c):
        # x: (N, C), c: (C,)
        ada = self.adaLN_modulation(jax.nn.silu(c))           # (2C,)
        shift = ada[:self.hidden_size]
        scale = ada[self.hidden_size:]
        x = _modulate(_layer_norm(x), shift, scale)
        return jax.vmap(self.linear)(x)                       # (N, P*P*outC)


# ═══════════════════════════════════════════════════════════════════════
#  Top-level DiT model
# ═══════════════════════════════════════════════════════════════════════

class DiT(eqx.Module):
    """
    Diffusion Transformer (single-example forward, no batch dim).

    Call signature:  model(x, t, y)
      x: (C, H, W)  — single latent image
      t: scalar int  — diffusion timestep (original 1000-step index)
      y: scalar int  — class label  (num_classes = null/uncond label)
    Returns: (out_C, H, W)  where out_C = 2*C when learn_sigma=True.
    """
    x_embedder:  PatchEmbed
    t_embedder:  TimestepEmbedder
    y_embedder:  LabelEmbedder
    pos_embed:   jnp.ndarray          # (num_patches, hidden_size)
    blocks:      list                  # [DiTBlock, ...]
    final_layer: FinalLayer
    # static config
    patch_size:   int = eqx.field(static=True)
    input_size:   int = eqx.field(static=True)
    in_channels:  int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    num_heads:    int = eqx.field(static=True)

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        *,
        key,
    ):
        keys = jr.split(key, depth + 4)

        self.input_size   = input_size
        self.patch_size   = patch_size
        self.in_channels  = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads    = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, key=keys[0],
        )
        self.t_embedder = TimestepEmbedder(hidden_size, key=keys[1])
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, key=keys[2],
        )

        # fixed sin-cos positional embedding (computed in numpy)
        pos = _get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.pos_embed = jnp.array(pos)                       # (N, D)

        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio, key=keys[3 + i])
            for i in range(depth)
        ]
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, key=keys[3 + depth],
        )

    # ---- forward ---------------------------------------------------------

    def unpatchify(self, x):
        """(num_patches, P*P*outC) → (outC, H, W)"""
        c = self.out_channels
        p = self.patch_size
        h = w = self.input_size // p
        x = x.reshape(h, w, p, p, c)
        # (h, w, p, q, c) → (c, h, p, w, q)  then flatten spatial
        x = jnp.einsum("hwpqc->chpwq", x)
        return x.reshape(c, h * p, w * p)

    def __call__(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed              # (N, D)
        t_emb = self.t_embedder(t)                            # (D,)
        y_emb = self.y_embedder(y)                            # (D,)
        c = t_emb + y_emb                                     # (D,)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)                            # (N, P²·outC)
        return self.unpatchify(x)                              # (outC, H, W)


# ═══════════════════════════════════════════════════════════════════════
#  Model configs  (match DiT_models dict in original models.py)
# ═══════════════════════════════════════════════════════════════════════

def DiT_XL_2(*, key, **kw):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, key=key, **kw)

def DiT_XL_4(*, key, **kw):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, key=key, **kw)

def DiT_L_2(*, key, **kw):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, key=key, **kw)

def DiT_B_2(*, key, **kw):
    return DiT(depth=12, hidden_size=768,  patch_size=2, num_heads=12, key=key, **kw)

def DiT_S_2(*, key, **kw):
    return DiT(depth=12, hidden_size=384,  patch_size=2, num_heads=6,  key=key, **kw)


# ═══════════════════════════════════════════════════════════════════════
#  Weight loading from PyTorch checkpoint
# ═══════════════════════════════════════════════════════════════════════

def load_dit_weights(model, checkpoint_path):
    """
    Load pretrained DiT weights from a PyTorch .pt checkpoint.

    Handles both raw state-dicts and {\"ema\": state_dict} checkpoints.
    Returns a new model with all weights replaced.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        ckpt = ckpt["ema"]

    def j(key):
        return jnp.array(ckpt[key].detach().float().numpy())

    # --- PatchEmbed ---
    model = eqx.tree_at(
        lambda m: m.x_embedder.proj.weight, model,
        j("x_embedder.proj.weight"),
    )
    model = eqx.tree_at(
        lambda m: m.x_embedder.proj.bias, model,
        j("x_embedder.proj.bias").reshape(-1, 1, 1),         # (C,) → (C,1,1)
    )

    # --- positional embedding  (stored as (1, N, D) in checkpoint) ---
    model = eqx.tree_at(
        lambda m: m.pos_embed, model,
        j("pos_embed").squeeze(0),                            # → (N, D)
    )

    # --- TimestepEmbedder ---
    model = eqx.tree_at(lambda m: m.t_embedder.mlp_0.weight, model,
                         j("t_embedder.mlp.0.weight"))
    model = eqx.tree_at(lambda m: m.t_embedder.mlp_0.bias,   model,
                         j("t_embedder.mlp.0.bias"))
    model = eqx.tree_at(lambda m: m.t_embedder.mlp_2.weight, model,
                         j("t_embedder.mlp.2.weight"))
    model = eqx.tree_at(lambda m: m.t_embedder.mlp_2.bias,   model,
                         j("t_embedder.mlp.2.bias"))

    # --- LabelEmbedder ---
    model = eqx.tree_at(lambda m: m.y_embedder.embedding_table.weight, model,
                         j("y_embedder.embedding_table.weight"))

    # --- Transformer blocks ---
    for i in range(len(model.blocks)):
        p = f"blocks.{i}"

        # attention
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].attn.qkv.weight,  model,
                             j(f"{p}.attn.qkv.weight"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].attn.qkv.bias,    model,
                             j(f"{p}.attn.qkv.bias"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].attn.proj.weight,  model,
                             j(f"{p}.attn.proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].attn.proj.bias,    model,
                             j(f"{p}.attn.proj.bias"))

        # MLP
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].mlp.fc1.weight,   model,
                             j(f"{p}.mlp.fc1.weight"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].mlp.fc1.bias,     model,
                             j(f"{p}.mlp.fc1.bias"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].mlp.fc2.weight,   model,
                             j(f"{p}.mlp.fc2.weight"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].mlp.fc2.bias,     model,
                             j(f"{p}.mlp.fc2.bias"))

        # adaLN modulation  (Sequential(SiLU, Linear) — Linear is index 1)
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].adaLN_modulation.weight, model,
                             j(f"{p}.adaLN_modulation.1.weight"))
        model = eqx.tree_at(lambda m, i=i: m.blocks[i].adaLN_modulation.bias,   model,
                             j(f"{p}.adaLN_modulation.1.bias"))

    # --- FinalLayer ---
    model = eqx.tree_at(lambda m: m.final_layer.linear.weight, model,
                         j("final_layer.linear.weight"))
    model = eqx.tree_at(lambda m: m.final_layer.linear.bias,   model,
                         j("final_layer.linear.bias"))
    model = eqx.tree_at(lambda m: m.final_layer.adaLN_modulation.weight, model,
                         j("final_layer.adaLN_modulation.1.weight"))
    model = eqx.tree_at(lambda m: m.final_layer.adaLN_modulation.bias,   model,
                         j("final_layer.adaLN_modulation.1.bias"))

    return model


# ═══════════════════════════════════════════════════════════════════════
#  Noise schedule  (port of DiT's SpacedDiffusion respacing)
# ═══════════════════════════════════════════════════════════════════════

def _space_timesteps(num_timesteps, desired_count):
    """
    Select `desired_count` timesteps from an original schedule of
    `num_timesteps`, using evenly-spaced fractional striding
    (matches respace.space_timesteps for a single section).
    """
    if desired_count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_timesteps - 1) / (desired_count - 1)
    steps = []
    cur = 0.0
    for _ in range(desired_count):
        steps.append(round(cur))
        cur += frac_stride
    return sorted(set(steps))


def dit_noise_schedule(T, diffusion_steps=1000):
    """
    Compute the respaced DDIM noise schedule matching DiT / ParaTAA.

    Returns
    -------
    alpha_bar    : jnp array (T+1,)  — cumulative alpha, index 0 = 1.0
    timestep_map : jnp int32 (T,)    — maps respaced index → original index
    """
    # original linear beta schedule  (get_named_beta_schedule("linear", 1000))
    scale = 1000 / diffusion_steps
    betas_orig = np.linspace(
        scale * 0.0001, scale * 0.02, diffusion_steps, dtype=np.float64,
    )
    alphas_orig    = 1.0 - betas_orig
    alpha_bar_orig = np.cumprod(alphas_orig)

    # respace to T steps
    use_timesteps = _space_timesteps(diffusion_steps, T)
    assert len(use_timesteps) == T

    timestep_map = np.array(use_timesteps, dtype=np.int32)

    # respaced betas  (SpacedDiffusion.__init__)
    last_alpha = 1.0
    new_betas  = []
    for idx in use_timesteps:
        new_betas.append(1.0 - alpha_bar_orig[idx] / last_alpha)
        last_alpha = alpha_bar_orig[idx]
    new_betas  = np.array(new_betas, dtype=np.float64)
    new_alphas = 1.0 - new_betas

    # alpha_bar for the respaced schedule  (length T+1, index 0 = 1.0)
    alpha_bar = np.concatenate([[1.0], np.cumprod(new_alphas)])

    return jnp.array(alpha_bar), jnp.array(timestep_map)


# ═══════════════════════════════════════════════════════════════════════
#  Classifier-free guidance wrapper
# ═══════════════════════════════════════════════════════════════════════

class CfgEpsModel(eqx.Module):
    """
    Classifier-free guidance wrapper around DiT.

    Implements the ``(state, t) → eps`` signature expected by DDIMDynamics.
    Stores the DiT model, class label, guidance scale, and timestep map as
    Equinox fields so the whole thing is a valid eqx.Module / JAX pytree.
    """
    model:        DiT
    timestep_map: jnp.ndarray           # (T,) int32
    y_cond:       jnp.ndarray           # scalar int32
    null_label:   jnp.ndarray           # scalar int32
    cfg_scale:    float = eqx.field(static=True)
    C: int = eqx.field(static=True)
    H: int = eqx.field(static=True)
    W: int = eqx.field(static=True)

    def __call__(self, state, t):
        """
        state : (D,) flat latent
        t     : scalar int — DDIM timestep index (1 … T)
        Returns (D,) eps prediction (direction for DDIM update)
        """
        x = state.reshape(self.C, self.H, self.W)
        # map respaced t (1-indexed) to original 1000-step index
        t_orig = self.timestep_map[t - 1]

        # conditional + unconditional forward
        out_cond   = self.model(x, t_orig, self.y_cond)       # (outC, H, W)
        out_uncond = self.model(x, t_orig, self.null_label)

        # CFG on first 3 channels only  (DiT convention for reproducibility)
        guided = out_uncond[:3] + self.cfg_scale * (out_cond[:3] - out_uncond[:3])
        # 4th channel: unguided conditional
        ch4 = out_cond[3:4]
        direction = jnp.concatenate([guided, ch4], axis=0)    # (4, H, W)

        return direction.reshape(-1)


def make_cfg_eps_model(model, class_label, cfg_scale, timestep_map,
                       latent_channels=4, latent_size=32):
    """
    Build a ``CfgEpsModel`` (eqx.Module) for DDIMDynamics.

    Parameters
    ----------
    model        : DiT equinox module (with pretrained weights loaded)
    class_label  : int — ImageNet class index (e.g. 207 = golden retriever)
    cfg_scale    : float — classifier-free guidance strength
    timestep_map : (T,) int array — respaced index → original timestep
    """
    return CfgEpsModel(
        model=model,
        timestep_map=timestep_map,
        y_cond=jnp.int32(class_label),
        null_label=jnp.int32(model.y_embedder.num_classes),
        cfg_scale=float(cfg_scale),
        C=latent_channels,
        H=latent_size,
        W=latent_size,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Factory: build DDIMDynamics + initial state + drivers for DiT
# ═══════════════════════════════════════════════════════════════════════

def make_dit_diffusion(
    checkpoint_path,
    T=100,
    class_label=207,
    cfg_scale=4.0,
    eta=0.0,
    seed=0,
    latent_size=32,
    use_fp16=False,
):
    """
    Build everything needed to run DEER / TAA on DiT-XL/2.

    Returns
    -------
    deer_fxn : callable(state, driver) → next_state
    dynamics : DDIMDynamics  (for taa_ddim_alg)
    x0       : (D,) initial noise
    drivers  : (T, D+1) per-step [noise_vector | timestep]
    params   : dict with extra info (timestep_map, etc.)
    """
    from dynamics.diffusion import build_ddim_coeffs, DDIMDynamics

    C, H, W = 4, latent_size, latent_size
    D = C * H * W

    # --- 1. model ---
    key = jr.PRNGKey(seed)
    k_model, k_noise = jr.split(key)
    model = DiT_XL_2(input_size=latent_size, num_classes=1000, key=k_model)
    model = load_dit_weights(model, checkpoint_path)
    if use_fp16:
        # cast all floating leaves to float16
        model = jax.tree.map(
            lambda x: x.astype(jnp.float16) if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
            model,
        )

    # --- 2. noise schedule ---
    alpha_bar, timestep_map = dit_noise_schedule(T)
    coef_x, coef_eps, coef_noise, noise_thr = build_ddim_coeffs(alpha_bar, eta=eta)

    # --- 3. eps model with CFG ---
    eps_fn = make_cfg_eps_model(
        model, class_label, cfg_scale, timestep_map,
        latent_channels=C, latent_size=latent_size,
    )

    # --- 4. DDIMDynamics ---
    dynamics = DDIMDynamics(
        D=D,
        T=T,
        eps_model=eps_fn,
        coef_x=coef_x,
        coef_eps=coef_eps,
        coef_noise=coef_noise,
        noise_thr=noise_thr,
    )

    # --- 5. deer_fxn  (single-step dynamics for DEER) ---
    coef_x_local     = coef_x
    coef_eps_local   = coef_eps
    coef_noise_local = coef_noise

    def deer_fxn(state, driver):
        z = driver[:D]
        t = driver[D].astype(jnp.int32)
        eps = eps_fn(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    # --- 6. initial state + drivers ---
    # noise vectors: (T+1, D)  — last one is the initial pure noise
    noise_keys = jr.split(k_noise, T + 1)
    noise_vecs = jnp.stack(
        [jr.normal(noise_keys[i], (D,)) for i in range(T + 1)]
    )
    x0 = noise_vecs[-1]                                       # pure noise

    # drivers[s] = [z_s | t_s] where t_s = T - s (DDIM timestep)
    t_per_s = (T - jnp.arange(T)).astype(jnp.float32)
    drivers = jnp.concatenate(
        [noise_vecs[:T], t_per_s[:, None]], axis=1,
    )                                                         # (T, D+1)

    params = {
        "timestep_map": timestep_map,
        "alpha_bar": alpha_bar,
        "latent_shape": (C, H, W),
    }

    return deer_fxn, dynamics, x0, drivers, params


# ═══════════════════════════════════════════════════════════════════════
#  VAE decoder  (PyTorch — only used for final visualization)
# ═══════════════════════════════════════════════════════════════════════

def load_vae(vae_path="stabilityai/sd-vae-ft-mse", device="cpu"):
    """
    Load the Stable Diffusion VAE decoder.

    Parameters
    ----------
    vae_path : str
        HuggingFace model id or local path.  The default
        ``stabilityai/sd-vae-ft-mse`` is the standard VAE used with DiT.
    device : str
        "cpu" or "cuda"

    Returns
    -------
    vae : AutoencoderKL (PyTorch module, eval mode, no grad)
    """
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    return vae


def decode_latent(vae, latent):
    """
    Decode a single latent to a uint8 RGB image.

    Parameters
    ----------
    vae    : AutoencoderKL returned by ``load_vae``
    latent : (4, H, W) JAX array, numpy array, or PyTorch tensor
             — the raw latent from DEER / TAA (in latent space)

    Returns
    -------
    img : (H*8, W*8, 3) uint8 numpy array  (suitable for PIL / matplotlib)
    """
    import torch
    import numpy as np

    # accept JAX arrays
    if hasattr(latent, "devices"):
        latent = np.array(latent)
    if isinstance(latent, np.ndarray):
        latent = torch.tensor(latent)

    with torch.no_grad():
        # scale factor 0.18215 matches DiT / Stable Diffusion convention
        decoded = vae.decode(latent.unsqueeze(0).float() / 0.18215).sample
        img = torch.clamp(127.5 * decoded.cpu() + 128.0, 0, 255)
        img = img.permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img[0]
