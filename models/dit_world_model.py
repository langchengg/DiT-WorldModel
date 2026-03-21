"""
DiT-based World Model for Interactive Environment Prediction.

核心创新: 将 DIAMOND 的 U-Net backbone 替换为 Diffusion Transformer (DiT),
通过 adaLN-Zero 机制注入 action + timestep 条件信息。

Architecture:
    观测序列 [o_{t-k}, ..., o_t] + 动作 a_t
              ↓
       Patch Embedding + Positional Encoding
              ↓
       DiT Blocks (adaLN-Zero conditioning)
              ↓
       Unpatchify → 预测噪声 ε
              ↓
       辅助 heads: reward + done prediction

Reference:
    - DiT: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
    - DIAMOND: "Diffusion for World Modeling" (Alonso et al., NeurIPS 2024)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep encoding.

    将标量 timestep t 映射为 d-维向量, 使用正余弦函数:
        PE(t, 2i)   = sin(t / 10000^{2i/d})
        PE(t, 2i+1) = cos(t / 10000^{2i/d})

    Args:
        dim: Embedding dimension (must be even).
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) diffusion timesteps (integer or float).
        Returns:
            (B, dim) sinusoidal embeddings.
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)
        return emb


class Attention(nn.Module):
    """
    Multi-head self-attention with optional Flash Attention support.

    Args:
        dim:       Hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias:  Whether to use bias in QKV projection.
        attn_drop: Dropout rate on attention weights.
        proj_drop: Dropout rate on output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tokens.
        Returns:
            (B, N, D) attended output.
        """
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)

        # Use PyTorch 2.0 scaled_dot_product_attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(attn_out))


class Mlp(nn.Module):
    """
    MLP block with GELU activation (matching DiT / ViT convention).

    Args:
        in_features:     Input dimension.
        hidden_features: Hidden dimension (default 4 * in_features).
        out_features:    Output dimension (default = in_features).
        drop:            Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class DiTBlock(nn.Module):
    """
    DiT Transformer block with adaptive Layer Norm Zero (adaLN-Zero).

    关键设计:
    - adaLN-Zero 将 conditioning 向量 c (action + timestep) 变换为
      6 个调制参数: (shift, scale, gate) × (attn, mlp)
    - 零初始化确保训练初期 block 是恒等映射 → 稳定训练
    - 相比 cross-attention conditioning, adaLN-Zero 更高效且效果更好

    Args:
        hidden_size: Hidden dimension D.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP hidden dim = hidden_size * mlp_ratio.
        drop:        Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        # Layer norms WITHOUT learnable affine (affine comes from adaLN)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop, proj_drop=drop,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=drop,
        )

        # adaLN-Zero modulation: condition → 6 * D parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # 零初始化: 训练初期 DiT block = identity mapping
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch tokens.
            c: (B, D)    conditioning vector (action + timestep).
        Returns:
            (B, N, D) output tokens.
        """
        # Generate 6 modulation parameters from conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # --- Attention branch with modulation ---
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)

        # --- MLP branch with modulation ---
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using 2D convolution.

    将 (B, C, H, W) 图像划分为 (H/P × W/P) 个 patch,
    每个 patch 通过线性投影映射到 D 维.

    Args:
        img_size:    Input image size (assumed square).
        patch_size:  Patch size P.
        in_channels: Number of input channels.
        embed_dim:   Output embedding dimension D.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images.
        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class FinalLayer(nn.Module):
    """
    Final output layer: adaLN modulation + linear unpatchify.

    将 (B, N, D) patch tokens 还原为 (B, C_out, H, W) 像素空间.

    Args:
        hidden_size:  Hidden dimension D.
        patch_size:   Patch size P.
        out_channels: Output channels (typically 3 for RGB).
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True,
        )
        # adaLN modulation for final layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, h: int, w: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch tokens.
            c: (B, D) conditioning.
            h: Number of patches along height.
            w: Number of patches along width.
        Returns:
            (B, C_out, H, W) unpatchified output.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)  # (B, N, P*P*C_out)

        # Unpatchify: rearrange patches back to spatial grid
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=self.out_channels,
        )
        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class DiTWorldModel(nn.Module):
    """
    Diffusion Transformer World Model.

    用 DiT 架构替换 DIAMOND 的 U-Net，用于 next-frame prediction.
    输入: 加噪目标帧 x_t + 历史观测 obs_history + action + timestep
    输出: 预测噪声 ε + reward logits + done logits

    创新点:
    1. adaLN-Zero conditioning: 统一的 action + timestep 条件注入
    2. 辅助 reward/done 预测 heads: 复用世界模型特征
    3. 零初始化策略: 确保训练稳定性

    Model Variants:
        DiT-S: hidden=384,  depth=12, heads=6   (~22M params)
        DiT-B: hidden=768,  depth=12, heads=12  (~86M params)
        DiT-L: hidden=1024, depth=24, heads=16  (~304M params)

    Args:
        img_size:             Input image size (square).
        patch_size:           Patch size P.
        in_channels:          Input channels = obs_history_channels + noisy_target_channels.
        hidden_size:          Transformer hidden dimension D.
        depth:                Number of DiT blocks.
        num_heads:            Number of attention heads.
        mlp_ratio:            MLP hidden dim ratio.
        action_dim:           Number of discrete actions (Atari: 18).
        num_diffusion_steps:  Maximum diffusion timestep T.
        out_channels:         Output channels (3 for RGB noise prediction).
        drop_rate:            Dropout rate.
        num_reward_classes:   Number of reward classes (default 3: {-1, 0, +1}).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        action_dim: int = 18,
        num_diffusion_steps: int = 1000,
        out_channels: int = 3,
        drop_rate: float = 0.0,
        num_reward_classes: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads

        # Grid dimensions
        self.h_patches = img_size // patch_size
        self.w_patches = img_size // patch_size
        self.num_patches = self.h_patches * self.w_patches

        # ----- Input Embedding -----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
        )

        # ----- Conditioning -----
        # Action embedding (discrete actions)
        self.action_embed = nn.Embedding(action_dim, hidden_size)

        # Timestep embedding (sinusoidal → MLP)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Fuse action + timestep → conditioning vector
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ----- DiT Backbone -----
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

        # ----- Output -----
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        # ----- Auxiliary Prediction Heads -----
        # Reward prediction: {-1, 0, +1} → 3-class classification
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size // 2, num_reward_classes),
        )

        # Done prediction: binary classification
        self.done_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size // 2, 2),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Careful weight initialization following DiT paper:
        - Position embedding: small normal
        - Linear layers: Xavier uniform
        - adaLN output layers: zero init (handled in DiTBlock/FinalLayer)
        """
        # Position embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Apply to all modules
        self.apply(self._init_weight_fn)

    @staticmethod
    def _init_weight_fn(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        obs_history: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict noise, reward, and done.

        Args:
            x_noisy:     (B, out_channels, H, W) — noisy target frame.
            t:           (B,) — diffusion timestep.
            obs_history: (B, C_hist, H, W) — historical observation frames.
                         C_hist = in_channels - out_channels.
            action:      (B,) — discrete action index.

        Returns:
            noise_pred:  (B, out_channels, H, W) — predicted noise ε.
            reward_pred: (B, num_reward_classes)  — reward logits.
            done_pred:   (B, 2)                   — done logits.
        """
        # --- Concatenate noisy target with history ---
        # x_noisy: (B, 3, H, W), obs_history: (B, 1, H, W) → (B, 4, H, W)
        x = torch.cat([x_noisy, obs_history], dim=1)

        # --- Patchify + positional embedding ---
        x = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # --- Build conditioning vector ---
        t_emb = self.time_embed(t)            # (B, D)
        a_emb = self.action_embed(action)     # (B, D)
        c = self.cond_proj(torch.cat([t_emb, a_emb], dim=-1))  # (B, D)

        # --- DiT blocks ---
        for block in self.blocks:
            x = block(x, c)

        # --- Predict noise (unpatchify) ---
        noise_pred = self.final_layer(x, c, self.h_patches, self.w_patches)

        # --- Auxiliary heads (from mean-pooled features) ---
        feat = x.mean(dim=1)  # (B, D) — global average pooling over patches
        reward_pred = self.reward_head(feat)
        done_pred = self.done_head(feat)

        return noise_pred, reward_pred, done_pred

    def forward_with_continuous_action(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        obs_history: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with pre-computed continuous action embedding.
        Used for robotic tasks where actions are continuous.

        Args:
            x_noisy:          (B, out_channels, H, W) noisy target frame.
            t:                (B,) diffusion timestep.
            obs_history:      (B, C_hist, H, W) historical observations.
            action_embedding: (B, D) pre-computed action embedding.

        Returns:
            Same as forward().
        """
        x = torch.cat([x_noisy, obs_history], dim=1)
        x = self.patch_embed(x) + self.pos_embed

        t_emb = self.time_embed(t)
        c = self.cond_proj(torch.cat([t_emb, action_embedding], dim=-1))

        for block in self.blocks:
            x = block(x, c)

        noise_pred = self.final_layer(x, c, self.h_patches, self.w_patches)
        feat = x.mean(dim=1)
        reward_pred = self.reward_head(feat)
        done_pred = self.done_head(feat)

        return noise_pred, reward_pred, done_pred

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters (optionally excluding position/action embeddings)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embed.numel()
            n_params -= self.action_embed.weight.numel()
        return n_params

    def __repr__(self) -> str:
        n_params = self.get_num_params() / 1e6
        return (
            f"DiTWorldModel(\n"
            f"  img_size={self.img_size}, patch_size={self.patch_size},\n"
            f"  hidden_size={self.hidden_size}, depth={self.depth}, "
            f"heads={self.num_heads},\n"
            f"  params={n_params:.1f}M\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Predefined configurations
# ---------------------------------------------------------------------------

def dit_small_world_model(**kwargs) -> DiTWorldModel:
    """DiT-S: ~22M parameters."""
    return DiTWorldModel(hidden_size=384, depth=12, num_heads=6, **kwargs)


def dit_base_world_model(**kwargs) -> DiTWorldModel:
    """DiT-B: ~86M parameters."""
    return DiTWorldModel(hidden_size=768, depth=12, num_heads=12, **kwargs)


def dit_large_world_model(**kwargs) -> DiTWorldModel:
    """DiT-L: ~304M parameters."""
    return DiTWorldModel(hidden_size=1024, depth=24, num_heads=16, **kwargs)
