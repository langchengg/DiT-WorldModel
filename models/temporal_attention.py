"""
Multi-Scale Temporal Attention for World Models.

创新: 在 World Model 中引入多尺度时序建模.

直觉来源:
- 机器人动作有不同时间尺度的影响
- 抓取动作: 短期 (接触物体) + 长期 (物体位移)
- 导航动作: 短期 (转向) + 长期 (到达目标)
- 传统方法只用固定窗口的历史帧, 忽略了多尺度时序依赖

方法:
- 用不同 dilation rate 的因果注意力捕捉多尺度时序模式
- Scale 1 (rate=1): 逐帧细粒度变化
- Scale 2 (rate=2): 每隔一帧, 捕捉中期趋势
- Scale 3 (rate=4): 每隔三帧, 捕捉长期动态

这个模块可以插入 DiTWorldModel 的 patch embedding 之后,
为每个 spatial patch 注入时序信息.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalMultiHeadAttention(nn.Module):
    """
    因果多头注意力 (Causal Multi-Head Attention).
    
    使用因果 mask 确保 t 时刻只能看到 t 及之前的帧,
    防止信息泄漏 (future information leakage).
    
    Args:
        dim:       Hidden dimension.
        num_heads: Number of attention heads.
        dropout:   Dropout rate.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) temporal sequence.
            mask: (T, T) causal mask (True = masked positions).
        Returns:
            (B, T, D) attended output.
        """
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv.unbind(0)

        # Build causal mask if not provided
        if mask is None:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1,
            )

        # Scaled dot-product attention
        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=~mask.unsqueeze(0).unsqueeze(0) if mask is not None else None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")
        return self.proj_drop(self.proj(attn_out))


class TemporalMultiScaleAttention(nn.Module):
    """
    多尺度时序注意力模块.
    
    对每个时间尺度 (dilation rate = 1, 2, 4) 使用独立的因果注意力,
    然后将多尺度的最后一帧特征融合为单个输出向量.
    
    设计选择:
    - 为什么用 dilation 而不是 pooling?
      → Dilation 保留了原始帧的信息, 只是采样间隔不同
      → Pooling 会损失信息, 尤其是快速变化的场景
    
    - 为什么融合最后一帧而不是所有帧?
      → 世界模型只需要预测下一帧, 只需最新的融合特征
      → 减少计算开销
    
    Args:
        dim:        Hidden dimension.
        num_heads:  Attention heads per scale.
        num_scales: Number of temporal scales.
        scales:     List of dilation rates. Default: [1, 2, 4].
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_scales: int = 3,
        scales: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.scales = scales or [1, 2, 4]
        assert len(self.scales) == num_scales, \
            f"Number of scales {len(self.scales)} != num_scales {num_scales}"

        # Independent attention per scale
        self.temporal_attns = nn.ModuleList([
            CausalMultiHeadAttention(dim, num_heads, dropout)
            for _ in range(num_scales)
        ])

        # Layer norms per scale
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_scales)
        ])

        # Fusion: concatenate multi-scale features → linear → output
        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, D) — T timesteps of frame features.
        
        Returns:
            fused: (B, D) — multi-scale temporal feature for the latest frame.
        """
        B, T, D = frame_features.shape
        multi_scale_outputs = []

        # Normalize scale weights
        scale_w = F.softmax(self.scale_weights, dim=0)

        for i, (scale, attn, norm) in enumerate(
            zip(self.scales, self.temporal_attns, self.norms)
        ):
            # Sample frames at this dilation rate
            # For scale=2: take frames [0, 2, 4, ...] or [1, 3, 5, ...]
            # We always include the last frame
            indices = list(range(T - 1, -1, -scale))
            indices.reverse()
            if len(indices) < 2:
                # If too few frames, just use the latest
                indices = [T - 1]

            sampled = frame_features[:, indices, :]  # (B, T_sampled, D)

            # Causal attention
            attended = attn(norm(sampled))  # (B, T_sampled, D)

            # Residual connection
            attended = sampled + attended

            # Take the last timestep's feature (weighted)
            last_feat = attended[:, -1, :]  # (B, D)
            multi_scale_outputs.append(last_feat * scale_w[i])

        # Fuse multi-scale features
        fused = self.scale_fusion(torch.cat(multi_scale_outputs, dim=-1))  # (B, D)
        return fused


class TemporalMultiScaleBlock(nn.Module):
    """
    Complete temporal block that can be integrated into DiTWorldModel.
    
    包含:
    1. Frame encoder: 将每帧的 patch features 池化为帧级特征
    2. Multi-scale temporal attention: 捕捉多尺度时序动态
    3. Broadcast: 将时序特征广播回每个 patch
    
    用法: 在 DiTWorldModel 的 DiT blocks 之前插入,
    为 patch tokens 注入时序信息.
    
    Args:
        dim:         Hidden dimension.
        num_heads:   Attention heads.
        num_patches: Number of spatial patches per frame.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_patches: int = 256,
        num_scales: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches

        # Frame-level feature extraction (from patch tokens)
        self.frame_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        # Multi-scale temporal attention
        self.temporal_attn = TemporalMultiScaleAttention(
            dim, num_heads, num_scales, dropout=dropout,
        )

        # Project temporal feature and add to patch tokens
        self.temporal_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Gating: control how much temporal info to inject
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        history_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens:     (B, N, D) current frame patch tokens.
            history_features: (B, T-1, D) past frame-level features (optional).
                              If None, only uses current frame.
        Returns:
            (B, N, D) patch tokens augmented with temporal information.
        """
        B, N, D = patch_tokens.shape

        # Extract frame-level feature from current patches
        current_frame_feat = self.frame_pool(patch_tokens.mean(dim=1))  # (B, D)

        if history_features is not None:
            # Stack current with history: (B, T, D)
            all_frames = torch.cat([
                history_features,
                current_frame_feat.unsqueeze(1),
            ], dim=1)
        else:
            all_frames = current_frame_feat.unsqueeze(1)  # (B, 1, D)

        # Multi-scale temporal attention
        temporal_feat = self.temporal_attn(all_frames)  # (B, D)

        # Project and gate
        temporal_feat = self.temporal_proj(temporal_feat)  # (B, D)
        gate = self.gate(temporal_feat)                     # (B, D)

        # Broadcast to all patches and add with gating
        temporal_feat = gate * temporal_feat  # (B, D)
        patch_tokens = patch_tokens + temporal_feat.unsqueeze(1)  # (B, N, D)

        return patch_tokens
