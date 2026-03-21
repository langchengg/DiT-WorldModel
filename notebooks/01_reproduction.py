# ---
# DiT-WorldModel: 01 - DIAMOND 论文复现与 DiT 创新
# Kaggle Notebook for reproduction and ablation
# GPU: T4 x2 or P100 recommended
# ---

"""
# 🎮 DiT-WorldModel: DIAMOND 复现 + DiT 架构创新

## 实验目标
1. 理解 DIAMOND 的核心思想: 用 Diffusion Model 作为 World Model
2. 实现 DiT 替换 U-Net 的创新
3. 对比实验: DiT vs U-Net 在世界模型上的表现

## 环境要求
- Kaggle GPU: T4/P100
- Python 3.10+
- PyTorch 2.1+
"""

# ============================================================
# Cell 1: 环境安装
# ============================================================

# !pip install torch torchvision torchaudio
# !pip install gymnasium[atari] ale-py
# !pip install einops timm lpips imageio wandb tqdm pyyaml
# !pip install opencv-python pillow matplotlib

# If cloning from your GitHub:
# !git clone https://github.com/YOUR_USERNAME/DiT-WorldModel.git
# %cd DiT-WorldModel

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Cell 2: 导入项目模块
# ============================================================

# Add project to path (if running from notebook)
sys.path.insert(0, '.')

from models.dit_world_model import DiTWorldModel, dit_small_world_model
from models.diffusion import DiffusionProcess, DDIMSampler, CosineNoiseSchedule
from models.temporal_attention import TemporalMultiScaleAttention
from models.action_discretizer import ActionDiscretizer
from training.trainer import WorldModelTrainer, WorldModelDataset
from training.progressive_schedule import ProgressiveDiffusionScheduler
from training.augmentation import RoboticAugmentationPipeline
from evaluation.metrics import MetricsTracker, SSIMCalculator, PSNRCalculator
from evaluation.visualize import FrameVisualizer, plot_training_curves


# ============================================================
# Cell 3: 模型架构详解
# ============================================================

"""
## 🏗️ DiT World Model 架构

核心创新: 用 DiT (Diffusion Transformer) 替换 DIAMOND 的 U-Net backbone

**为什么 DiT 优于 U-Net (在 World Model 场景)?**

1. **全局注意力**: U-Net 的 conv 是局部操作, DiT 的 self-attention 是全局的
   → World Model 需要理解物体间的全局交互 (如 Breakout 中球和砖块的关系)

2. **统一架构**: DiT 的每层都是相同的 Transformer block, 更容易 scale up
   → U-Net 有 encoder-decoder-skip connections, 结构复杂

3. **adaLN-Zero**: 优雅的条件注入机制
   → 比 cross-attention 更高效, 比 concat-conditioning 更有效
"""

# 创建模型
model = dit_small_world_model(
    img_size=64,
    patch_size=4,       # 64/4 = 16x16 = 256 patches
    in_channels=6,      # 3 (noisy target) + 1 (history)
    action_dim=18,      # Atari action space
)

print(model)
print(f"\nTotal parameters: {model.get_num_params() / 1e6:.1f}M")

# Verify forward pass
x_noisy = torch.randn(2, 3, 64, 64)
t = torch.randint(0, 1000, (2,))
obs_history = torch.randn(2, 1, 64, 64)
action = torch.randint(0, 18, (2,))

noise_pred, reward_pred, done_pred = model(x_noisy, t, obs_history, action)
print(f"\nForward pass shapes:")
print(f"  noise_pred:  {noise_pred.shape}")
print(f"  reward_pred: {reward_pred.shape}")
print(f"  done_pred:   {done_pred.shape}")


# ============================================================
# Cell 4: Diffusion Process 详解
# ============================================================

"""
## 🌀 Diffusion Process

DIAMOND 的核心: 用扩散模型预测下一帧

Forward: x_0 (clean) → x_T (noise)  通过逐步加噪
Reverse: x_T (noise) → x_0 (clean)  通过模型去噪

这里使用 Cosine 噪声调度 (比 Linear 更平滑)
"""

diffusion = DiffusionProcess(
    num_timesteps=1000,
    schedule_type="cosine",
    prediction_type="epsilon",
)

# Visualize noise schedule
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

timesteps = np.arange(1000)
axes[0].plot(timesteps, diffusion.alphas_cumprod.numpy())
axes[0].set_title("ᾱ_t (Signal Retention)")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("ᾱ_t")
axes[0].grid(True, alpha=0.3)

axes[1].plot(timesteps, diffusion.sqrt_alphas_cumprod.numpy(), label="√ᾱ_t")
axes[1].plot(timesteps, diffusion.sqrt_one_minus_alphas_cumprod.numpy(), label="√(1-ᾱ_t)")
axes[1].set_title("Signal vs Noise Coefficients")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(timesteps, diffusion.betas.numpy())
axes[2].set_title("β_t (Noise Added Per Step)")
axes[2].set_xlabel("Timestep")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("noise_schedule.png", dpi=150)
plt.show()
print("✅ Noise schedule visualization saved")


# ============================================================
# Cell 5: Forward Diffusion 可视化
# ============================================================

"""
## 📷 Forward Diffusion 过程可视化

观察图像如何在不同 timestep 被逐步加噪
"""

# Create a synthetic "game frame"
x_0 = torch.rand(1, 3, 64, 64)

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
timesteps_to_show = [0, 100, 300, 500, 700, 999]

for ax, t_val in zip(axes, timesteps_to_show):
    t = torch.tensor([t_val])
    x_t, _ = diffusion.forward_process(x_0, t)
    
    img = x_t[0].permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(img)
    ax.set_title(f"t={t_val}")
    ax.axis("off")

plt.suptitle("Forward Diffusion: Clean → Noisy", fontsize=14)
plt.tight_layout()
plt.savefig("forward_diffusion.png", dpi=150)
plt.show()


# ============================================================
# Cell 6: 训练 Demo (合成数据)
# ============================================================

"""
## 🏋️ Training Demo

先用合成数据验证训练管道是否正确, 然后再用真实 Atari 数据
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create synthetic dataset
dataset = WorldModelDataset(obs_history_len=1, max_size=2000)
for i in range(2000):
    obs = torch.rand(3, 64, 64)
    next_obs = obs + 0.05 * torch.randn_like(obs)
    next_obs = next_obs.clamp(0, 1)
    action = torch.randint(0, 18, (1,)).squeeze()
    reward = float(torch.randn(1).item() > 0.5)
    done = i % 200 == 199
    dataset.add(obs, action, next_obs, reward, done)

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Build small model for demo
demo_model = DiTWorldModel(
    img_size=64, patch_size=4, in_channels=6,
    hidden_size=128, depth=4, num_heads=4,  # Tiny model for demo
    action_dim=18,
)

# Build trainer
trainer = WorldModelTrainer(
    model=demo_model,
    diffusion=diffusion,
    lr=3e-4,
    warmup_steps=100,
    total_steps=5000,
    use_amp=(device == "cuda"),
    device=device,
    output_dir="demo_output",
)

# Train 5 epochs
print("\n🏋️ Training demo (5 epochs)...")
history = trainer.fit(dataloader, num_epochs=5)

# Plot losses
if history["loss"]:
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], 'b-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Demo)")
    plt.grid(True, alpha=0.3)
    plt.savefig("demo_training_loss.png", dpi=150)
    plt.show()
    print("✅ Demo training complete!")


# ============================================================
# Cell 7: Progressive Diffusion Schedule 可视化
# ============================================================

"""
## 📈 Progressive Diffusion Training

创新: 训练初期用少量扩散步数, 逐步增加

直觉: 
- 模型还很差时, 用 100 步扩散 = 浪费算力
- 先学粗糙的世界动态 (10步), 再学精细细节 (100步)
"""

scheduler = ProgressiveDiffusionScheduler(
    max_steps=100,
    min_steps=10,
    warmup_epochs=20,
    total_epochs=100,
    schedule="cosine",
)

epochs = list(range(100))
train_steps = [scheduler.get_num_steps(e) for e in epochs]
sample_steps = [scheduler.get_sampling_steps(e) for e in epochs]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, train_steps, 'b-', linewidth=2, label="Training Steps")
ax.plot(epochs, sample_steps, 'r--', linewidth=2, label="Sampling Steps")
ax.axvline(x=20, color='gray', linestyle=':', label="Warmup End")
ax.set_xlabel("Epoch")
ax.set_ylabel("Diffusion Steps")
ax.set_title("Progressive Diffusion Schedule (Cosine)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("progressive_schedule.png", dpi=150)
plt.show()


# ============================================================
# Cell 8: Multi-Scale Temporal Attention
# ============================================================

"""
## ⏰ Multi-Scale Temporal Attention

创新: 用不同 dilation rate 的因果注意力
捕捉短/中/长期时序依赖

Scale 1 (rate=1): 逐帧 → 物体运动
Scale 2 (rate=2): 每隔一帧 → 运动趋势  
Scale 3 (rate=4): 每隔三帧 → 长期动态
"""

temporal_attn = TemporalMultiScaleAttention(
    dim=128, num_heads=4, num_scales=3,
)

# Simulate 8 time steps of frame features
frame_features = torch.randn(2, 8, 128)  # (B=2, T=8, D=128)
output = temporal_attn(frame_features)    # (B=2, D=128)

print(f"Input:  {frame_features.shape}")
print(f"Output: {output.shape}")
print(f"Scale weights: {torch.softmax(temporal_attn.scale_weights, dim=0).data}")


# ============================================================
# Cell 9: Evaluation Metrics
# ============================================================

"""
## 📊 Evaluation Metrics

评估世界模型质量的三个维度:
1. SSIM: 结构相似性 (越高越好, >0.8 好)
2. PSNR: 信噪比 (越高越好, >25dB 好)
3. LPIPS: 感知距离 (越低越好, <0.3 好)
4. FID: 分布距离 (越低越好, <50 好)
"""

ssim_calc = SSIMCalculator()
psnr_calc = PSNRCalculator()

# Simulate predictions
gt = torch.rand(4, 3, 64, 64)
pred_good = gt + 0.05 * torch.randn_like(gt)
pred_bad = gt + 0.3 * torch.randn_like(gt)

ssim_good = ssim_calc.compute(pred_good.clamp(0,1), gt).mean().item()
ssim_bad = ssim_calc.compute(pred_bad.clamp(0,1), gt).mean().item()
psnr_good = psnr_calc.compute(pred_good.clamp(0,1), gt).mean().item()
psnr_bad = psnr_calc.compute(pred_bad.clamp(0,1), gt).mean().item()

print(f"Good prediction: SSIM={ssim_good:.4f}, PSNR={psnr_good:.1f}dB")
print(f"Bad prediction:  SSIM={ssim_bad:.4f}, PSNR={psnr_bad:.1f}dB")


# ============================================================
# Cell 10: 完整训练 (Atari - 需要 GPU)
# ============================================================

"""
## 🎮 Full Atari Training

取消下面的注释在 Kaggle GPU 上运行完整训练.
预计时间: ~8-12 小时 (T4 GPU)
"""

# # Full training on Breakout
# !python main.py \
#     --config configs/dit_small.yaml \
#     --epochs 50 \
#     --batch_size 32 \
#     --output_dir outputs/breakout_dit_s

# # Or with progressive training disabled for baseline
# !python main.py \
#     --config configs/dit_small.yaml \
#     --epochs 50 \
#     --output_dir outputs/breakout_baseline

print("\n🎉 Notebook 01 complete!")
print("Next: 02_dit_ablation.py - DiT vs U-Net ablation experiments")
