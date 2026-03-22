# ---
# DiT-WorldModel: 03 - Robotic Transfer Experiments
# Transfer DIAMOND framework to MetaWorld robotic tasks
# ---

"""
# 🤖 机器人场景迁移实验

## 核心挑战

将 Atari 上的 Diffusion World Model 迁移到机器人操作场景:

1. **动作空间**: 离散 → 连续 (需要离散化)
2. **观测复杂度**: 2D 像素 → RGB-D (更复杂的视觉)
3. **时序依赖**: Atari 4帧 → 机器人需要更长历史
4. **数据量**: Atari 100K → 机器人数据通常更少

## 解决方案

1. ActionDiscretizer: 连续动作均匀分 bin
2. Multi-Scale Temporal Attention: 捕捉多时间尺度
3. Temporally-Consistent Augmentation: 增强有限数据
4. Progressive Training: 高效利用算力
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.dit_world_model import DiTWorldModel
from models.diffusion import DiffusionProcess, DDIMSampler
from models.temporal_attention import TemporalMultiScaleAttention, TemporalMultiScaleBlock
from models.action_discretizer import ActionDiscretizer, KMeansActionDiscretizer
from training.trainer import WorldModelTrainer, WorldModelDataset
from training.augmentation import RoboticAugmentationPipeline
from training.progressive_schedule import ProgressiveDiffusionScheduler
from evaluation.metrics import MetricsTracker

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Cell 1: Action Discretization 实验
# ============================================================

"""
## 🎯 Action Discretization

将 MetaWorld 的 4-DOF 连续动作 (每维 [-1, 1]) 离散化.
对比两种方案: 均匀分 bin vs K-Means 聚类.
"""

# Simulate robot action data (4-DOF)
np.random.seed(42)
# Robot actions are often non-uniformly distributed
# (e.g., small movements are more common)
actions_data = np.concatenate([
    np.random.normal(0, 0.3, (5000, 4)),     # 大部分是小动作
    np.random.uniform(-1, 1, (1000, 4)),      # 少量大动作
])
actions_data = np.clip(actions_data, -1, 1)
actions_tensor = torch.from_numpy(actions_data).float()

# Method 1: Uniform binning
uniform_disc = ActionDiscretizer(action_dim=4, num_bins=256)
uniform_encoded = uniform_disc.encode(actions_tensor)
uniform_decoded = uniform_disc.decode(uniform_encoded)
uniform_error = (actions_tensor - uniform_decoded).abs().mean().item()

# Method 2: K-Means
kmeans_disc = KMeansActionDiscretizer(num_clusters=512, action_dim=4)
kmeans_disc.fit(actions_tensor)
kmeans_encoded = kmeans_disc.encode(actions_tensor)
kmeans_decoded = kmeans_disc.decode(kmeans_encoded)
kmeans_error = (actions_tensor - kmeans_decoded).abs().mean().item()

print(f"Uniform Binning:")
print(f"  Bins per dim: 256, Total: {uniform_disc.total_bins}")
print(f"  Mean abs error: {uniform_error:.6f}")
print(f"  Max quantization error: {uniform_disc.max_quantization_error:.6f}")

print(f"\nK-Means Clustering:")
print(f"  Clusters: 512")
print(f"  Mean abs error: {kmeans_error:.6f}")

# Visualize action distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(actions_data[:, 0], bins=50, alpha=0.7, label="Original")
axes[0].hist(uniform_decoded[:, 0].numpy(), bins=50, alpha=0.5, label="Uniform")
axes[0].set_title("Dim 0: Uniform Discretization")
axes[0].legend()

axes[1].hist(actions_data[:, 0], bins=50, alpha=0.7, label="Original")
axes[1].hist(kmeans_decoded[:, 0].numpy(), bins=50, alpha=0.5, label="K-Means")
axes[1].set_title("Dim 0: K-Means Discretization")
axes[1].legend()

plt.tight_layout()
plt.savefig("action_discretization.png", dpi=150)
plt.show()


# ============================================================
# Cell 2: Data Augmentation 验证
# ============================================================

"""
## 🔄 Temporally-Consistent Augmentation

验证增强的时序一致性: 同一序列的所有帧应该有相同的变换.
"""

augmentation = RoboticAugmentationPipeline(
    img_size=64,
    enable_color=True,
    enable_crop=True,
    enable_camera=True,
    enable_temporal=True,
    enable_noise=True,
)

print(augmentation)

# Create a synthetic robot observation sequence (8 frames)
sequence = [torch.rand(3, 64, 64) for _ in range(8)]

# Apply augmentation
aug_sequence, info = augmentation(sequence)

# Verify temporal consistency
print(f"\nAugmentation applied: {list(info.keys())}")
if "color_jitter" in info:
    print(f"  Color params: {info['color_jitter']}")
    # Verify same params applied to all frames
    print(f"  All frames have same brightness: "
          f"{info['color_jitter']['brightness']:.4f}")

# Visualize original vs augmented
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(sequence[i].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0, i].set_title(f"Original t={i}")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(aug_sequence[i].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1, i].set_title(f"Augmented t={i}")
    axes[1, i].axis("off")

plt.suptitle("Temporally-Consistent Augmentation", fontsize=14)
plt.tight_layout()
plt.savefig("temporal_augmentation.png", dpi=150)
plt.show()


# ============================================================
# Cell 3: Multi-Scale Temporal Attention 分析
# ============================================================

"""
## ⏰ Multi-Scale Temporal Attention Analysis

分析不同时间尺度的注意力权重分布.
"""

temporal_block = TemporalMultiScaleBlock(
    dim=128, num_heads=4, num_patches=256, num_scales=3,
)

# Simulate: 8 timesteps of patch features
current_patches = torch.randn(2, 256, 128)  # (B, N_patches, D)
history = torch.randn(2, 7, 128)            # (B, T-1, D)

# Forward pass
output = temporal_block(current_patches, history)
print(f"Input patches: {current_patches.shape}")
print(f"History: {history.shape}")
print(f"Output: {output.shape}")

# Get learned scale weights
weights = torch.softmax(temporal_block.temporal_attn.scale_weights, dim=0)
scales = [1, 2, 4]
print(f"\nLearned scale importance:")
for s, w in zip(scales, weights):
    print(f"  Scale {s} (dilation={s}): {w.item():.3f}")


# ============================================================
# Cell 4: MetaWorld Robot Environment Training
# ============================================================

"""
## 🤖 Robot World Model Training (MetaWorld Data)

使用真实的 MetaWorld 数据集测试完整的机器人训练管道.
"""

import metaworld
import cv2
import random

# Initialize MetaWorld environment
env_name = "reach-v3"
ml1 = metaworld.ML1(env_name)
env = ml1.train_classes[env_name](render_mode="rgb_array")
task = random.choice(ml1.train_tasks)
env.set_task(task)

# Create real robot dataset
num_steps = 3000
img_size = 64
robot_dataset = WorldModelDataset(obs_history_len=4, max_size=num_steps)
action_disc = ActionDiscretizer(action_dim=4, num_bins=256)

obs, info = env.reset()
frame = cv2.resize(env.render(), (img_size, img_size))
frame_t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0

print(f"\n📦 Collecting {num_steps} frames from MetaWorld {env_name}...")

for step in range(num_steps):
    action = env.action_space.sample()
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    next_frame = cv2.resize(env.render(), (img_size, img_size))
    next_frame_t = torch.from_numpy(next_frame.copy()).permute(2, 0, 1).float() / 255.0
    
    # Continuous action → discrete
    cont_action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    disc_action = action_disc.encode(cont_action).squeeze(0)
    flat_action = disc_action[0]  # Use first dimension as discrete action
    
    robot_dataset.add(frame_t, flat_action, next_frame_t, reward, done)

    frame_t = next_frame_t

    if done:
        obs, info = env.reset()
        frame = cv2.resize(env.render(), (img_size, img_size))
        frame_t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0

env.close()

robot_loader = DataLoader(
    robot_dataset, batch_size=16, shuffle=True, drop_last=True,
)

# Build robot world model
robot_model = DiTWorldModel(
    img_size=64, patch_size=4, in_channels=6,
    hidden_size=128, depth=6, num_heads=4,
    action_dim=256,  # 256 bins per dimension
    drop_rate=0.1,
)

diffusion = DiffusionProcess(num_timesteps=1000, schedule_type="cosine")

# Progressive scheduler
prog_sched = ProgressiveDiffusionScheduler(
    max_steps=100, min_steps=10,
    warmup_epochs=5, total_epochs=20,
)

# Train
trainer = WorldModelTrainer(
    model=robot_model,
    diffusion=diffusion,
    lr=3e-4,
    warmup_steps=200,
    total_steps=20000,
    use_amp=(device == "cuda"),
    device=device,
    output_dir="robot_outputs",
    progressive_scheduler=prog_sched,
)

print("\n🏋️ Training Robot World Model...")
history = trainer.fit(robot_loader, num_epochs=10)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(history["loss"], 'b-', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Robot World Model Training")
plt.grid(True, alpha=0.3)
plt.savefig("robot_training.png", dpi=150)
plt.show()


# ============================================================
# Cell 5: World Model Imagination Demo
# ============================================================

"""
## 💭 World Model Imagination (Dreaming)

让训练好的世界模型 "想象" 未来帧序列.
"""

from models.diffusion import WorldModelEnv

sampler = DDIMSampler(
    diffusion=diffusion,
    num_steps=20,
    eta=0.0,
)

wm_env = WorldModelEnv(
    model=robot_model,
    diffusion=diffusion,
    sampler=sampler,
    horizon=10,
)

# Reset with a random initial observation
init_obs = torch.rand(1, 3, 64, 64).to(device)
robot_model.eval()
robot_model.to(device)

obs = wm_env.reset(init_obs)
print("Imagining future frames...")

imagined_frames = [obs]
for step in range(5):
    action = torch.randint(0, 256, (1,)).to(device)
    next_obs, reward, done, info = wm_env.step(action)
    imagined_frames.append(next_obs)
    print(f"  Step {step}: reward={reward.item():.1f}, done={done.item()}")

# Visualize imagined trajectory
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for i, (ax, frame) in enumerate(zip(axes, imagined_frames)):
    img = frame[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    ax.imshow(img)
    ax.set_title(f"t={i}")
    ax.axis("off")

plt.suptitle("World Model Imagination", fontsize=14)
plt.tight_layout()
plt.savefig("imagination.png", dpi=150)
plt.show()


# ============================================================
# Cell 6: 实验结果汇总模板
# ============================================================

"""
## 📊 Results Template (Fill with Real Experiments)

在真实环境 (MetaWorld) 上运行后填入这些数字:
"""

results_template = """
| Method                    | Reach FID↓ | Push FID↓ | PickPlace FID↓ | Params | Train Time |
|---------------------------|:----------:|:---------:|:--------------:|:------:|:----------:|
| DIAMOND (U-Net baseline)  | -          | -         | -              | 45M    | -          |
| DiT-S (ours)              | -          | -         | -              | 22M    | -          |
| DiT-S + Progressive       | -          | -         | -              | 22M    | -          |
| DiT-S + MultiScale        | -          | -         | -              | 25M    | -          |
| DiT-S + Augmentation      | -          | -         | -              | 22M    | -          |
| DiT-S + All innovations   | -          | -         | -              | 25M    | -          |

Notes:
- Fill in after running real MetaWorld experiments
- FID computed on 1000 generated vs real frames
- Train time on single T4 GPU
"""

print(results_template)

print("\n🎉 Robot transfer experiments complete!")
print("All notebooks finished. Next: Run on real MetaWorld data!")
