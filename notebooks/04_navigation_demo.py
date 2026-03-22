# ---
# DiT-WorldModel: 04 - Navigation Demo
# World Model + MPC for Visual Navigation
# ---

"""
# 🧭 World Model 视觉导航 Demo

## 核心思路

用训练好的 World Model 做 "想象" → "评估" → "执行" 的 MPC 导航:

1. 生成/加载导航数据 (Synthetic Grid / RECON / TartanDrive)
2. 训练 DiT World Model 学习 "画面怎么随动作变化"
3. MPC 导航: 想象多条未来轨迹 → 选最好的 → 执行
4. 可视化: 实际轨迹 vs 想象轨迹对比

## 运行方式

    # 使用合成数据 (无需下载, CPU 可跑):
    python notebooks/04_navigation_demo.py

    # 使用 RECON 数据集:
    python notebooks/04_navigation_demo.py --dataset recon --data_dir /path/to/recon

    # 使用 TartanDrive 数据集:
    python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path/to/tartan
"""

import sys
import argparse
sys.path.insert(0, ".")

import torch
import numpy as np
import random

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Navigation Demo")
parser.add_argument("--dataset", type=str, default="synthetic",
                    choices=["synthetic", "recon", "tartan"],
                    help="Dataset type")
parser.add_argument("--data_dir", type=str, default=None,
                    help="Data directory for recon/tartan datasets")
parser.add_argument("--epochs", type=int, default=5,
                    help="Training epochs")
parser.add_argument("--device", type=str, default="auto",
                    help="Device: cuda, cpu, mps, or auto")
parser.add_argument("--skip_training", action="store_true",
                    help="Skip training, use random weights (for testing pipeline)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to pretrained checkpoint")
args, _ = parser.parse_known_args()

# Device
if args.device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = args.device

print(f"🖥️  Device: {device}")

# Seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ============================================================
# Cell 1: Create Navigation Dataset
# ============================================================

"""
## 📦 Step 1: 准备导航数据

三种数据源:
- synthetic: GridNavigationEnv 自动生成 (推荐起步)
- recon:     RECON 室外导航真实数据
- tartan:    TartanDrive 越野导航真实数据
"""

from torch.utils.data import DataLoader
from navigation.dataset import create_navigation_dataset

dataset = create_navigation_dataset(
    dataset_type=args.dataset,
    data_dir=args.data_dir,
    img_size=64,
    obs_history_len=1,
    num_episodes=50,
    episode_length=100,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

print(f"\n📊 Dataset: {args.dataset}")
print(f"   Samples: {len(dataset)}")

# Visualize a few samples
sample = dataset[0]
print(f"   obs_history shape: {sample['obs_history'].shape}")
print(f"   obs_next shape:    {sample['obs_next'].shape}")
print(f"   action:            {sample['action'].item()}")


# ============================================================
# Cell 2: Build World Model
# ============================================================

"""
## 🏗️ Step 2: 构建 DiT World Model

使用 DiT-Small 架构, 4 个离散动作 (导航).
输入: 历史帧 + 加噪目标帧 + 动作 + 时间步
输出: 预测噪声 + reward + done
"""

from models.dit_world_model import DiTWorldModel
from models.diffusion import DiffusionProcess, DDIMSampler

# Build model
model = DiTWorldModel(
    img_size=64,
    patch_size=4,
    in_channels=6,       # 3 (history) + 3 (noisy target)
    hidden_size=384,
    depth=12,
    num_heads=6,
    action_dim=4,        # FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT
    num_diffusion_steps=1000,
    out_channels=3,
    drop_rate=0.1,
)
print(f"\n🏗️  Model: {model}")
print(f"   Parameters: {model.get_num_params() / 1e6:.1f}M")

# Build diffusion
diffusion = DiffusionProcess(
    num_timesteps=1000,
    schedule_type="cosine",
    prediction_type="epsilon",
)

# Build DDIM sampler (fast inference)
sampler = DDIMSampler(
    diffusion=diffusion,
    num_steps=20,
    eta=0.0,
)


# ============================================================
# Cell 3: Train World Model
# ============================================================

"""
## 🏋️ Step 3: 训练 World Model

让模型学习 "给定当前画面+动作 → 预测下一帧画面".
"""

from training.trainer import WorldModelTrainer
from training.progressive_schedule import ProgressiveDiffusionScheduler

if not args.skip_training:
    prog_sched = ProgressiveDiffusionScheduler(
        max_steps=100, min_steps=10,
        warmup_epochs=2, total_epochs=args.epochs,
    )

    trainer = WorldModelTrainer(
        model=model,
        diffusion=diffusion,
        lr=3e-4,
        warmup_steps=500,
        total_steps=50000,
        use_amp=(device == "cuda"),
        device=device,
        output_dir="outputs/navigation",
        progressive_scheduler=prog_sched,
        save_every=5,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(args.checkpoint)
        print(f"📂 Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0

    print(f"\n🏋️ Training World Model for {args.epochs} epochs ...")
    history = trainer.fit(dataloader, num_epochs=args.epochs)

    # Plot training curves
    try:
        from evaluation.visualize import plot_training_curves
        plot_training_curves(history, "outputs/navigation/training_curves.png")
    except Exception as e:
        print(f"  ⚠️ Could not plot training curves: {e}")
else:
    print("\n⏭️  Skipping training (using random weights)")
    model.to(device)


# ============================================================
# Cell 4: World Model Imagination Demo
# ============================================================

"""
## 💭 Step 4: 测试 World Model 想象能力

给定当前画面 + 动作序列, 让模型 "想象" 未来画面.
"""

from models.diffusion import WorldModelEnv
from navigation.sim_env import GridNavigationEnv, ACTION_NAMES

print("\n💭 Testing World Model imagination ...")

model.eval()
model.to(device)

# Create a test environment
test_env = GridNavigationEnv(grid_size=8, img_size=64, seed=123)
test_obs = test_env.reset()

# Create WorldModelEnv (imagination environment)
wm_env = WorldModelEnv(
    model=model,
    diffusion=diffusion,
    sampler=sampler,
    horizon=8,
    num_history=1,
)

# Imagine 4 steps with random actions
imagined_obs = wm_env.reset(test_obs.unsqueeze(0).to(device))
imagined_frames = [test_obs]
imagined_actions = []

print("  Imagining 4 future frames:")
for step in range(4):
    action = torch.randint(0, 4, (1,)).to(device)
    next_obs, reward, done, info = wm_env.step(action)
    imagined_frames.append(next_obs.squeeze(0).cpu())
    imagined_actions.append(action.item())
    print(f"    Step {step+1}: action={ACTION_NAMES[action.item()]}")

# Get Ground Truth for those 4 actions
gt_obs = test_env.reset(seed=123)
gt_frames = [gt_obs]
for action_idx in imagined_actions:
    gt_obs, gt_r, gt_done, gt_info = test_env.step(action_idx)
    gt_frames.append(gt_obs)

# Save exactly 4 images comparison (Top: Truth, Bottom: Predicted)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    # Process ground truth numpy
    gt_img = gt_frames[i+1].permute(1, 2, 0).numpy()
    gt_img = np.clip(gt_img, 0, 1)
    axes[0, i].imshow(gt_img)
    axes[0, i].set_title(f"Truth (t={i+1})")
    axes[0, i].axis("off")
    
    # Process predicted numpy
    pred_img = imagined_frames[i+1].permute(1, 2, 0).numpy()
    pred_img = np.clip(pred_img, 0, 1)
    axes[1, i].imshow(pred_img)
    axes[1, i].set_title(f"Predicted (t={i+1})")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("outputs/navigation/4_frames_comparison.png", dpi=150)
plt.close()
print("  ✅ Saved 4-frame comparison to: outputs/navigation/4_frames_comparison.png")


# ============================================================
# Cell 5: MPC Navigation
# ============================================================

"""
## 🧭 Step 5: MPC 导航

用 World Model + MPC 让 agent 导航到目标:
1. 采样 N 条候选动作序列
2. 想象每条序列的未来画面
3. 评估: 碰撞扣分, 接近目标加分
4. 执行最优轨迹的第一个动作
5. 重复
"""

from navigation.navigator import WorldModelNavigator, navigate

print("\n🧭 Running MPC navigation ...")

nav_env = GridNavigationEnv(grid_size=8, img_size=64, max_steps=50, seed=456)

# Run navigation
nav_result = navigate(
    env=nav_env,
    world_model=model,
    diffusion=diffusion,
    sampler=sampler,
    max_steps=50,
    planning_horizon=5,    # Imagine 5 steps ahead
    num_candidates=16,     # 16 candidate trajectories (keep small for CPU)
    device=device,
    verbose=True,
)

print(f"\n📊 Navigation Result:")
print(f"   Success:      {nav_result['success']}")
print(f"   Total steps:  {nav_result['total_steps']}")
print(f"   Total reward:  {sum(nav_result['rewards']):.2f}")

# Visualize navigation episode
from navigation.visualize_nav import visualize_navigation, create_navigation_video

visualize_navigation(
    trajectory=nav_result["trajectory"],
    actions=nav_result["actions"],
    rewards=nav_result["rewards"],
    imaginations=nav_result["imaginations"],
    title="MPC Navigation Episode",
    output_path="outputs/navigation/navigation_episode.png",
    show_every=5,
)

create_navigation_video(
    trajectory=nav_result["trajectory"],
    output_path="outputs/navigation/navigation.mp4",
    fps=3,
    imaginations=nav_result["imaginations"],
)


# ============================================================
# Cell 6: Batch Evaluation
# ============================================================

"""
## 📊 Step 6: 批量评估

在多个随机推导环境上测试导航性能.
"""

print("\n📊 Batch evaluation (5 episodes) ...")

all_results = []
for ep_seed in range(5):
    eval_env = GridNavigationEnv(
        grid_size=8, img_size=64, max_steps=30, seed=1000 + ep_seed,
    )
    result = navigate(
        env=eval_env,
        world_model=model,
        diffusion=diffusion,
        sampler=sampler,
        max_steps=30,
        planning_horizon=3,
        num_candidates=8,
        device=device,
        verbose=False,
    )
    all_results.append(result)
    status = "✅" if result["success"] else "❌"
    print(f"  Episode {ep_seed}: {status} | "
          f"steps={result['total_steps']} | "
          f"reward={sum(result['rewards']):.1f}")

# Summary
from navigation.visualize_nav import plot_navigation_metrics

successes = sum(1 for r in all_results if r["success"])
print(f"\n📊 Summary:")
print(f"   Success rate: {successes}/{len(all_results)} "
      f"({successes / len(all_results) * 100:.1f}%)")
print(f"   Avg steps:    {np.mean([r['total_steps'] for r in all_results]):.1f}")
print(f"   Avg reward:   {np.mean([sum(r['rewards']) for r in all_results]):.1f}")

plot_navigation_metrics(all_results, "outputs/navigation/navigation_metrics.png")


# ============================================================
# Cell 7: 项目路线图
# ============================================================

"""
## 🗺️ 下一步

### 提升预测质量
1. 训练更多 epoch (50-100 epochs)
2. 用更大模型 (DiT-B, 86M params)
3. 用 progressive training (逐步增加扩散步数)

### 接入真实数据
1. 一键下载 RECON sample → python notebooks/04_navigation_demo.py --dataset recon
   (如果是完整 50G 数据, 下载后: --dataset recon --data_dir /path)
2. 下载 TartanDrive → python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path

### 提升导航性能
1. 增加 planning_horizon (8-15)
2. 增加 num_candidates (64-256), 需要 GPU
3. 使用 CEM 替代 Random Shooting
4. 训练 CNNCollisionDetector 替代 done-signal 碰撞检测
5. 加入 visual feature extractor 做目标相似度评估

### 速度优化
1. DDIM 步数减少 (20 → 5-10)
2. 批量并行想象 (batch all candidates)
3. 使用 DiT-Tiny (更小的模型)
4. 低分辨率预测 (32×32)
"""

print("\n🎉 Navigation demo complete!")
print("   Outputs saved to: outputs/navigation/")
print("   Files:")
print("     - training_curves.png")
print("     - imagination_demo.png")
print("     - ground_truth_comparison.png")
print("     - navigation_episode.png")
print("     - navigation.mp4")
print("     - navigation_metrics.png")
