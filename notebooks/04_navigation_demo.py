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

    # 默认使用 RECON 真实数据路径 (若未提供 data_dir, 自动下载完整 RECON):
    python notebooks/04_navigation_demo.py

    # 显式使用 RECON 数据集:
    python notebooks/04_navigation_demo.py --dataset recon --data_dir /path/to/recon

    # 仅在快速调试时下载 sample:
    python notebooks/04_navigation_demo.py --dataset recon --download_mode sample

    # 使用 TartanDrive 数据集:
    python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path/to/tartan

    # 仅做本地合成调试:
    python notebooks/04_navigation_demo.py --dataset synthetic
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, ".")

import torch
import numpy as np
import random

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Navigation Demo")
parser.add_argument("--dataset", type=str, default="recon",
                    choices=["synthetic", "recon", "tartan"],
                    help="Dataset type")
parser.add_argument("--data_dir", type=str, default=None,
                    help="Data directory for recon/tartan datasets")
parser.add_argument("--epochs", type=int, default=5,
                    help="Training epochs")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Maximum training batch size")
parser.add_argument("--eval_ratio", type=float, default=0.1,
                    help="Fraction of full trajectories reserved for eval")
parser.add_argument("--download_mode", type=str, default="full",
                    choices=["full", "sample"],
                    help="RECON download mode when data_dir is omitted")
parser.add_argument("--sample_num_files", type=int, default=3,
                    help="Number of RECON files to extract when using sample mode")
parser.add_argument("--max_trajectories", type=int, default=None,
                    help="Optional cap on RECON trajectories loaded after download")
parser.add_argument("--max_runs", type=int, default=None,
                    help="Optional cap on TartanDrive runs loaded")
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

OUTPUT_DIR = Path("outputs/navigation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATASETS = {"recon", "tartan"}
ACTION_BINS = 256


def resolve_action_dim(dataset_name: str) -> int:
    return 4 if dataset_name == "synthetic" else ACTION_BINS


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
from navigation.dataset import (
    create_navigation_dataset,
    get_num_trajectories,
    split_navigation_dataset_by_trajectory,
)

dataset_kwargs = {
    "dataset_type": args.dataset,
    "data_dir": args.data_dir,
    "img_size": 64,
    "obs_history_len": 1,
    "action_bins": ACTION_BINS,
    "download_mode": args.download_mode,
    "sample_num_files": args.sample_num_files,
    "max_trajectories": args.max_trajectories,
    "max_runs": args.max_runs,
}
if args.dataset == "synthetic":
    dataset_kwargs.update({
        "num_episodes": 50,
        "episode_length": 100,
    })

dataset = create_navigation_dataset(
    **dataset_kwargs,
)

if len(dataset) == 0:
    raise RuntimeError(
        f"Loaded dataset '{args.dataset}' but found 0 valid transitions. "
        "Check that the downloaded files match the expected schema and are not corrupted."
    )

num_trajectories = get_num_trajectories(dataset)
train_dataset, eval_dataset, split_info = split_navigation_dataset_by_trajectory(
    dataset,
    eval_ratio=args.eval_ratio,
    min_eval_trajectories=1,
    seed=42,
)

if len(train_dataset) == 0:
    raise RuntimeError(
        f"Trajectory split for '{args.dataset}' produced 0 training samples. "
        "Increase the dataset size or reduce eval_ratio."
    )

if len(eval_dataset) == 0:
    raise RuntimeError(
        f"Trajectory split for '{args.dataset}' produced 0 eval samples. "
        "At least two trajectories are required for a real held-out evaluation."
    )

train_batch_size = min(args.batch_size, max(1, len(train_dataset)))
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

steps_per_epoch = max(1, len(train_loader))
total_train_steps = max(1, steps_per_epoch * args.epochs)
warmup_steps = min(total_train_steps, max(steps_per_epoch, total_train_steps // 10))

print(f"\n📊 Dataset: {args.dataset}")
print(f"   Total samples: {len(dataset)}")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Eval samples:  {len(eval_dataset)}")
print(f"   Trajectories:  {num_trajectories}")
print(f"   Train trajs:   {split_info['train_trajectories']}")
print(f"   Eval trajs:    {split_info['eval_trajectories']}")
print(f"   Steps/epoch:   {steps_per_epoch}")
print(f"   Warmup steps:  {warmup_steps}")
print(f"   Total steps:   {total_train_steps}")

# Visualize a sample
sample = eval_dataset[0]
print(f"   obs_history shape: {sample['obs_history'].shape}")
print(f"   obs_next shape:    {sample['obs_next'].shape}")
print(f"   action:            {sample['action'].item()}")

action_dim = resolve_action_dim(args.dataset)


# ============================================================
# Cell 2: Build World Model
# ============================================================

"""
## 🏗️ Step 2: 构建 DiT World Model

使用 DiT-Small 架构.
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
    action_dim=action_dim,
    num_diffusion_steps=1000,
    out_channels=3,
    drop_rate=0.1,
)
print(f"\n🏗️  Model: {model}")
print(f"   Parameters: {model.get_num_params() / 1e6:.1f}M")
print(f"   Action dim: {action_dim}")

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
        warmup_steps=warmup_steps,
        total_steps=total_train_steps,
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
    history = trainer.fit(train_loader, num_epochs=args.epochs)

    # Plot training curves
    try:
        from evaluation.visualize import plot_training_curves
        plot_training_curves(history, str(OUTPUT_DIR / "training_curves.png"))
    except Exception as e:
        print(f"  ⚠️ Could not plot training curves: {e}")
else:
    print("\n⏭️  Skipping training (using random weights)")
    model.to(device)


def get_eval_sample(idx: int):
    return eval_dataset[idx]


@torch.no_grad()
def predict_next_frame(sample_dict):
    obs_history = sample_dict["obs_history"]
    if obs_history.dim() == 4:
        obs_ctx = obs_history[-1]
    else:
        obs_ctx = obs_history

    obs_ctx_b = obs_ctx.unsqueeze(0).to(device)
    gt_next_b = sample_dict["obs_next"].unsqueeze(0).to(device)
    action_b = sample_dict["action"].view(1).to(device)

    pred_next_b, reward_logits, done_logits = sampler.sample(
        model=model,
        shape=gt_next_b.shape,
        obs_history=obs_ctx_b,
        action=action_b,
        device=device,
    )
    pred_next_b = pred_next_b.clamp(0, 1)
    copy_last_b = obs_ctx_b.clamp(0, 1)

    return {
        "obs_ctx": obs_ctx_b.squeeze(0).cpu(),
        "gt_next": gt_next_b.squeeze(0).cpu(),
        "copy_last": copy_last_b.squeeze(0).cpu(),
        "pred_next": pred_next_b.squeeze(0).cpu(),
        "action": int(sample_dict["action"].item()),
        "reward_logits": reward_logits.squeeze(0).cpu(),
        "done_logits": done_logits.squeeze(0).cpu(),
    }


# ============================================================
# Cell 4: World Model Imagination Demo
# ============================================================

"""
## 💭 Step 4: 测试 World Model 想象能力

给定当前画面 + 动作序列, 让模型 "想象" 未来画面.
"""

from models.diffusion import WorldModelEnv

print("\n💭 Testing World Model imagination ...")

model.eval()
model.to(device)

import matplotlib.pyplot as plt

if args.dataset == "synthetic":
    from navigation.sim_env import GridNavigationEnv, ACTION_NAMES

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

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        gt_img = gt_frames[i + 1].permute(1, 2, 0).numpy()
        gt_img = np.clip(gt_img, 0, 1)
        axes[0, i].imshow(gt_img)
        axes[0, i].set_title(f"Truth (t={i + 1})")
        axes[0, i].axis("off")

        pred_img = imagined_frames[i + 1].permute(1, 2, 0).numpy()
        pred_img = np.clip(pred_img, 0, 1)
        axes[1, i].imshow(pred_img)
        axes[1, i].set_title(f"Predicted (t={i + 1})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_frames_comparison.png", dpi=150)
    plt.close()
    print(f"  ✅ Saved 4-frame comparison to: {OUTPUT_DIR / '4_frames_comparison.png'}")
else:
    num_vis = min(4, len(eval_dataset))
    predictions = [predict_next_frame(get_eval_sample(i)) for i in range(num_vis)]

    print(f"  Evaluating {num_vis} held-out real samples ...")

    fig, axes = plt.subplots(3, num_vis, figsize=(3 * num_vis, 8))
    if num_vis == 1:
        axes = np.array(axes).reshape(3, 1)

    for i, pred in enumerate(predictions):
        gt_img = pred["gt_next"].permute(1, 2, 0).numpy()
        copy_img = pred["copy_last"].permute(1, 2, 0).numpy()
        pred_img = pred["pred_next"].permute(1, 2, 0).numpy()

        axes[0, i].imshow(np.clip(gt_img, 0, 1))
        axes[0, i].set_title(f"GT next #{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(copy_img, 0, 1))
        axes[1, i].set_title("Copy-last baseline")
        axes[1, i].axis("off")

        axes[2, i].imshow(np.clip(pred_img, 0, 1))
        axes[2, i].set_title(f"DiT pred | a={pred['action']}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_frames_comparison.png", dpi=150)
    plt.close()
    print(f"  ✅ Saved held-out real-data comparison to: {OUTPUT_DIR / '4_frames_comparison.png'}")


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

if args.dataset == "synthetic":
    from navigation.navigator import WorldModelNavigator, navigate
    from navigation.visualize_nav import visualize_navigation, create_navigation_video
    from navigation.sim_env import GridNavigationEnv

    print("\n🧭 Running MPC navigation ...")

    nav_env = GridNavigationEnv(grid_size=8, img_size=64, max_steps=50, seed=456)

    nav_result = navigate(
        env=nav_env,
        world_model=model,
        diffusion=diffusion,
        sampler=sampler,
        max_steps=50,
        planning_horizon=5,
        num_candidates=16,
        device=device,
        verbose=True,
    )

    print(f"\n📊 Navigation Result:")
    print(f"   Success:      {nav_result['success']}")
    print(f"   Total steps:  {nav_result['total_steps']}")
    print(f"   Total reward: {sum(nav_result['rewards']):.2f}")

    visualize_navigation(
        trajectory=nav_result["trajectory"],
        actions=nav_result["actions"],
        rewards=nav_result["rewards"],
        imaginations=nav_result["imaginations"],
        title="MPC Navigation Episode",
        output_path=str(OUTPUT_DIR / "navigation_episode.png"),
        show_every=5,
    )

    create_navigation_video(
        trajectory=nav_result["trajectory"],
        output_path=str(OUTPUT_DIR / "navigation.mp4"),
        fps=3,
        imaginations=nav_result["imaginations"],
    )
else:
    from evaluation.visualize import create_video

    print("\n🧭 Building held-out real-data qualitative panel ...")

    num_episode = min(8, len(eval_dataset))
    episode_preds = [predict_next_frame(get_eval_sample(i)) for i in range(num_episode)]

    fig, axes = plt.subplots(4, num_episode, figsize=(3 * num_episode, 10))
    if num_episode == 1:
        axes = np.array(axes).reshape(4, 1)

    video_frames = []
    for i, pred in enumerate(episode_preds):
        ctx_img = pred["obs_ctx"].permute(1, 2, 0).numpy()
        gt_img = pred["gt_next"].permute(1, 2, 0).numpy()
        copy_img = pred["copy_last"].permute(1, 2, 0).numpy()
        pred_img = pred["pred_next"].permute(1, 2, 0).numpy()

        axes[0, i].imshow(np.clip(ctx_img, 0, 1))
        axes[0, i].set_title(f"Context #{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(gt_img, 0, 1))
        axes[1, i].set_title("GT next")
        axes[1, i].axis("off")

        axes[2, i].imshow(np.clip(copy_img, 0, 1))
        axes[2, i].set_title("Copy-last baseline")
        axes[2, i].axis("off")

        axes[3, i].imshow(np.clip(pred_img, 0, 1))
        axes[3, i].set_title(f"DiT pred | a={pred['action']}")
        axes[3, i].axis("off")

        video_frames.append(pred["gt_next"])
        video_frames.append(pred["copy_last"])
        video_frames.append(pred["pred_next"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "navigation_episode.png", dpi=150)
    plt.close()

    create_video(
        frames=video_frames,
        output_path=str(OUTPUT_DIR / "navigation.mp4"),
        fps=2,
    )


# ============================================================
# Cell 6: Batch Evaluation
# ============================================================

"""
## 📊 Step 6: 批量评估

在多个随机推导环境上测试导航性能.
"""

if args.dataset == "synthetic":
    from navigation.visualize_nav import plot_navigation_metrics
    from navigation.sim_env import GridNavigationEnv
    from navigation.navigator import navigate

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

    successes = sum(1 for r in all_results if r["success"])
    print(f"\n📊 Summary:")
    print(f"   Success rate: {successes}/{len(all_results)} "
          f"({successes / len(all_results) * 100:.1f}%)")
    print(f"   Avg steps:    {np.mean([r['total_steps'] for r in all_results]):.1f}")
    print(f"   Avg reward:   {np.mean([sum(r['rewards']) for r in all_results]):.1f}")

    plot_navigation_metrics(all_results, str(OUTPUT_DIR / "navigation_metrics.png"))
else:
    from evaluation.metrics import MetricsTracker

    print("\n📊 Held-out real-data evaluation ...")

    model_tracker = MetricsTracker(device=device)
    baseline_tracker = MetricsTracker(device=device)
    model_per_sample_metrics = []
    baseline_per_sample_metrics = []
    num_eval_samples = min(32, len(eval_dataset))

    for idx in range(num_eval_samples):
        pred = predict_next_frame(get_eval_sample(idx))
        model_metrics = model_tracker.evaluate_batch(
            pred["pred_next"].unsqueeze(0),
            pred["gt_next"].unsqueeze(0),
        )
        baseline_metrics = baseline_tracker.evaluate_batch(
            pred["copy_last"].unsqueeze(0),
            pred["gt_next"].unsqueeze(0),
        )
        model_per_sample_metrics.append(model_metrics)
        baseline_per_sample_metrics.append(baseline_metrics)

    model_ssim_vals = [m["ssim"] for m in model_per_sample_metrics]
    model_psnr_vals = [m["psnr"] for m in model_per_sample_metrics]
    model_lpips_vals = [m["lpips"] for m in model_per_sample_metrics]

    baseline_ssim_vals = [m["ssim"] for m in baseline_per_sample_metrics]
    baseline_psnr_vals = [m["psnr"] for m in baseline_per_sample_metrics]
    baseline_lpips_vals = [m["lpips"] for m in baseline_per_sample_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(baseline_ssim_vals, "o-", color="tab:gray", markersize=3, label="Copy-last")
    axes[0].plot(model_ssim_vals, "o-", color="tab:green", markersize=3, label="DiT")
    axes[0].set_title(
        f"SSIM | Copy={np.mean(baseline_ssim_vals):.3f} | DiT={np.mean(model_ssim_vals):.3f}"
    )
    axes[0].set_xlabel("Held-out sample")
    axes[0].set_ylabel("SSIM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(baseline_psnr_vals, "o-", color="tab:gray", markersize=3, label="Copy-last")
    axes[1].plot(model_psnr_vals, "o-", color="tab:blue", markersize=3, label="DiT")
    axes[1].set_title(
        f"PSNR | Copy={np.mean(baseline_psnr_vals):.2f} dB | DiT={np.mean(model_psnr_vals):.2f} dB"
    )
    axes[1].set_xlabel("Held-out sample")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(baseline_lpips_vals, "o-", color="tab:gray", markersize=3, label="Copy-last")
    axes[2].plot(model_lpips_vals, "o-", color="tab:orange", markersize=3, label="DiT")
    axes[2].set_title(
        f"LPIPS | Copy={np.mean(baseline_lpips_vals):.3f} | DiT={np.mean(model_lpips_vals):.3f}"
    )
    axes[2].set_xlabel("Held-out sample")
    axes[2].set_ylabel("LPIPS")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "navigation_metrics.png", dpi=150)
    plt.close()

    model_summary = model_tracker.get_summary()
    baseline_summary = baseline_tracker.get_summary()
    print("\n📊 Summary:")
    if "ssim" in baseline_summary and "ssim" in model_summary:
        print(
            f"   SSIM mean:  Copy={baseline_summary['ssim']['mean']:.4f} | "
            f"DiT={model_summary['ssim']['mean']:.4f}"
        )
    if "psnr" in baseline_summary and "psnr" in model_summary:
        print(
            f"   PSNR mean:  Copy={baseline_summary['psnr']['mean']:.4f} | "
            f"DiT={model_summary['psnr']['mean']:.4f}"
        )
    if "lpips" in baseline_summary and "lpips" in model_summary:
        print(
            f"   LPIPS mean: Copy={baseline_summary['lpips']['mean']:.4f} | "
            f"DiT={model_summary['lpips']['mean']:.4f}"
        )


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
1. 默认自动下载完整 RECON → python notebooks/04_navigation_demo.py --dataset recon
2. 快速 smoke test 才使用 sample → python notebooks/04_navigation_demo.py --dataset recon --download_mode sample
3. 下载 TartanDrive → python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path

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
print("     - 4_frames_comparison.png")
print("     - navigation_episode.png")
print("     - navigation.mp4")
print("     - navigation_metrics.png")
