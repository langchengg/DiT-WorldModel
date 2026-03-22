#!/usr/bin/env python3
"""
DiT-WorldModel: Main Training Entry Point.

用法:
    # 训练 DiT-S 在 Atari Breakout
    python main.py --config configs/dit_small.yaml

    # 训练 DiT-B (更大模型)
    python main.py --config configs/dit_base.yaml

    # 训练机器人场景
    python main.py --config configs/robotic_env.yaml

    # 自定义参数
    python main.py --config configs/dit_small.yaml --batch_size 16 --lr 1e-4
"""

import argparse
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.dit_world_model import DiTWorldModel, dit_small_world_model, dit_base_world_model
from models.diffusion import DiffusionProcess, DDIMSampler
from models.temporal_attention import TemporalMultiScaleAttention
from models.action_discretizer import ActionDiscretizer
from training.trainer import WorldModelTrainer, WorldModelDataset
from training.progressive_schedule import (
    ProgressiveDiffusionScheduler,
    ProgressiveResolutionScheduler,
    CombinedProgressiveScheduler,
)
from training.augmentation import RoboticAugmentationPipeline
from evaluation.metrics import MetricsTracker
from evaluation.visualize import FrameVisualizer, plot_training_curves


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> DiTWorldModel:
    """Build DiT World Model from config."""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "dit_small")

    common_kwargs = {
        "img_size": model_cfg.get("img_size", 64),
        "patch_size": model_cfg.get("patch_size", 4),
        "in_channels": model_cfg.get("in_channels", 6),
        "action_dim": model_cfg.get("action_dim", 18),
        "num_diffusion_steps": model_cfg.get("num_diffusion_steps", 1000),
        "out_channels": model_cfg.get("out_channels", 3),
        "drop_rate": model_cfg.get("drop_rate", 0.0),
        "num_reward_classes": model_cfg.get("num_reward_classes", 3),
    }

    if arch == "dit_small":
        model = dit_small_world_model(**common_kwargs)
    elif arch == "dit_base":
        model = dit_base_world_model(**common_kwargs)
    else:
        model = DiTWorldModel(
            hidden_size=model_cfg.get("hidden_size", 384),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 6),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            **common_kwargs,
        )

    print(f"\n🏗️  Model: {model}")
    return model


def build_diffusion(config: dict) -> DiffusionProcess:
    """Build diffusion process from config."""
    diff_cfg = config.get("diffusion", {})
    return DiffusionProcess(
        num_timesteps=diff_cfg.get("num_timesteps", 1000),
        schedule_type=diff_cfg.get("schedule_type", "cosine"),
        prediction_type=diff_cfg.get("prediction_type", "epsilon"),
        loss_type=diff_cfg.get("loss_type", "mse"),
    )


def build_progressive_scheduler(config: dict):
    """Build progressive training scheduler."""
    train_cfg = config.get("training", {})
    prog_cfg = train_cfg.get("progressive", {})

    if not prog_cfg.get("enabled", False):
        return None

    diff_scheduler = ProgressiveDiffusionScheduler(
        max_steps=prog_cfg.get("max_steps", 100),
        min_steps=prog_cfg.get("min_steps", 10),
        warmup_epochs=prog_cfg.get("warmup_epochs", 10),
        total_epochs=train_cfg.get("num_epochs", 100),
        schedule=prog_cfg.get("schedule", "linear"),
    )

    res_cfg = train_cfg.get("resolution", {})
    if res_cfg.get("enabled", False):
        stages = res_cfg.get("stages", [])
        grow_epochs = [s["epoch"] for s in stages]
        sizes = [s["size"] for s in stages]
        res_scheduler = ProgressiveResolutionScheduler(
            target_size=config["model"].get("img_size", 64),
            grow_epochs=grow_epochs,
            sizes=sizes,
        )
    else:
        res_scheduler = None

    combined = CombinedProgressiveScheduler(diff_scheduler, res_scheduler)
    combined.print_schedule(train_cfg.get("num_epochs", 100))
    return combined


def collect_atari_data(game: str, num_steps: int = 10000, img_size: int = 64):
    """
    Collect training data from Atari environment.
    Uses random policy for initial data collection.
    """
    import gymnasium as gym
    import cv2

    env = gym.make(game, render_mode="rgb_array")
    dataset = WorldModelDataset(obs_history_len=1, max_size=num_steps)

    obs, _ = env.reset()
    frame = cv2.resize(env.render(), (img_size, img_size))
    frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    print(f"\n📦 Collecting {num_steps} frames from {game}...")

    for step in range(num_steps):
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_frame = cv2.resize(env.render(), (img_size, img_size))
        next_frame_t = torch.from_numpy(next_frame).permute(2, 0, 1).float() / 255.0

        dataset.add(
            frame_t,
            torch.tensor(action, dtype=torch.long),
            next_frame_t,
            reward,
            done,
        )

        frame_t = next_frame_t

        if done:
            obs, _ = env.reset()
            frame = cv2.resize(env.render(), (img_size, img_size))
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        if (step + 1) % 1000 == 0:
            print(f"  Collected {step + 1}/{num_steps} frames")

    env.close()
    print(f"  Done! Dataset size: {len(dataset)}")
    return dataset


def collect_metaworld_data(env_name: str, num_steps: int = 10000, img_size: int = 64):
    """
    Collect training data from MetaWorld environment.
    Uses random policy for initial data collection.
    """
    import metaworld
    import cv2
    import random

    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name](render_mode="rgb_array")
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    dataset = WorldModelDataset(obs_history_len=4, max_size=num_steps)
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
        
        cont_action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        disc_action = action_disc.encode(cont_action).squeeze(0)
        flat_action = disc_action[0]

        dataset.add(
            frame_t,
            flat_action,
            next_frame_t,
            reward,
            done,
        )

        frame_t = next_frame_t

        if done:
            obs, info = env.reset()
            frame = cv2.resize(env.render(), (img_size, img_size))
            frame_t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0

        if (step + 1) % 1000 == 0:
            print(f"  Collected {step + 1}/{num_steps} frames")

    env.close()
    print(f"  Done! Dataset size: {len(dataset)}")
    return dataset


def collect_navigation_data(config: dict, img_size: int = 64):
    """
    Collect training data from navigation environment.
    Supports synthetic grid, RECON, and TartanDrive datasets.
    """
    from navigation.dataset import create_navigation_dataset

    env_cfg = config.get("environment", {})
    ds_cfg = config.get("dataset", {})

    dataset_type = ds_cfg.get("type", "synthetic")
    data_dir = ds_cfg.get("data_dir", None)

    dataset = create_navigation_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        img_size=img_size,
        obs_history_len=1,
        num_episodes=ds_cfg.get("num_episodes", 50),
        episode_length=ds_cfg.get("episode_length", 100),
    )

    print(f"  Navigation dataset ({dataset_type}): {len(dataset)} transitions")
    return dataset


def demo_training(config: dict):
    """
    Demo training with synthetic data (no environment required).
    Useful for testing the pipeline.
    """
    print("\n🎮 Demo mode: using synthetic data")

    model_cfg = config["model"]
    img_size = model_cfg.get("img_size", 64)
    action_dim = model_cfg.get("action_dim", 18)

    dataset = WorldModelDataset(obs_history_len=1, max_size=5000)

    for i in range(5000):
        obs = torch.rand(3, img_size, img_size)
        next_obs = obs + 0.1 * torch.randn_like(obs)
        next_obs = next_obs.clamp(0, 1)
        action = torch.randint(0, action_dim, (1,)).squeeze()
        reward = float(torch.randn(1).item())
        done = random.random() < 0.01

        dataset.add(obs, action, next_obs, reward, done)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="DiT World Model Training")
    parser.add_argument("--config", type=str, default="configs/dit_small.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, mps, or auto")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    train_cfg = config.get("training", {})
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.epochs:
        train_cfg["num_epochs"] = args.epochs

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

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Build components
    model = build_model(config)
    diffusion = build_diffusion(config)
    progressive = build_progressive_scheduler(config)

    # Collect or generate data
    if args.demo:
        import random
        random.seed(42)
        dataset = demo_training(config)
    else:
        env_cfg = config.get("environment", {})
        env_type = env_cfg.get("type", "atari")
        img_size = config["model"].get("img_size", 64)
        
        if env_type == "navigation":
            dataset = collect_navigation_data(config, img_size=img_size)
        elif env_type == "metaworld":
            tasks = env_cfg.get("tasks", ["reach-v3"])
            game = tasks[0] if isinstance(tasks, list) and len(tasks) > 0 else "reach-v3"
            dataset = collect_metaworld_data(game, num_steps=10000, img_size=img_size)
        else:
            game = env_cfg.get("game", "BreakoutNoFrameskip-v4")
            dataset = collect_atari_data(game, num_steps=10000, img_size=img_size)

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Build trainer
    trainer = WorldModelTrainer(
        model=model,
        diffusion=diffusion,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 5000)),
        total_steps=int(train_cfg.get("total_steps", 200000)),
        use_amp=train_cfg.get("use_amp", True) and device == "cuda",
        device=device,
        save_every=train_cfg.get("save_every", 5),
        output_dir=args.output_dir,
        progressive_scheduler=progressive.diffusion if progressive else None,
        use_wandb=config.get("wandb", {}).get("enabled", False),
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"📂 Resumed from epoch {start_epoch}")

    # Train
    num_epochs = train_cfg.get("num_epochs", 100)
    history = trainer.fit(
        dataloader,
        num_epochs=num_epochs - start_epoch,
    )

    # Plot training curves
    plot_training_curves(history, f"{args.output_dir}/training_curves.png")

    print("\n✅ Training complete!")
    print(f"   Checkpoints saved to: {args.output_dir}/checkpoints/")
    print(f"   Training curves: {args.output_dir}/training_curves.png")


if __name__ == "__main__":
    main()
