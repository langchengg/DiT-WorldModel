"""
World Model Trainer — 完整的训练循环.

设计要点:
- Mixed Precision Training (AMP): 减少显存使用 + 加速训练
- 梯度裁剪 + 梯度 norm 监控: 防止梯度爆炸
- 学习率 warmup + cosine decay: 对 DiT 的 adaLN-Zero 初始化至关重要
- 对接 ProgressiveDiffusionScheduler: 训练初期用少量扩散步数
- 支持 WandB / TensorBoard 日志
- Checkpoint save/load + 定期评估

训练流程:
    1. 从真实环境收集 dataset: {obs_t, action_t, obs_{t+1}, reward, done}
    2. 采样 batch → 前向加噪 → 模型预测噪声 → 计算 loss
    3. 反向传播 → 梯度裁剪 → 参数更新
    4. 定期在 imagination 中评估 (FID, LPIPS, RL reward)
"""

import os
import time
import math
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    学习率: linear warmup → cosine decay.
    
    为什么用这个策略:
    - Warmup: DiT 的 adaLN-Zero 初始化使所有 block 近似于恒等映射,
      训练初期梯度信号弱, 需要较大 lr 来启动学习.
      但一开始就用大 lr 会导致 attention 权重不稳定.
      → 折中: 从 0 线性增到 max_lr.
    - Cosine decay: 后期慢慢降低 lr, 精细调整权重,
      比 step decay 更平滑, 效果通常更好.
    
    Args:
        optimizer:      PyTorch optimizer.
        warmup_steps:   Number of linear warmup steps.
        total_steps:    Total training steps.
        min_lr:         Minimum learning rate at the end.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return current lr."""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps,
            )
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


# ---------------------------------------------------------------------------
# Gradient Monitor
# ---------------------------------------------------------------------------

class GradientMonitor:
    """
    实时监控每层的梯度统计.
    
    监控指标:
    - grad_norm: 各层梯度范数 (检测梯度消失/爆炸)
    - update_ratio: 参数更新量 / 参数量 (理想范围 ~1e-3)
    - 如果某层 grad_norm 突然变大 → 梯度爆炸 → 需要更强的 clipping
    - 如果某层 grad_norm 接近 0 → 梯度消失 → 需要检查初始化
    """

    def __init__(self, model: nn.Module, log_every: int = 100):
        self.model = model
        self.log_every = log_every
        self.step_count = 0
        self.param_snapshots = {}

    def snapshot_params(self) -> None:
        """Save current parameter values for update_ratio computation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_snapshots[name] = param.data.clone()

    def compute_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute gradient statistics for all layers.
        
        Returns:
            dict: {layer_name: {grad_norm, param_norm, update_ratio}}
        """
        stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()

                update_ratio = 0.0
                if name in self.param_snapshots:
                    update = (param.data - self.param_snapshots[name]).norm(2).item()
                    update_ratio = update / max(param_norm, 1e-8)

                stats[name] = {
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "update_ratio": update_ratio,
                }
        return stats

    def check_health(self) -> List[str]:
        """
        检查梯度健康状态, 返回警告列表.
        """
        warnings = []
        stats = self.compute_stats()

        for name, s in stats.items():
            if s["grad_norm"] > 100:
                warnings.append(
                    f"⚠️ Gradient explosion in {name}: grad_norm={s['grad_norm']:.2f}"
                )
            if s["grad_norm"] < 1e-8 and "bias" not in name:
                warnings.append(
                    f"⚠️ Gradient vanishing in {name}: grad_norm={s['grad_norm']:.2e}"
                )
            if s["update_ratio"] > 0.1:
                warnings.append(
                    f"⚠️ Large update ratio in {name}: {s['update_ratio']:.4f}"
                )
        return warnings


# ---------------------------------------------------------------------------
# Replay Buffer / Dataset
# ---------------------------------------------------------------------------

class WorldModelDataset(Dataset):
    """
    World Model 训练数据集.
    
    存储 (obs_t, action_t, obs_{t+1}, reward_t, done_t) 五元组.
    支持从文件加载或在线收集.
    
    Args:
        obs_history_len: Number of history frames for conditioning.
        max_size:        Maximum buffer size.
    """

    def __init__(self, obs_history_len: int = 1, max_size: int = 100000):
        self.obs_history_len = obs_history_len
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.pointer = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """Add a transition to the dataset."""
        if len(self.observations) < self.max_size:
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            idx = self.pointer % self.max_size
            self.observations[idx] = obs
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.dones[idx] = done
        self.pointer += 1

    def add_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a batch of transitions."""
        for i in range(len(observations) - 1):
            self.add(
                observations[i], actions[i],
                observations[i + 1], rewards[i].item(), dones[i].item(),
            )

    def __len__(self) -> int:
        return max(0, min(len(self.observations) - 1, self.max_size))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys: obs_history, obs_next, action, reward, done.
        """
        # Build history context
        start = max(0, idx - self.obs_history_len + 1)
        history_frames = self.observations[start: idx + 1]

        # Pad if not enough history
        while len(history_frames) < self.obs_history_len:
            history_frames.insert(0, history_frames[0])

        obs_history = torch.stack(history_frames, dim=0)  # (T, C, H, W)
        obs_next = self.observations[idx + 1]              # (C, H, W)
        action = self.actions[idx]
        reward = self._discretize_reward(self.rewards[idx])
        done = int(self.dones[idx])

        return {
            "obs_history": obs_history,
            "obs_next": obs_next,
            "action": action,
            "reward": torch.tensor(reward, dtype=torch.long),
            "done": torch.tensor(done, dtype=torch.long),
        }

    @staticmethod
    def _discretize_reward(reward: float) -> int:
        """Map continuous reward to {0: negative, 1: zero, 2: positive}."""
        if reward < -0.01:
            return 0
        elif reward > 0.01:
            return 2
        return 1


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """
    完整的 World Model 训练器.
    
    核心特性:
    - Mixed Precision Training (AMP): 半精度前向/反向, 减少显存 ~40%
    - 梯度裁剪: max_grad_norm=1.0, 对 DiT 至关重要
    - 学习率 warmup + cosine decay
    - 梯度健康监控: 实时检测梯度消失/爆炸
    - 对接渐进式扩散训练
    - WandB 日志 + Checkpoint 管理
    
    Args:
        model:          DiTWorldModel instance.
        diffusion:      DiffusionProcess instance.
        optimizer:      PyTorch optimizer (default: AdamW).
        lr:             Learning rate.
        weight_decay:   Weight decay for AdamW.
        max_grad_norm:  Gradient clipping threshold.
        warmup_steps:   LR warmup steps.
        total_steps:    Total training steps.
        use_amp:        Whether to use mixed precision.
        device:         Training device.
        log_every:      Log metrics every N steps.
        save_every:     Save checkpoint every N epochs.
        output_dir:     Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 5000,
        total_steps: int = 200000,
        use_amp: bool = True,
        device: str = "cuda",
        log_every: int = 100,
        save_every: int = 5,
        output_dir: str = "outputs",
        progressive_scheduler=None,
        use_wandb: bool = False,
        wandb_project: str = "dit-worldmodel",
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device != "cpu"
        self.log_every = log_every
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progressive_scheduler = progressive_scheduler
        self.use_wandb = use_wandb

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8,
        )

        # LR Scheduler
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps,
        )

        # AMP
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient monitor
        self.grad_monitor = GradientMonitor(model)

        # Metrics tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.loss_history = []

        # WandB
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={
                    "lr": lr, "weight_decay": weight_decay,
                    "max_grad_norm": max_grad_norm,
                    "warmup_steps": warmup_steps,
                    "model_params": model.get_num_params() if hasattr(model, "get_num_params") else 0,
                })
            except ImportError:
                print("WandB not installed, disabling logging")
                self.use_wandb = False

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            dict of average losses: {total, diffusion, reward, done}.
        """
        self.model.train()
        self.epoch = epoch

        # Get number of diffusion steps for this epoch (progressive training)
        if self.progressive_scheduler is not None:
            num_diff_steps = self.progressive_scheduler.get_num_steps(epoch)
        else:
            num_diff_steps = self.diffusion.num_timesteps

        epoch_losses = {"total": 0, "diffusion": 0, "reward": 0, "done": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            losses = self.train_step(batch, max_timestep=num_diff_steps)
            
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "diff_steps": num_diff_steps,
            })

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        self.loss_history.append(epoch_losses["total"])

        # Log to WandB
        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    f"train/{k}": v for k, v in epoch_losses.items()
                }, step=self.global_step)
                wandb.log({
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                    "train/diff_steps": num_diff_steps,
                }, step=self.global_step)
            except Exception:
                pass

        # Save checkpoint
        if (epoch + 1) % self.save_every == 0:
            self.save_checkpoint(epoch)

        # Check if best
        if epoch_losses["total"] < self.best_loss:
            self.best_loss = epoch_losses["total"]
            self.save_checkpoint(epoch, is_best=True)

        return epoch_losses

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        max_timestep: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: dict with obs_history, obs_next, action, reward, done.
            max_timestep: Max diffusion timestep (for progressive training).
        
        Returns:
            dict of loss values.
        """
        # Move to device
        obs_history = batch["obs_history"].to(self.device)  # (B, T, C, H, W)
        obs_next = batch["obs_next"].to(self.device)         # (B, C, H, W)
        action = batch["action"].to(self.device)             # (B,)
        reward = batch["reward"].to(self.device)             # (B,)
        done = batch["done"].to(self.device)                 # (B,)

        B = obs_next.shape[0]

        # Use last history frame as context
        if obs_history.dim() == 5:
            obs_ctx = obs_history[:, -1]  # (B, C, H, W) — last frame
        else:
            obs_ctx = obs_history

        # Sample random timestep
        max_t = max_timestep or self.diffusion.num_timesteps
        t = torch.randint(0, max_t, (B,), device=self.device)

        # Forward diffusion: add noise to target
        x_t, noise = self.diffusion.forward_process(obs_next, t)

        # Snapshot for gradient monitoring
        if self.global_step % self.log_every == 0:
            self.grad_monitor.snapshot_params()

        # Forward pass with AMP
        with autocast(enabled=self.use_amp):
            # Model predicts noise (or v)
            noise_pred, reward_pred, done_pred = self.model(
                x_t, t, obs_ctx, action,
            )

            # Compute target based on prediction type
            if self.diffusion.prediction_type == "v_prediction":
                target = self.diffusion.get_v_target(obs_next, noise, t)
            else:
                target = noise

            # Compute losses
            losses = self.diffusion.compute_loss(
                noise_pred, target,
                reward_pred, reward,
                done_pred, done,
                reward_weight=1.0,
                done_weight=1.0,
            )

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(losses["total"]).backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm,
            )
        else:
            total_norm = 0.0

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler
        current_lr = self.lr_scheduler.step()

        # Gradient monitoring
        if self.global_step % self.log_every == 0:
            warnings = self.grad_monitor.check_health()
            for w in warnings:
                print(w)

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "grad/total_norm": total_norm if isinstance(total_norm, float) else total_norm.item(),
                        "grad/lr": current_lr,
                    }, step=self.global_step)
                except Exception:
                    pass

        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_loss": self.best_loss,
            "loss_history": self.loss_history,
        }

        path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)

        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            torch.save(ckpt, best_path)

        return str(path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.use_amp and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.loss_history = ckpt.get("loss_history", [])
        return ckpt.get("epoch", 0) + 1

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        eval_fn=None,
        eval_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training data.
            num_epochs:   Total number of epochs.
            eval_fn:      Optional evaluation function f(model, epoch) -> dict.
            eval_every:   Evaluate every N epochs.
        
        Returns:
            Training history: {loss: [...], eval_metric: [...]}.
        """
        history = {"loss": [], "eval": []}

        print(f"\n{'='*60}")
        print(f"  DiT World Model Training")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.get_num_params()/1e6:.1f}M" 
              if hasattr(self.model, 'get_num_params') else "")
        print(f"  Epochs: {num_epochs}")
        print(f"  AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            t0 = time.time()

            # Train one epoch
            epoch_losses = self.train_epoch(train_loader, epoch)
            dt = time.time() - t0

            history["loss"].append(epoch_losses["total"])

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {epoch_losses['total']:.4f} | "
                f"Diff: {epoch_losses['diffusion']:.4f} | "
                f"Rew: {epoch_losses.get('reward', 0):.4f} | "
                f"Done: {epoch_losses.get('done', 0):.4f} | "
                f"Time: {dt:.1f}s"
            )

            # Evaluation
            if eval_fn is not None and (epoch + 1) % eval_every == 0:
                self.model.eval()
                eval_metrics = eval_fn(self.model, epoch)
                history["eval"].append(eval_metrics)
                print(f"  Eval: {eval_metrics}")
                self.model.train()

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return history
