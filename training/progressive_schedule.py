"""
Progressive Training Schedulers.

创新 4: 渐进式扩散训练策略.

核心直觉:
- 训练初期, World Model 质量很差, 用太多扩散步数浪费算力
- 随着训练进展, 逐步增加扩散步数以提升生成质量
- 类似 Progressive GAN 的思想, 但应用在扩散步数上

实验验证假设:
- H1: 渐进式训练 vs 固定步数, 收敛速度提升 20%+
- H2: 最终性能不低于固定步数训练
- 通过 FID/LPIPS 和下游 RL reward 双指标验证

两种渐进策略:
1. ProgressiveDiffusionScheduler: 渐进增加扩散步数
2. ProgressiveResolutionScheduler: 渐进提高分辨率 (32→64)
"""

from typing import Optional, Tuple


class ProgressiveDiffusionScheduler:
    """
    渐进式扩散步数训练策略.
    
    训练分为两个阶段:
    1. Warmup phase (0 → warmup_epochs):
       - 扩散步数从 min_steps 线性增长到 max_steps
       - 允许模型先学粗糙特征, 再学精细细节
    
    2. Full phase (warmup_epochs → total_epochs):
       - 使用完整的 max_steps 扩散步数
    
    为什么这样做有效?
    - 少量扩散步数 ≈ 粗粒度去噪 ≈ 学习全局结构
    - 更多扩散步数 ≈ 细粒度去噪 ≈ 学习纹理细节
    - 从粗到细的课程学习 (curriculum learning)
    
    推理时的采样步数也渐进增加:
    - 用 DDIM-style 的更少步数采样
    - 训练步数 / 4 作为推理步数 (经验值)
    
    Args:
        max_steps:       Maximum diffusion steps at full training.
        min_steps:       Minimum diffusion steps at start.
        warmup_epochs:   Number of warmup epochs.
        total_epochs:    Total training epochs.
        schedule:        Growth schedule: 'linear', 'cosine', 'step'.
    """

    def __init__(
        self,
        max_steps: int = 100,
        min_steps: int = 10,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        schedule: str = "linear",
    ):
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule

    def get_num_steps(self, current_epoch: int) -> int:
        """
        获取当前 epoch 应使用的扩散步数.
        
        Args:
            current_epoch: Current training epoch (0-indexed).
        Returns:
            Number of diffusion steps to use.
        """
        if current_epoch >= self.warmup_epochs:
            return self.max_steps

        ratio = current_epoch / max(1, self.warmup_epochs)

        if self.schedule == "linear":
            steps = self.min_steps + ratio * (self.max_steps - self.min_steps)
        elif self.schedule == "cosine":
            import math
            # Cosine annealing from min to max
            steps = self.min_steps + (self.max_steps - self.min_steps) * (
                1 - math.cos(math.pi * ratio)
            ) / 2
        elif self.schedule == "step":
            # Discrete jumps: 25%, 50%, 75%, 100%
            milestones = [0.25, 0.5, 0.75, 1.0]
            step_values = [
                self.min_steps,
                self.min_steps + (self.max_steps - self.min_steps) * 0.33,
                self.min_steps + (self.max_steps - self.min_steps) * 0.66,
                self.max_steps,
            ]
            for ms, sv in zip(milestones, step_values):
                if ratio <= ms:
                    steps = sv
                    break
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(self.min_steps, int(steps))

    def get_sampling_steps(self, current_epoch: int) -> int:
        """
        获取推理时的采样步数 (通常为训练步数的 1/4).
        
        Args:
            current_epoch: Current training epoch.
        Returns:
            Number of DDIM sampling steps.
        """
        train_steps = self.get_num_steps(current_epoch)
        return max(5, train_steps // 4)

    def get_schedule_info(self, current_epoch: int) -> dict:
        """Get current schedule information for logging."""
        return {
            "epoch": current_epoch,
            "train_diffusion_steps": self.get_num_steps(current_epoch),
            "sampling_steps": self.get_sampling_steps(current_epoch),
            "phase": "warmup" if current_epoch < self.warmup_epochs else "full",
            "warmup_progress": min(1.0, current_epoch / max(1, self.warmup_epochs)),
        }


class ProgressiveResolutionScheduler:
    """
    渐进式分辨率训练策略.
    
    先在低分辨率 (e.g. 32×32) 训练基础特征,
    再逐步提高到目标分辨率 (e.g. 64×64).
    
    优势:
    - 低分辨率训练速度更快 (像素数减少 4x)
    - 先学全局结构, 再学局部细节
    - 总训练时间减少 ~30%
    
    注意: 需要在模型中支持不同分辨率的输入.
    PatchEmbed 的 num_patches 会随分辨率变化.
    → 使用可插值的位置编码 (interpolatable positional embedding)
    
    Args:
        target_size:     Target resolution (e.g. 64).
        min_size:        Starting resolution (e.g. 32).
        grow_epochs:     List of epochs at which to increase resolution.
        sizes:           List of resolutions at each stage.
    """

    def __init__(
        self,
        target_size: int = 64,
        min_size: int = 32,
        grow_epochs: Optional[list] = None,
        sizes: Optional[list] = None,
    ):
        self.target_size = target_size
        self.min_size = min_size

        # Default: 2 stages (32→64)
        if grow_epochs is None:
            grow_epochs = [0, 20]
        if sizes is None:
            sizes = [min_size, target_size]

        assert len(grow_epochs) == len(sizes), \
            "grow_epochs and sizes must have same length"

        self.grow_epochs = grow_epochs
        self.sizes = sizes

    def get_resolution(self, current_epoch: int) -> int:
        """
        Get current training resolution.
        
        Args:
            current_epoch: Current training epoch.
        Returns:
            Image resolution for this epoch.
        """
        current_size = self.sizes[0]
        for epoch, size in zip(self.grow_epochs, self.sizes):
            if current_epoch >= epoch:
                current_size = size
        return current_size

    def should_increase(self, current_epoch: int) -> bool:
        """Check if resolution should increase at this epoch."""
        return current_epoch in self.grow_epochs and current_epoch > 0

    def get_schedule_info(self, current_epoch: int) -> dict:
        """Get current schedule information for logging."""
        return {
            "epoch": current_epoch,
            "resolution": self.get_resolution(current_epoch),
            "target_resolution": self.target_size,
            "stage": sum(1 for e in self.grow_epochs if current_epoch >= e),
            "total_stages": len(self.grow_epochs),
        }


class CombinedProgressiveScheduler:
    """
    组合渐进式调度器: 同时调度扩散步数和分辨率.
    
    典型配置:
    - Epoch 0-10:  32×32, 10 diffusion steps  → 快速迭代
    - Epoch 10-20: 32×32, 50 diffusion steps  → 提升质量
    - Epoch 20-30: 64×64, 50 diffusion steps  → 增加分辨率
    - Epoch 30+:   64×64, 100 diffusion steps → 全量训练
    
    Args:
        diffusion_scheduler: ProgressiveDiffusionScheduler instance.
        resolution_scheduler: ProgressiveResolutionScheduler instance.
    """

    def __init__(
        self,
        diffusion_scheduler: Optional[ProgressiveDiffusionScheduler] = None,
        resolution_scheduler: Optional[ProgressiveResolutionScheduler] = None,
    ):
        self.diffusion = diffusion_scheduler or ProgressiveDiffusionScheduler()
        self.resolution = resolution_scheduler or ProgressiveResolutionScheduler()

    def get_config(self, current_epoch: int) -> dict:
        """
        Get combined configuration for current epoch.
        
        Returns:
            dict with 'num_diffusion_steps', 'sampling_steps', 'resolution'.
        """
        return {
            "num_diffusion_steps": self.diffusion.get_num_steps(current_epoch),
            "sampling_steps": self.diffusion.get_sampling_steps(current_epoch),
            "resolution": self.resolution.get_resolution(current_epoch),
            "should_resize": self.resolution.should_increase(current_epoch),
        }

    def print_schedule(self, total_epochs: int) -> None:
        """Print the full training schedule."""
        print("\n📋 Progressive Training Schedule:")
        print("-" * 60)
        print(f"{'Epoch':>6} | {'Resolution':>10} | {'Diff Steps':>10} | {'Sample Steps':>12}")
        print("-" * 60)

        prev_config = None
        for epoch in range(total_epochs):
            config = self.get_config(epoch)
            if prev_config is None or config != prev_config:
                print(
                    f"{epoch:>6} | {config['resolution']:>10} | "
                    f"{config['num_diffusion_steps']:>10} | "
                    f"{config['sampling_steps']:>12}"
                )
            prev_config = config

        print("-" * 60)
