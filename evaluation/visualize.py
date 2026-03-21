"""
Visualization Tools for World Model Evaluation.

提供多种可视化工具:
1. FrameVisualizer: 生成帧 vs 真实帧逐帧对比
2. TrajectoryVisualizer: 多步自回归预测轨迹
3. DiffusionProcessVisualizer: 扩散过程中间步骤
4. save_comparison_grid: 批量生成对比网格图
5. create_video: 生成 MP4 视频

所有可视化输出保存到 results/figures/ 目录.
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def _ensure_dir(path: str) -> Path:
    """Create directory if not exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy for visualization."""
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]  # Take first image
    if tensor.dim() == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # → (H, W, C)
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1) * 255
    return arr.astype(np.uint8)


class FrameVisualizer:
    """
    生成帧 vs 真实帧可视化对比器.
    
    创建 side-by-side 对比图:
    [Ground Truth | Predicted | Difference Map]
    
    差异图 (difference map) 突出显示预测不准确的区域,
    帮助分析模型的弱点 (如细节纹理、物体边缘、远处区域).
    
    Args:
        output_dir: 输出目录.
    """

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = _ensure_dir(output_dir)

    def compare_frames(
        self,
        gt_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        step: int = 0,
        filename: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create comparison image: GT | Pred | Diff.
        
        Args:
            gt_frames:   (B, C, H, W) or (C, H, W) ground truth.
            pred_frames: (B, C, H, W) or (C, H, W) predictions.
            step:        Training step (for filename).
            filename:    Optional custom filename.
        Returns:
            (H, W*3, C) comparison image as numpy array.
        """
        if gt_frames.dim() == 4:
            gt_frames = gt_frames[0]
            pred_frames = pred_frames[0]

        gt_np = _to_numpy(gt_frames)       # (H, W, C)
        pred_np = _to_numpy(pred_frames)    # (H, W, C)

        # Difference map (amplified for visibility)
        diff = np.abs(gt_np.astype(float) - pred_np.astype(float))
        diff = np.clip(diff * 3, 0, 255).astype(np.uint8)

        # Concatenate horizontally
        comparison = np.concatenate([gt_np, pred_np, diff], axis=1)

        # Save
        if filename is None:
            filename = f"comparison_step_{step:06d}.png"

        try:
            from PIL import Image
            img = Image.fromarray(comparison)
            img.save(str(self.output_dir / filename))
        except ImportError:
            # Fallback: save as numpy
            np.save(str(self.output_dir / filename.replace(".png", ".npy")), comparison)

        return comparison

    def create_grid(
        self,
        gt_batch: torch.Tensor,
        pred_batch: torch.Tensor,
        nrow: int = 4,
        filename: str = "grid_comparison.png",
    ) -> np.ndarray:
        """
        Create grid comparison for a batch of images.
        
        Layout:
        Row 1: GT_1  | GT_2  | GT_3  | GT_4
        Row 2: Pred_1| Pred_2| Pred_3| Pred_4
        
        Args:
            gt_batch:   (B, C, H, W) ground truth batch.
            pred_batch: (B, C, H, W) prediction batch.
            nrow:       Images per row.
            filename:   Output filename.
        """
        B = min(gt_batch.shape[0], nrow * 2)
        num_show = min(B, nrow)

        rows = []

        # GT row
        gt_row = [_to_numpy(gt_batch[i]) for i in range(num_show)]
        rows.append(np.concatenate(gt_row, axis=1))

        # Pred row
        pred_row = [_to_numpy(pred_batch[i]) for i in range(num_show)]
        rows.append(np.concatenate(pred_row, axis=1))

        grid = np.concatenate(rows, axis=0)

        try:
            from PIL import Image
            img = Image.fromarray(grid)
            img.save(str(self.output_dir / filename))
        except ImportError:
            np.save(str(self.output_dir / filename.replace(".png", ".npy")), grid)

        return grid


class TrajectoryVisualizer:
    """
    多步自回归预测轨迹可视化.
    
    展示世界模型在多步预测中的衰减:
    - 前几步: 通常质量很好
    - 后面的步: 误差会累积 (error compounding)
    
    通过可视化帮助判断:
    - 多少步后质量开始明显下降?
    - 误差主要出现在哪里? (背景? 前景物体? 纹理?)
    
    Args:
        output_dir: Output directory.
    """

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = _ensure_dir(output_dir)

    def visualize_trajectory(
        self,
        gt_trajectory: List[torch.Tensor],
        pred_trajectory: List[torch.Tensor],
        filename: str = "trajectory.png",
        max_steps: int = 10,
    ) -> np.ndarray:
        """
        Visualize multi-step prediction trajectory.
        
        Layout:
        Row 1 (GT):   t=0 | t=1 | t=2 | ... | t=N
        Row 2 (Pred): t=0 | t=1 | t=2 | ... | t=N
        Row 3 (Diff): d=0 | d=1 | d=2 | ... | d=N
        
        Args:
            gt_trajectory:   List of (C, H, W) ground truth frames.
            pred_trajectory: List of (C, H, W) predicted frames.
            filename:        Output filename.
            max_steps:       Maximum steps to show.
        """
        T = min(len(gt_trajectory), len(pred_trajectory), max_steps)

        rows = [[], [], []]  # GT, Pred, Diff

        for t in range(T):
            gt_np = _to_numpy(gt_trajectory[t])
            pred_np = _to_numpy(pred_trajectory[t])
            diff_np = np.clip(
                np.abs(gt_np.astype(float) - pred_np.astype(float)) * 3,
                0, 255,
            ).astype(np.uint8)

            rows[0].append(gt_np)
            rows[1].append(pred_np)
            rows[2].append(diff_np)

        # Concatenate
        grid_rows = [np.concatenate(row, axis=1) for row in rows]
        grid = np.concatenate(grid_rows, axis=0)

        try:
            from PIL import Image
            img = Image.fromarray(grid)
            img.save(str(self.output_dir / filename))
        except ImportError:
            np.save(str(self.output_dir / filename.replace(".png", ".npy")), grid)

        return grid

    def compute_trajectory_metrics(
        self,
        gt_trajectory: List[torch.Tensor],
        pred_trajectory: List[torch.Tensor],
    ) -> dict:
        """
        Compute per-step metrics along trajectory.
        
        Useful for analyzing error accumulation:
        - If MSE grows linearly → constant per-step error
        - If MSE grows exponentially → error compounding
        
        Returns:
            dict: {step: {mse, psnr}} for each timestep.
        """
        from .metrics import SSIMCalculator, PSNRCalculator
        ssim_calc = SSIMCalculator()
        psnr_calc = PSNRCalculator()

        results = {}
        T = min(len(gt_trajectory), len(pred_trajectory))

        for t in range(T):
            gt = gt_trajectory[t].unsqueeze(0)    # (1, C, H, W)
            pred = pred_trajectory[t].unsqueeze(0)

            mse = F.mse_loss(pred, gt).item()
            ssim = ssim_calc.compute(pred, gt).item()
            psnr = psnr_calc.compute(pred, gt).item()

            results[t] = {"mse": mse, "ssim": ssim, "psnr": psnr}

        return results


class DiffusionStepsVisualizer:
    """
    可视化扩散过程的中间步骤.
    
    展示从纯噪声到清晰图像的渐进去噪过程:
    x_T (pure noise) → x_{T/4} → x_{T/2} → x_{3T/4} → x_0 (clean)
    
    帮助理解:
    - 前几步: 去除高频噪声, 出现大致形状
    - 中间步: 细化纹理和边缘
    - 最后几步: 微调细节
    
    Args:
        output_dir: Output directory.
    """

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = _ensure_dir(output_dir)

    def visualize_denoising(
        self,
        intermediates: List[torch.Tensor],
        filename: str = "denoising_process.png",
        num_show: int = 8,
    ) -> np.ndarray:
        """
        Visualize denoising steps.
        
        Args:
            intermediates: List of (C, H, W) intermediate states.
            filename:      Output filename.
            num_show:      Number of steps to display.
        """
        T = len(intermediates)
        # Select evenly spaced steps
        indices = np.linspace(0, T - 1, num_show, dtype=int)

        frames = []
        for idx in indices:
            frame = intermediates[idx]
            if frame.dim() == 4:
                frame = frame[0]
            frame_np = _to_numpy(frame)
            frames.append(frame_np)

        # Concatenate into single row
        strip = np.concatenate(frames, axis=1)

        try:
            from PIL import Image
            img = Image.fromarray(strip)
            img.save(str(self.output_dir / filename))
        except ImportError:
            np.save(str(self.output_dir / filename.replace(".png", ".npy")), strip)

        return strip


def create_video(
    frames: List[torch.Tensor],
    output_path: str = "results/video.mp4",
    fps: int = 10,
) -> str:
    """
    Generate MP4 video from frame sequence.
    
    Args:
        frames:      List of (C, H, W) tensors in [0, 1].
        output_path: Output video path.
        fps:         Frames per second.
    Returns:
        Path to saved video.
    """
    _ensure_dir(str(Path(output_path).parent))

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            frame_np = _to_numpy(frame)
            writer.append_data(frame_np)
        writer.close()
        print(f"Video saved to {output_path}")
    except ImportError:
        print("imageio not installed. Saving frames as numpy.")
        np.savez(
            output_path.replace(".mp4", ".npz"),
            frames=[_to_numpy(f) for f in frames],
        )

    return output_path


def plot_training_curves(
    history: dict,
    output_path: str = "results/figures/training_curves.png",
) -> None:
    """
    Plot training loss curves.
    
    Args:
        history: dict with lists of metrics (e.g. from trainer.fit()).
        output_path: Output image path.
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
        if len(history) == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, history.items()):
            ax.plot(values, linewidth=1.5)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to {output_path}")

    except ImportError:
        print("matplotlib not installed. Skipping plot.")


def plot_ablation_table(
    results: dict,
    output_path: str = "results/figures/ablation_table.png",
) -> None:
    """
    Generate ablation experiment table as an image.
    
    Args:
        results: dict of {method_name: {metric: value}}.
        output_path: Output path.
    """
    try:
        import matplotlib.pyplot as plt

        methods = list(results.keys())
        metrics = list(results[methods[0]].keys())

        fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.6)))
        ax.axis("off")

        # Table data
        cell_text = []
        for method in methods:
            row = [f"{results[method].get(m, '-')}" for m in metrics]
            cell_text.append(row)

        table = ax.table(
            cellText=cell_text,
            rowLabels=methods,
            colLabels=metrics,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.title("Ablation Study Results", fontsize=14, pad=20)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        print("matplotlib not installed. Printing table to console.")
        print("\nAblation Results:")
        for method, metrics_dict in results.items():
            print(f"  {method}: {metrics_dict}")
