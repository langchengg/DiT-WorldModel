"""
Navigation-Specific Visualization Tools.

提供导航任务专用的可视化:
1. visualize_imagination   — 展示 World Model 想象的未来画面
2. visualize_navigation    — 绘制导航轨迹 + 帧时间线
3. create_navigation_video — 生成导航 MP4 视频
4. plot_navigation_metrics — 绘制导航指标曲线
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) float tensor → (H, W, C) uint8 numpy."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1) * 255
    return arr.astype(np.uint8)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Imagination Visualization
# ---------------------------------------------------------------------------

def visualize_imagination(
    imagined_frames: List[torch.Tensor],
    current_obs: Optional[torch.Tensor] = None,
    goal_image: Optional[torch.Tensor] = None,
    title: str = "World Model Imagination",
    output_path: str = "results/imagination.png",
    max_frames: int = 8,
) -> Optional[np.ndarray]:
    """
    Visualize imagined future frames in a single row.

    Layout:
        [Current | Imagined_t1 | Imagined_t2 | ... | Goal]

    Args:
        imagined_frames: List of (3, H, W) predicted frames.
        current_obs:     Optional (3, H, W) current observation.
        goal_image:      Optional (3, H, W) goal observation.
        title:           Figure title.
        output_path:     Where to save.
        max_frames:      Maximum imagined frames to show.

    Returns:
        numpy array of the figure, or None if matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return None

    _ensure_dir(str(Path(output_path).parent))

    frames_to_show = []
    labels = []

    if current_obs is not None:
        frames_to_show.append(_to_numpy(current_obs))
        labels.append("Current (t=0)")

    step_size = max(1, len(imagined_frames) // max_frames)
    for i in range(0, len(imagined_frames), step_size):
        if len(frames_to_show) >= max_frames + 2:
            break
        frames_to_show.append(_to_numpy(imagined_frames[i]))
        labels.append(f"t={i + 1}")

    if goal_image is not None:
        frames_to_show.append(_to_numpy(goal_image))
        labels.append("Goal")

    n = len(frames_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, frame, label in zip(axes, frames_to_show, labels):
        ax.imshow(frame)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"💭 Imagination visualization saved to {output_path}")
    return np.concatenate(frames_to_show, axis=1)


# ---------------------------------------------------------------------------
# 2. Navigation Episode Visualization
# ---------------------------------------------------------------------------

def visualize_navigation(
    trajectory: List[torch.Tensor],
    actions: List[int],
    rewards: List[float],
    imaginations: Optional[List[List[torch.Tensor]]] = None,
    title: str = "Navigation Episode",
    output_path: str = "results/navigation_episode.png",
    show_every: int = 5,
) -> None:
    """
    Visualize a full navigation episode.

    Layout:
        Row 1: Actual trajectory frames (sampled)
        Row 2: Imagined futures at each decision point
        Row 3: Reward/action timeline

    Args:
        trajectory:   List of (3, H, W) actual observations.
        actions:      List of int actions taken.
        rewards:      List of float rewards received.
        imaginations: Optional list of imagined futures at each step.
        title:        Figure title.
        output_path:  Where to save.
        show_every:   Show every N-th frame.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    _ensure_dir(str(Path(output_path).parent))

    # Select frames to show
    indices = list(range(0, len(trajectory), show_every))
    if len(trajectory) - 1 not in indices:
        indices.append(len(trajectory) - 1)
    n_show = len(indices)

    has_imagination = (imaginations is not None and len(imaginations) > 0)
    n_rows = 3 if has_imagination else 2

    fig = plt.figure(figsize=(3 * n_show, 3 * n_rows))
    gs = GridSpec(n_rows, n_show, figure=fig)

    # Row 1: Actual trajectory
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_to_numpy(trajectory[idx]))
        ax.set_title(f"t={idx}", fontsize=9)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Actual", fontsize=11)

    # Row 2: Imagined futures (first imagined frame at each decision point)
    if has_imagination:
        for col, idx in enumerate(indices):
            ax = fig.add_subplot(gs[1, col])
            if idx < len(imaginations) and len(imaginations[idx]) > 0:
                ax.imshow(_to_numpy(imaginations[idx][0]))
                ax.set_title(f"Imagine t={idx}→{idx + 1}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel("Imagined", fontsize=11)

    # Row 3 (or 2): Reward/action timeline
    ax_rew = fig.add_subplot(gs[-1, :])
    steps = list(range(len(rewards)))
    ax_rew.bar(steps, rewards, alpha=0.6, label="Reward", color="steelblue")
    ax_rew.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Reward Timeline")
    ax_rew.legend(fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"🗺️  Navigation visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# 3. Navigation Video
# ---------------------------------------------------------------------------

def create_navigation_video(
    trajectory: List[torch.Tensor],
    output_path: str = "results/navigation.mp4",
    fps: int = 5,
    imaginations: Optional[List[List[torch.Tensor]]] = None,
) -> str:
    """
    Create MP4 video of navigation episode.

    If imaginations are provided, shows a side-by-side view:
        [Actual | Imagined future]

    Args:
        trajectory:   List of (3, H, W) observations.
        output_path:  Video output path.
        fps:          Frames per second.
        imaginations: Optional imagined futures at each step.

    Returns:
        Path to saved video.
    """
    _ensure_dir(str(Path(output_path).parent))

    try:
        import imageio
    except ImportError:
        print("imageio not installed, saving frames as numpy instead")
        np.savez(
            output_path.replace(".mp4", ".npz"),
            frames=[_to_numpy(f) for f in trajectory],
        )
        return output_path.replace(".mp4", ".npz")

    writer = imageio.get_writer(output_path, fps=fps)

    for t, frame in enumerate(trajectory):
        actual = _to_numpy(frame)

        if imaginations is not None and t < len(imaginations) and len(imaginations[t]) > 0:
            imagined = _to_numpy(imaginations[t][0])
            # Resize imagined to match actual
            if imagined.shape != actual.shape:
                from PIL import Image
                imagined = np.array(
                    Image.fromarray(imagined).resize(
                        (actual.shape[1], actual.shape[0])
                    )
                )
            # Side by side
            combined = np.concatenate([actual, imagined], axis=1)
        else:
            combined = actual

        writer.append_data(combined)

    writer.close()
    print(f"🎬 Navigation video saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Navigation Metrics Plot
# ---------------------------------------------------------------------------

def plot_navigation_metrics(
    results: List[Dict[str, Any]],
    output_path: str = "results/navigation_metrics.png",
) -> None:
    """
    Plot navigation metrics across multiple episodes.

    Args:
        results: List of navigation result dicts (from navigate()).
        output_path: Where to save.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    _ensure_dir(str(Path(output_path).parent))

    successes = [r["success"] for r in results]
    steps = [r["total_steps"] for r in results]
    total_rewards = [sum(r["rewards"]) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Success rate
    sr = sum(successes) / max(len(successes), 1) * 100
    axes[0].bar(["Success", "Failure"],
                [sum(successes), len(successes) - sum(successes)],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_title(f"Success Rate: {sr:.1f}%")
    axes[0].set_ylabel("Count")

    # Steps distribution
    axes[1].hist(steps, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(np.mean(steps), color="red", linestyle="--",
                    label=f"Mean: {np.mean(steps):.1f}")
    axes[1].set_title("Steps Distribution")
    axes[1].set_xlabel("Steps")
    axes[1].legend()

    # Total reward
    axes[2].plot(total_rewards, "o-", markersize=3, color="orange")
    axes[2].axhline(np.mean(total_rewards), color="red", linestyle="--",
                    label=f"Mean: {np.mean(total_rewards):.1f}")
    axes[2].set_title("Total Reward per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].legend()

    plt.suptitle("Navigation Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Navigation metrics saved to {output_path}")
