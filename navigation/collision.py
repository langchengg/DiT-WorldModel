"""
Collision Detection from Predicted Frames.

提供三种碰撞检测方案, 从预测画面判断是否即将碰撞:

1. DoneSignalCollisionDetector
   - 复用 World Model 的 done head 输出
   - 零额外参数, 最简单

2. CNNCollisionDetector
   - 轻量 CNN 分类器
   - 从 RGB 帧直接判断是否碰撞
   - 需要额外训练 (可在 GridNavigationEnv 上快速训练)

3. DepthBasedCollisionDetector
   - 用 MiDaS 单目深度估计
   - 判断前方最近障碍距离
   - 最鲁棒, 但最慢
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoneSignalCollisionDetector:
    """
    基于 World Model done head 的碰撞检测.

    思路: World Model 预测 done=True → 回合结束 → 在导航中 = 碰撞.
    直接复用 World Model 的 done logits, 无需额外训练.

    Args:
        threshold: Probability threshold above which we declare collision.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, done_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            done_logits: (B, 2) done logits from World Model.
        Returns:
            (B,) collision probability in [0, 1].
        """
        probs = F.softmax(done_logits, dim=-1)
        collision_prob = probs[:, 1]  # P(done=True)
        return collision_prob

    def is_collision(self, done_logits: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B,) bool tensor — True if collision detected.
        """
        return self.predict(done_logits) > self.threshold


class CNNCollisionDetector(nn.Module):
    """
    轻量 CNN 碰撞分类器.

    从 RGB 帧直接分类: 碰撞 vs 安全.
    可在 GridNavigationEnv 中快速训练:
    - 收集碰撞/非碰撞帧 → 二分类训练
    - 几分钟 CPU 即可训练完成

    Architecture:
        Conv(3→16, 4, s2) → ReLU → Conv(16→32, 4, s2) → ReLU →
        Conv(32→64, 4, s2) → ReLU → AdaptiveAvgPool(1) → FC(64→1)

    Args:
        img_size:  Input image size.
        threshold: Decision threshold.
    """

    def __init__(self, img_size: int = 64, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),   # → H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # → H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # → H/8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                     # → 1×1
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (B, 3, H, W) RGB frame in [0, 1].
        Returns:
            (B,) collision probability in [0, 1].
        """
        feat = self.features(frame).squeeze(-1).squeeze(-1)  # (B, 64)
        logit = self.classifier(feat).squeeze(-1)             # (B,)
        return torch.sigmoid(logit)

    def is_collision(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B,) bool tensor.
        """
        return self.forward(frame) > self.threshold

    @staticmethod
    def train_from_env(
        img_size: int = 64,
        num_samples: int = 5000,
        epochs: int = 20,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> "CNNCollisionDetector":
        """
        Train a collision detector on data from GridNavigationEnv.

        Collects frames and labels:
            label=1 if the action resulted in a collision (wall hit)
            label=0 otherwise

        Args:
            img_size:    Image resolution.
            num_samples: Total frames to collect.
            epochs:      Training epochs.
            lr:          Learning rate.
            device:      Training device.

        Returns:
            Trained CNNCollisionDetector.
        """
        from .sim_env import GridNavigationEnv

        env = GridNavigationEnv(grid_size=8, img_size=img_size, seed=42)
        frames, labels = [], []

        obs = env.reset()
        for _ in range(num_samples):
            action = env.sample_action()
            next_obs, reward, done, info = env.step(action)

            frames.append(obs)
            labels.append(1.0 if info["collision"] else 0.0)
            obs = next_obs

            if done:
                obs = env.reset()

        X = torch.stack(frames).to(device)
        y = torch.tensor(labels, dtype=torch.float32).to(device)

        model = CNNCollisionDetector(img_size=img_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dataset_size = len(X)
        batch_size = min(128, dataset_size)

        print(f"🏋️ Training collision detector: {dataset_size} samples, "
              f"{epochs} epochs ...")

        for epoch in range(epochs):
            perm = torch.randperm(dataset_size)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, dataset_size, batch_size):
                idx = perm[i: i + batch_size]
                pred = model(X[idx])
                loss = F.binary_cross_entropy(pred, y[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                acc = ((model(X) > 0.5).float() == y).float().mean()
                print(f"  Epoch {epoch + 1}: loss={total_loss / n_batches:.4f}, "
                      f"acc={acc:.3f}")

        model.eval()
        print("  ✅ Collision detector trained")
        return model


class DepthBasedCollisionDetector:
    """
    基于单目深度估计的碰撞检测.

    使用 MiDaS 估计 RGB → depth, 然后判断前方区域
    (图像下半部分中央) 的最小深度是否低于阈值.

    NOTE: 需要安装 torch hub 上的 MiDaS.

    Args:
        min_depth_threshold: 深度低于此值 → 碰撞.
        device:              Computation device.
    """

    def __init__(
        self,
        min_depth_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.min_depth_threshold = min_depth_threshold
        self.device = device
        self._model = None
        self._transform = None

    def _load_model(self) -> None:
        """Lazy-load MiDaS model."""
        if self._model is not None:
            return
        try:
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True,
            )
            self._model.eval().to(self.device)

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True,
            )
            self._transform = midas_transforms.small_transform
        except Exception as e:
            print(f"⚠️ Could not load MiDaS: {e}")
            print("  Falling back to simple depth heuristic.")
            self._model = "fallback"

    @torch.no_grad()
    def predict(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (B, 3, H, W) RGB in [0, 1].
        Returns:
            (B,) collision probability estimated from depth.
        """
        self._load_model()

        if self._model == "fallback":
            # Fallback: check if lower-center of image is dark (= close wall)
            B, C, H, W = frame.shape
            roi = frame[:, :, H // 2:, W // 4: 3 * W // 4]
            brightness = roi.mean(dim=(1, 2, 3))
            # Dark = close obstacle → high collision probability
            return (1.0 - brightness).clamp(0, 1)

        # MiDaS forward
        B = frame.shape[0]
        collision_probs = []

        for i in range(B):
            img_np = (frame[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            import cv2
            input_batch = self._transform(img_np).to(self.device)
            depth = self._model(input_batch)
            depth = F.interpolate(
                depth.unsqueeze(1), size=frame.shape[2:], mode="bilinear",
            ).squeeze(1)

            # Check front region
            H, W = depth.shape[1], depth.shape[2]
            front = depth[:, H // 2:, W // 4: 3 * W // 4]
            min_depth = front.min()
            collision_prob = (1.0 - min_depth / self.min_depth_threshold).clamp(0, 1)
            collision_probs.append(collision_prob)

        return torch.stack(collision_probs)

    def is_collision(self, frame: torch.Tensor) -> torch.Tensor:
        return self.predict(frame) > 0.5
