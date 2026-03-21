"""
Evaluation Metrics for World Model Quality.

评估世界模型的生成质量和下游 RL 性能.

指标:
1. FID (Fréchet Inception Distance): 生成帧分布 vs 真实帧分布的距离
   - 越低越好, <50 算较好, <20 算很好
   - 使用预训练 InceptionV3 提取特征

2. LPIPS (Learned Perceptual Image Patch Similarity): 感知相似度
   - 越低越好, <0.3 算较好
   - 比 PSNR/SSIM 更符合人类感知

3. SSIM (Structural Similarity): 结构相似性
   - 越高越好, >0.8 算较好
   - 快速计算, 可用于实时监控

4. PSNR (Peak Signal-to-Noise Ratio): 信噪比
   - 越高越好, >25dB 算较好
"""

import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMCalculator:
    """
    Structural Similarity Index (SSIM).
    
    快速, 不需要预训练模型, 适合实时监控.
    
    Args:
        window_size: Gaussian window size.
        channel:     Number of image channels.
    """

    def __init__(self, window_size: int = 11, channel: int = 3):
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    @staticmethod
    def _create_window(window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SSIM between predicted and target images.
        
        Args:
            pred:   (B, C, H, W) predicted images.
            target: (B, C, H, W) ground truth images.
        Returns:
            (B,) SSIM values.
        """
        C = pred.shape[1]
        window = self.window.to(pred.device)
        if C != self.channel:
            window = self._create_window(self.window_size, C).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=C) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean(dim=[1, 2, 3])  # (B,)


class PSNRCalculator:
    """
    Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    """

    @staticmethod
    def compute(
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) predicted images in [0, max_val].
            target: (B, C, H, W) ground truth images.
        Returns:
            (B,) PSNR values in dB.
        """
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
        return psnr


class LPIPSCalculator:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    使用预训练 VGG/AlexNet 网络提取多层特征,
    计算特征空间中的 L2 距离.
    
    比 PSNR/SSIM 更符合人类感知:
    - PSNR 高不代表看起来好 (模糊图像 PSNR 也可能高)
    - LPIPS 低 = 人类感知上更相似
    
    Args:
        net_type:  Backbone network: 'vgg' or 'alex'.
        use_gpu:   Whether to use GPU.
    """

    def __init__(self, net_type: str = "vgg", use_gpu: bool = True):
        self.net_type = net_type
        self.use_gpu = use_gpu
        self._model = None

    @property
    def model(self):
        """Lazy initialization of LPIPS model."""
        if self._model is None:
            try:
                import lpips
                self._model = lpips.LPIPS(net=self.net_type)
                if self.use_gpu and torch.cuda.is_available():
                    self._model = self._model.cuda()
                self._model.eval()
            except ImportError:
                print("Warning: lpips not installed. Using MSE as fallback.")
                self._model = "fallback"
        return self._model

    @torch.no_grad()
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS distance.
        
        Args:
            pred:   (B, C, H, W) predicted images in [0, 1].
            target: (B, C, H, W) ground truth images.
        Returns:
            (B,) LPIPS distances (lower = more similar).
        """
        if self.model == "fallback":
            return F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])

        # LPIPS expects input in [-1, 1]
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1

        return self.model(pred_scaled, target_scaled).squeeze()


class FIDCalculator:
    """
    Fréchet Inception Distance (FID).
    
    FID 衡量生成图像分布与真实图像分布的距离:
    FID = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^{1/2})
    
    使用 InceptionV3 的 pool3 层特征 (2048-d).
    
    注意:
    - FID 对样本量敏感, 至少需要 ~1000 张图像
    - 对图像预处理要求一致 (resize to 299×299)
    - 数值越低越好: FID<10 优秀, FID<50 良好
    
    简化实现: 不使用 pytorch-fid 库, 直接计算.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._inception = None

    @property
    def inception(self):
        """Lazy-load InceptionV3."""
        if self._inception is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
                # Remove classification head — we want pool3 features
                model.fc = nn.Identity()
                model.eval()
                self._inception = model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load InceptionV3: {e}")
                self._inception = "fallback"
        return self._inception

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract InceptionV3 features.
        
        Args:
            images: (N, 3, H, W) images in [0, 1].
        Returns:
            (N, 2048) feature vectors.
        """
        if self.inception == "fallback":
            # Fallback: use flattened, downsampled images
            resized = F.interpolate(images, size=(32, 32), mode="bilinear")
            return resized.reshape(images.shape[0], -1)

        # Resize to 299×299 for InceptionV3
        resized = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False,
        )
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        resized = (resized - mean) / std

        features = self.inception(resized.to(self.device))
        return features

    def compute(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> float:
        """
        Compute FID from pre-extracted features.
        
        Args:
            real_features: (N, D) real image features.
            fake_features: (M, D) generated image features.
        Returns:
            FID score (float).
        """
        # Compute statistics
        mu_real = real_features.mean(dim=0)
        mu_fake = fake_features.mean(dim=0)

        sigma_real = self._compute_cov(real_features)
        sigma_fake = self._compute_cov(fake_features)

        # FID = ||mu_r - mu_g||^2 + Tr(Σ_r + Σ_g - 2 * sqrt(Σ_r·Σ_g))
        diff = mu_real - mu_fake
        mean_term = diff.dot(diff).item()

        # Matrix square root using eigendecomposition
        product = sigma_real @ sigma_fake
        try:
            eigvals, eigvecs = torch.linalg.eigh(product)
            eigvals = torch.clamp(eigvals, min=0)  # Numerical stability
            sqrt_product = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.t()
        except Exception:
            # Fallback: simple approximation
            sqrt_product = torch.zeros_like(product)

        trace_term = (
            torch.trace(sigma_real) + torch.trace(sigma_fake)
            - 2 * torch.trace(sqrt_product)
        ).item()

        return mean_term + trace_term

    @staticmethod
    def _compute_cov(features: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        n = features.shape[0]
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        cov = (centered.t() @ centered) / max(n - 1, 1)
        return cov

    def compute_from_images(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 64,
    ) -> float:
        """
        Compute FID directly from image tensors.
        
        Args:
            real_images: (N, 3, H, W) real images in [0, 1].
            fake_images: (M, 3, H, W) generated images in [0, 1].
        Returns:
            FID score.
        """
        # Extract features in batches
        real_feats = []
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i: i + batch_size]
            real_feats.append(self.extract_features(batch))
        real_feats = torch.cat(real_feats, dim=0)

        fake_feats = []
        for i in range(0, len(fake_images), batch_size):
            batch = fake_images[i: i + batch_size]
            fake_feats.append(self.extract_features(batch))
        fake_feats = torch.cat(fake_feats, dim=0)

        return self.compute(real_feats, fake_feats)


class MetricsTracker:
    """
    统一的指标追踪器.
    
    管理所有评估指标, 支持:
    - 批次级别和 epoch 级别的追踪
    - 自动计算均值、标准差
    - 导出为表格格式
    - 最佳模型选择
    
    Args:
        device: Computation device.
    """

    def __init__(self, device: str = "cuda"):
        self.ssim = SSIMCalculator()
        self.psnr = PSNRCalculator()
        self.lpips_calc = LPIPSCalculator(use_gpu=(device == "cuda"))
        self.fid_calc = FIDCalculator(device=device)

        self.history: Dict[str, List[float]] = {
            "ssim": [], "psnr": [], "lpips": [], "fid": [],
        }

    @torch.no_grad()
    def evaluate_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            pred:   (B, C, H, W) predicted frames in [0, 1].
            target: (B, C, H, W) ground truth frames.
        Returns:
            dict with ssim, psnr, lpips values.
        """
        ssim = self.ssim.compute(pred, target).mean().item()
        psnr = self.psnr.compute(pred, target).mean().item()

        try:
            lpips_val = self.lpips_calc.compute(pred, target).mean().item()
        except Exception:
            lpips_val = 0.0

        metrics = {"ssim": ssim, "psnr": psnr, "lpips": lpips_val}

        for k, v in metrics.items():
            self.history[k].append(v)

        return metrics

    def evaluate_fid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> float:
        """
        Compute FID (requires larger sample size).
        
        Args:
            real_images: (N, 3, H, W) real images.
            fake_images: (M, 3, H, W) generated images.
        Returns:
            FID score.
        """
        fid = self.fid_calc.compute_from_images(real_images, fake_images)
        self.history["fid"].append(fid)
        return fid

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}
        for name, values in self.history.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "last": float(arr[-1]),
                }
        return summary

    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.get_summary()
        print("\n📊 Evaluation Metrics Summary:")
        print("-" * 60)
        print(f"{'Metric':>10} | {'Mean':>10} | {'Std':>10} | {'Best':>10} | {'Last':>10}")
        print("-" * 60)
        for name, stats in summary.items():
            best = stats["max"] if name in ("ssim", "psnr") else stats["min"]
            print(
                f"{name:>10} | {stats['mean']:>10.4f} | {stats['std']:>10.4f} | "
                f"{best:>10.4f} | {stats['last']:>10.4f}"
            )
        print("-" * 60)

    def reset(self) -> None:
        """Clear all tracked metrics."""
        for key in self.history:
            self.history[key] = []
