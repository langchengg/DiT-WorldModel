"""
Temporally-Consistent Data Augmentation for World Models.

创新 5: 针对机器人世界模型的数据增强策略.

关键洞察:
- 机器人数据收集成本极高, 离线数据量通常很小 → 数据增强至关重要
- World Model 的增强需要保持时序一致性 (temporal consistency)
- 不能简单地对每帧独立增强 (会破坏时序连贯性, 模型会学到假的运动)
- 正确做法: 对整个序列应用相同的随机变换参数

包含 4 种增强策略:
1. ConsistentColorJitter: 序列级颜色抖动
2. SpatialConsistentCrop: 序列级随机裁剪
3. CameraViewpointNoise: 模拟相机抖动
4. TemporalDropout: 随机丢帧 (模拟帧率变化)
"""

import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F


class ConsistentColorJitter:
    """
    时序一致的颜色抖动.
    
    关键: 对同一序列的所有帧使用相同的颜色变换参数.
    
    为什么不能每帧独立增强?
    - 如果帧 t 变亮, 帧 t+1 变暗 → 模型会误认为场景亮度在变化
    - 这会污染世界模型对光照变化的建模
    
    参数:
        brightness: 亮度变化范围 [1-b, 1+b]
        contrast:   对比度变化范围 [1-c, 1+c]
        saturation: 饱和度变化范围 [1-s, 1+s]
        hue:        色相变化范围 [-h, h]
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def sample_params(self) -> Dict[str, float]:
        """随机采样一组颜色增强参数 (用于整个序列)."""
        return {
            "brightness": 1.0 + random.uniform(-self.brightness, self.brightness),
            "contrast": 1.0 + random.uniform(-self.contrast, self.contrast),
            "saturation": 1.0 + random.uniform(-self.saturation, self.saturation),
            "hue": random.uniform(-self.hue, self.hue),
        }

    def apply(
        self,
        frame: torch.Tensor,
        params: Dict[str, float],
    ) -> torch.Tensor:
        """
        对单帧应用颜色增强.
        
        Args:
            frame:  (C, H, W) 单帧图像, 值域 [0, 1].
            params: 增强参数 (由 sample_params 生成).
        Returns:
            (C, H, W) 增强后的帧.
        """
        # Brightness
        frame = frame * params["brightness"]

        # Contrast (相对于均值)
        mean = frame.mean()
        frame = (frame - mean) * params["contrast"] + mean

        # Saturation (simplified: blend with grayscale)
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            gray = gray.unsqueeze(0).expand_as(frame)
            frame = frame * params["saturation"] + gray * (1 - params["saturation"])

        return frame.clamp(0, 1)

    def __call__(
        self,
        sequence: List[torch.Tensor],
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        对整个序列应用一致的颜色增强.
        
        Args:
            sequence: List of (C, H, W) frames.
            params:   Optional pre-sampled params (for reproducibility).
        Returns:
            augmented_sequence, params
        """
        if params is None:
            params = self.sample_params()

        augmented = [self.apply(frame, params) for frame in sequence]
        return augmented, params


class SpatialConsistentCrop:
    """
    时序一致的空间裁剪.
    
    对整个序列使用相同的裁剪位置和大小.
    这是最基本的空间增强, 在所有视觉任务中都有效.
    
    Args:
        output_size:  Output crop size (square).
        scale_range:  Random scale factor range for crop area.
    """

    def __init__(
        self,
        output_size: int = 64,
        scale_range: Tuple[float, float] = (0.8, 1.0),
    ):
        self.output_size = output_size
        self.scale_range = scale_range

    def sample_params(
        self, img_h: int, img_w: int,
    ) -> Dict[str, int]:
        """Sample crop parameters for the sequence."""
        scale = random.uniform(*self.scale_range)
        crop_h = int(img_h * scale)
        crop_w = int(img_w * scale)

        top = random.randint(0, max(0, img_h - crop_h))
        left = random.randint(0, max(0, img_w - crop_w))

        return {
            "top": top,
            "left": left,
            "crop_h": crop_h,
            "crop_w": crop_w,
        }

    def apply(
        self,
        frame: torch.Tensor,
        params: Dict[str, int],
    ) -> torch.Tensor:
        """Apply crop + resize to a single frame."""
        _, H, W = frame.shape
        cropped = frame[
            :,
            params["top"]: params["top"] + params["crop_h"],
            params["left"]: params["left"] + params["crop_w"],
        ]
        # Resize to output_size
        resized = F.interpolate(
            cropped.unsqueeze(0),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized

    def __call__(
        self,
        sequence: List[torch.Tensor],
        params: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, int]]:
        if params is None:
            _, H, W = sequence[0].shape
            params = self.sample_params(H, W)

        augmented = [self.apply(frame, params) for frame in sequence]
        return augmented, params


class CameraViewpointNoise:
    """
    模拟相机视角轻微抖动.
    
    通过平移 (shift) 图像模拟相机微小移动.
    同一序列使用相同的 shift 量 → 保持时序一致性.
    
    这增强了 World Model 对视角变化的鲁棒性,
    在 sim-to-real transfer 中尤为重要 (真实相机总有微小晃动).
    
    Args:
        max_shift: 最大平移像素数.
    """

    def __init__(self, max_shift: int = 3):
        self.max_shift = max_shift

    def sample_params(self) -> Dict[str, int]:
        """Sample shift parameters."""
        return {
            "dx": random.randint(-self.max_shift, self.max_shift),
            "dy": random.randint(-self.max_shift, self.max_shift),
        }

    def apply(
        self,
        frame: torch.Tensor,
        params: Dict[str, int],
    ) -> torch.Tensor:
        """Apply pixel shift to a single frame."""
        return torch.roll(
            frame,
            shifts=(params["dy"], params["dx"]),
            dims=(-2, -1),
        )

    def __call__(
        self,
        sequence: List[torch.Tensor],
        params: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, int]]:
        if params is None:
            params = self.sample_params()

        augmented = [self.apply(frame, params) for frame in sequence]
        return augmented, params


class TemporalDropout:
    """
    时序丢帧增强.
    
    随机丢弃序列中的一些帧, 然后用相邻帧填补.
    模拟不同帧率的数据, 增强模型对时间间隔变化的鲁棒性.
    
    注意: 丢帧率不能太高, 否则会破坏动作-观测对应关系.
    建议 drop_rate ≤ 0.2.
    
    Args:
        drop_rate: 丢帧概率 (每帧独立地以此概率被丢弃).
    """

    def __init__(self, drop_rate: float = 0.1):
        self.drop_rate = drop_rate

    def __call__(
        self,
        sequence: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply temporal dropout. Dropped frames are replaced by
        the most recent non-dropped frame.
        """
        if len(sequence) <= 2:
            return sequence

        # Never drop the first and last frame
        result = [sequence[0]]
        last_valid = sequence[0]

        for i in range(1, len(sequence) - 1):
            if random.random() > self.drop_rate:
                result.append(sequence[i])
                last_valid = sequence[i]
            else:
                result.append(last_valid)  # Fill with last valid frame

        result.append(sequence[-1])
        return result


class GaussianNoise:
    """
    高斯噪声注入 (时序一致).
    
    为整个序列添加相同的噪声模式.
    模拟传感器噪声, 增强鲁棒性.
    
    Args:
        std: Noise standard deviation.
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(
        self,
        sequence: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Add same Gaussian noise pattern to all frames."""
        noise = torch.randn_like(sequence[0]) * self.std
        return [(frame + noise).clamp(0, 1) for frame in sequence]


class RandomHorizontalFlip:
    """
    随机水平翻转 (时序一致).
    
    注意: 需要同时翻转动作!
    - 对 Atari: 某些游戏 (如 Pong) 可以翻转
    - 对机器人: 需要同时翻转 x/yaw 方向的动作
    
    Args:
        p: Flip probability.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        sequence: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], bool]:
        """
        Returns:
            augmented_sequence, did_flip (for action transformation).
        """
        if random.random() < self.p:
            flipped = [torch.flip(frame, dims=[-1]) for frame in sequence]
            return flipped, True
        return sequence, False


class RoboticAugmentationPipeline:
    """
    完整的机器人数据增强管道.
    
    组合多种增强策略, 支持自定义概率.
    所有增强都保持时序一致性.
    
    默认配置 (针对机器人):
    - ColorJitter: 80% 概率, 弱增强 (机器人场景光照较稳定)
    - SpatialCrop: 50% 概率 (只在有足够分辨率时使用)
    - CameraViewpoint: 70% 概率 (模拟相机抖动)
    - TemporalDropout: 30% 概率 (模拟帧率变化)
    - GaussianNoise: 50% 概率 (模拟传感器噪声)
    
    Args:
        img_size:        Target image size.
        enable_color:    Enable color jitter.
        enable_crop:     Enable spatial crop.
        enable_camera:   Enable camera viewpoint noise.
        enable_temporal: Enable temporal dropout.
        enable_noise:    Enable Gaussian noise.
        enable_flip:     Enable horizontal flip.
    """

    def __init__(
        self,
        img_size: int = 64,
        enable_color: bool = True,
        enable_crop: bool = True,
        enable_camera: bool = True,
        enable_temporal: bool = True,
        enable_noise: bool = True,
        enable_flip: bool = False,  # Off by default for robot
    ):
        self.augmentations = []

        if enable_color:
            self.augmentations.append(
                ("color_jitter", ConsistentColorJitter(0.15, 0.15, 0.15, 0.03), 0.8)
            )
        if enable_crop:
            self.augmentations.append(
                ("spatial_crop", SpatialConsistentCrop(img_size, (0.85, 1.0)), 0.5)
            )
        if enable_camera:
            self.augmentations.append(
                ("camera_noise", CameraViewpointNoise(max_shift=2), 0.7)
            )
        if enable_temporal:
            self.augmentations.append(
                ("temporal_dropout", TemporalDropout(drop_rate=0.1), 0.3)
            )
        if enable_noise:
            self.augmentations.append(
                ("gaussian_noise", GaussianNoise(std=0.015), 0.5)
            )
        if enable_flip:
            self.augmentations.append(
                ("horizontal_flip", RandomHorizontalFlip(p=0.5), 0.5)
            )

    def __call__(
        self,
        sequence: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Apply augmentation pipeline to a frame sequence.
        
        Args:
            sequence: List of (C, H, W) frames in [0, 1].
        
        Returns:
            augmented_sequence, augmentation_info.
        """
        info = {}

        for name, aug, prob in self.augmentations:
            if random.random() < prob:
                if name in ("temporal_dropout", "gaussian_noise"):
                    sequence = aug(sequence)
                    info[name] = True
                elif name == "horizontal_flip":
                    sequence, did_flip = aug(sequence)
                    info[name] = did_flip
                else:
                    sequence, params = aug(sequence)
                    info[name] = params

        return sequence, info

    def __repr__(self) -> str:
        lines = ["RoboticAugmentationPipeline:"]
        for name, aug, prob in self.augmentations:
            lines.append(f"  {name}: p={prob}")
        return "\n".join(lines)
