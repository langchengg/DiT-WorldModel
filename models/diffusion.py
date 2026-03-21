"""
Diffusion Process for DiT World Model.

实现完整的 DDPM/DDIM 扩散过程:
- 前向过程 (forward process): 逐步加噪 x_0 → x_T
- 反向过程 (reverse process): 逐步去噪 x_T → x_0
- 支持多种噪声调度: linear, cosine, sigmoid

设计考量:
- 使用 v-prediction 参数化 (与 DIAMOND 一致)
- 支持 classifier-free guidance (CFG)
- DDIM 采样器支持可变步数加速推理

Reference:
    - DDPM: "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
    - DDIM: "Denoising Diffusion Implicit Models" (Song et al., ICLR 2021)
    - Improved DDPM: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, ICML 2021)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise Schedules
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """
    线性噪声调度: β_t 从 beta_start 线性增长到 beta_end.
    
    这是最经典的 DDPM 调度方式, 简单但在大 T 时效果不如 cosine.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


class CosineNoiseSchedule:
    """
    余弦噪声调度 (Improved DDPM).
    
    优势: 在 t 接近 0 和 T 时变化更平滑, 避免信息损失过快.
    公式: ᾱ_t = f(t) / f(0), where f(t) = cos²((t/T + s) / (1+s) · π/2)
    
    Args:
        num_timesteps: Total number of diffusion steps T.
        s:             Offset to prevent β_t becoming too small near t=0.
        max_beta:      Maximum value of β_t (clamp for stability).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
        max_beta: float = 0.999,
    ):
        self.num_timesteps = num_timesteps

        steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
        f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]

        # Compute betas from alpha_bar
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0, max_beta)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


class SigmoidNoiseSchedule:
    """
    Sigmoid 噪声调度.
    
    使用 sigmoid 函数将 β_t 从 beta_start 平滑过渡到 beta_end,
    在中间时间步变化最快, 两端变化较慢.
    适合高分辨率图像生成.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        betas = torch.sigmoid(torch.linspace(-6, 6, num_timesteps))
        betas = betas * (beta_end - beta_start) + beta_start

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


# ---------------------------------------------------------------------------
# Diffusion Process
# ---------------------------------------------------------------------------

class DiffusionProcess:
    """
    Complete diffusion process: forward (add noise) + reverse (remove noise).
    
    支持两种预测目标:
    - ε-prediction: 预测噪声 ε (经典 DDPM)
    - v-prediction: 预测 v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0 (稳定性更好)
    
    Args:
        num_timesteps:    Total diffusion steps T.
        schedule_type:    Noise schedule type: 'cosine', 'linear', 'sigmoid'.
        prediction_type:  What the model predicts: 'epsilon' or 'v_prediction'.
        loss_type:        Loss function: 'mse' or 'huber'.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
    ):
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type

        # Build noise schedule
        if schedule_type == "cosine":
            schedule = CosineNoiseSchedule(num_timesteps)
        elif schedule_type == "linear":
            schedule = LinearNoiseSchedule(num_timesteps)
        elif schedule_type == "sigmoid":
            schedule = SigmoidNoiseSchedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = schedule.betas
        self.alphas = schedule.alphas
        self.alphas_cumprod = schedule.alphas_cumprod
        self.sqrt_alphas_cumprod = schedule.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod

        # Precompute for posterior q(x_{t-1} | x_t, x_0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, schedule_values: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract schedule values at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = schedule_values.to(t.device).gather(0, t)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def forward_process(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I).
        
        Args:
            x_0:   (B, C, H, W) clean images.
            t:     (B,) timestep indices.
            noise: (B, C, H, W) optional pre-sampled noise.
        
        Returns:
            x_t:   (B, C, H, W) noised images at timestep t.
            noise: (B, C, H, W) the added noise (for loss computation).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape,
        )

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from noise prediction: x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape,
        )
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def predict_x0_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from v-prediction: x_0 = √ᾱ_t · x_t - √(1-ᾱ_t) · v."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape,
        )
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def get_v_target(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute v-prediction target: v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape,
        )
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        reward_pred: Optional[torch.Tensor] = None,
        reward_target: Optional[torch.Tensor] = None,
        done_pred: Optional[torch.Tensor] = None,
        done_target: Optional[torch.Tensor] = None,
        reward_weight: float = 1.0,
        done_weight: float = 1.0,
    ) -> dict:
        """
        Compute combined training loss.
        
        Args:
            model_output: (B, C, H, W) predicted noise or v.
            target:       (B, C, H, W) ground truth noise or v.
            reward_pred:  (B, num_classes) reward logits.
            reward_target: (B,) reward class labels.
            done_pred:    (B, 2) done logits.
            done_target:  (B,) done labels {0, 1}.
            reward_weight: Weight for reward loss.
            done_weight:  Weight for done loss.
        
        Returns:
            dict with 'total', 'diffusion', 'reward', 'done' losses.
        """
        # Diffusion loss
        if self.loss_type == "mse":
            diff_loss = F.mse_loss(model_output, target)
        elif self.loss_type == "huber":
            diff_loss = F.smooth_l1_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        total_loss = diff_loss
        losses = {"diffusion": diff_loss}

        # Reward loss
        if reward_pred is not None and reward_target is not None:
            rew_loss = F.cross_entropy(reward_pred, reward_target.long())
            total_loss = total_loss + reward_weight * rew_loss
            losses["reward"] = rew_loss

        # Done loss
        if done_pred is not None and done_target is not None:
            done_loss = F.cross_entropy(done_pred, done_target.long())
            total_loss = total_loss + done_weight * done_loss
            losses["done"] = done_loss

        losses["total"] = total_loss
        return losses

    def q_posterior_mean_variance(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Returns:
            mean:         Posterior mean.
            variance:     Posterior variance.
            log_variance: Log posterior variance.
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_0 + coef2 * x_t
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)
        return mean, variance, log_variance


# ---------------------------------------------------------------------------
# DDIM Sampler
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    DDIM deterministic/stochastic sampler for fast inference.
    
    优势: 可以用远少于 T 的步数 (e.g. 20-50 步) 生成高质量样本.
    eta=0 → 完全确定性; eta=1 → 等价于 DDPM.
    
    Args:
        diffusion:      DiffusionProcess instance.
        num_steps:      Number of denoising steps (can be << T).
        eta:            Stochasticity parameter [0, 1].
        clip_denoised: Whether to clip x_0 prediction to [-1, 1].
    """

    def __init__(
        self,
        diffusion: DiffusionProcess,
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ):
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.eta = eta
        self.clip_denoised = clip_denoised

        # Build subsequence of timesteps
        total_T = diffusion.num_timesteps
        self.timesteps = torch.linspace(
            total_T - 1, 0, num_steps, dtype=torch.long,
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        obs_history: torch.Tensor,
        action: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.
        
        Args:
            model:        DiTWorldModel instance.
            shape:        (B, C, H, W) output shape.
            obs_history:  (B, C_hist, H, W) observation context.
            action:       (B,) action indices.
            device:       Target device.
            return_intermediates: If True, return all intermediate x_t.
        
        Returns:
            x_0: (B, C, H, W) generated frames.
            (optional) intermediates: list of (B, C, H, W) at each step.
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)  # x_T ~ N(0, I)
        intermediates = [x] if return_intermediates else None

        alphas_cumprod = self.diffusion.alphas_cumprod.to(device)

        for i in range(len(self.timesteps)):
            t_cur = self.timesteps[i]
            t_batch = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

            # Model prediction
            model_out, reward_pred, done_pred = model(x, t_batch, obs_history, action)

            # Get predicted x_0
            if self.diffusion.prediction_type == "epsilon":
                x_0_pred = self.diffusion.predict_x0_from_eps(x, t_batch, model_out)
            else:  # v_prediction
                x_0_pred = self.diffusion.predict_x0_from_v(x, t_batch, model_out)

            if self.clip_denoised:
                x_0_pred = x_0_pred.clamp(-1, 1)

            # Compute x_{t-1}
            if i < len(self.timesteps) - 1:
                t_next = self.timesteps[i + 1]
                alpha_t = alphas_cumprod[t_cur]
                alpha_next = alphas_cumprod[t_next]
            else:
                alpha_t = alphas_cumprod[t_cur]
                alpha_next = torch.tensor(1.0, device=device)

            # DDIM update
            eps_pred = (x - alpha_t.sqrt() * x_0_pred) / (1 - alpha_t).sqrt()

            sigma = self.eta * torch.sqrt(
                (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
            )

            # Predicted direction
            dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred

            # Denoise step
            x = alpha_next.sqrt() * x_0_pred + dir_xt

            if self.eta > 0:
                x = x + sigma * torch.randn_like(x)

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, reward_pred, done_pred, intermediates
        return x, reward_pred, done_pred

    @torch.no_grad()
    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_cur: int,
        t_next: int,
        obs_history: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single DDIM denoising step. Useful for integration with
        external RL loops.
        """
        device = x_t.device
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

        model_out, reward_pred, done_pred = model(x_t, t_batch, obs_history, action)

        if self.diffusion.prediction_type == "epsilon":
            x_0_pred = self.diffusion.predict_x0_from_eps(x_t, t_batch, model_out)
        else:
            x_0_pred = self.diffusion.predict_x0_from_v(x_t, t_batch, model_out)

        if self.clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        alpha_t = alphas_cumprod[t_cur]
        alpha_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        eps_pred = (x_t - alpha_t.sqrt() * x_0_pred) / (1 - alpha_t).sqrt()
        sigma = self.eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
        )
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0)) * eps_pred
        x_next = alpha_next.sqrt() * x_0_pred + dir_xt

        if self.eta > 0:
            x_next = x_next + sigma * torch.randn_like(x_next)

        return x_next, reward_pred, done_pred


# ---------------------------------------------------------------------------
# World Model Environment (for RL in imagination)
# ---------------------------------------------------------------------------

class WorldModelEnv:
    """
    World Model as an Environment for RL agent training.
    
    允许 RL agent 在世界模型的 "想象" 中训练:
    1. 用真实数据初始化观测历史
    2. Agent 选择动作 → World Model 预测下一帧 + reward + done
    3. 完全自回归, 无需与真实环境交互
    
    这是 DIAMOND 的核心 idea: "dreaming" in the world model.
    
    Args:
        model:          DiTWorldModel instance.
        diffusion:      DiffusionProcess instance.
        sampler:        DDIMSampler instance.
        horizon:        Maximum imagination steps.
        num_history:    Number of history frames for conditioning.
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        sampler: DDIMSampler,
        horizon: int = 15,
        num_history: int = 1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.sampler = sampler
        self.horizon = horizon
        self.num_history = num_history

        self.obs_buffer = None
        self.step_count = 0

    def reset(self, initial_obs: torch.Tensor) -> torch.Tensor:
        """
        Reset environment with real observation.
        
        Args:
            initial_obs: (B, C, H, W) initial frame from real environment.
        Returns:
            current observation.
        """
        self.obs_buffer = [initial_obs]
        self.step_count = 0
        return initial_obs

    @torch.no_grad()
    def step(
        self, action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Take a step in imagination.
        
        Args:
            action: (B,) action to take.
        Returns:
            next_obs:  (B, C, H, W) predicted next frame.
            reward:    (B,) predicted reward.
            done:      (B,) predicted episode termination.
            info:      dict with auxiliary info.
        """
        assert self.obs_buffer is not None, "Call reset() first"

        device = self.obs_buffer[-1].device
        B, C, H, W = self.obs_buffer[-1].shape

        # Build history context
        if len(self.obs_buffer) >= self.num_history:
            obs_history = self.obs_buffer[-self.num_history]
        else:
            obs_history = self.obs_buffer[0]

        # Generate next frame via DDIM sampling
        shape = (B, C, H, W)
        next_obs, reward_logits, done_logits = self.sampler.sample(
            self.model, shape, obs_history, action, device=device,
        )

        # Convert logits to predictions
        reward = torch.argmax(reward_logits, dim=-1).float() - 1.0  # {0,1,2} → {-1,0,1}
        done = torch.argmax(done_logits, dim=-1).bool()

        # Update buffer
        self.obs_buffer.append(next_obs)
        self.step_count += 1

        # Check horizon
        if self.step_count >= self.horizon:
            done = torch.ones_like(done)

        info = {
            "step": self.step_count,
            "reward_logits": reward_logits,
            "done_logits": done_logits,
        }

        return next_obs, reward, done, info
