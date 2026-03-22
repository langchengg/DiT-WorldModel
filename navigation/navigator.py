"""
MPC-Based World Model Navigator.

核心链路:
    当前画面 → World Model 想象多条未来轨迹
    → 评估每条轨迹 (碰撞? 接近目标?)
    → 选最优轨迹的第一个动作
    → 执行 → 拿到新画面 → 重复

支持两种规划策略:
1. Random Shooting: 随机采样 N 条候选轨迹, 选最好的
2. CEM (Cross-Entropy Method): 迭代优化候选动作分布

Reference:
    - Model Predictive Control (MPC) for visual navigation
    - CEM: "The cross-entropy method for combinatorial optimization" (De Boer et al., 2005)
"""

import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion import DiffusionProcess, DDIMSampler


class WorldModelNavigator:
    """
    用训练好的 World Model + MPC 做视觉导航.

    核心思路: 想象 → 评估 → 执行.

    MPC 循环:
        1. 采样 N 条候选动作序列
        2. 对每条序列, 用 World Model 想象未来 H 步
        3. 评估每条轨迹的得分 (碰撞扣分 + 接近目标加分)
        4. 选得分最高的轨迹, 执行第一个动作
        5. 拿到真实观测, 回到步骤 1

    Args:
        world_model:      Trained DiTWorldModel.
        diffusion:        DiffusionProcess instance.
        sampler:          DDIMSampler for fast inference.
        action_dim:       Number of discrete actions (default 4 for navigation).
        planning_horizon: How many steps to imagine ahead.
        num_candidates:   Number of candidate trajectories to evaluate.
        device:           Computation device.
        collision_detector: Optional collision detector.
        feature_extractor:  Optional visual feature extractor for goal similarity.
    """

    def __init__(
        self,
        world_model: nn.Module,
        diffusion: DiffusionProcess,
        sampler: DDIMSampler,
        action_dim: int = 4,
        planning_horizon: int = 10,
        num_candidates: int = 64,
        device: str = "cpu",
        collision_detector=None,
        feature_extractor=None,
    ):
        self.world_model = world_model
        self.diffusion = diffusion
        self.sampler = sampler
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.device = device
        self.collision_detector = collision_detector
        self.feature_extractor = feature_extractor

        self.world_model.eval()

    # ------------------------------------------------------------------
    # Core: Imagine a trajectory
    # ------------------------------------------------------------------

    @torch.no_grad()
    def imagine_trajectory(
        self,
        current_obs: torch.Tensor,
        action_sequence: List[int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        给定当前画面和动作序列, 用 World Model 想象未来.

        Args:
            current_obs:    (3, H, W) current camera frame.
            action_sequence: list of action ints [a_0, a_1, ..., a_{H-1}].

        Returns:
            imagined_frames:  List of (3, H, W) predicted frames.
            reward_logits:    List of (num_reward_classes,) reward logits.
            done_logits:      List of (2,) done logits.
        """
        imagined_frames = []
        reward_logits_list = []
        done_logits_list = []

        obs = current_obs.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        C, H, W = current_obs.shape

        for action in action_sequence:
            action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)

            # Use DDIM sampling to predict next frame
            shape = (1, C, H, W)
            next_obs, reward_logits, done_logits = self.sampler.sample(
                self.world_model, shape, obs, action_tensor, device=self.device,
            )

            imagined_frames.append(next_obs.squeeze(0).cpu())
            reward_logits_list.append(reward_logits.squeeze(0).cpu())
            done_logits_list.append(done_logits.squeeze(0).cpu())

            # Autoregressive: use predicted frame as next input
            obs = next_obs

        return imagined_frames, reward_logits_list, done_logits_list

    @torch.no_grad()
    def imagine_trajectory_batch(
        self,
        current_obs: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """
        批量想象多条轨迹 (GPU 并行, 比逐条快得多).

        Args:
            current_obs:     (3, H, W) current observation.
            action_sequences: (N, H) tensor of N action sequences, each length H.

        Returns:
            all_frames:  List[List[Tensor]] — imagined frames per trajectory.
            all_scores:  (N,) preliminary scores from done logits.
        """
        N, horizon = action_sequences.shape
        C, H, W = current_obs.shape

        # Replicate current obs for all candidates
        obs = current_obs.unsqueeze(0).expand(N, -1, -1, -1).to(self.device)
        # (N, C, H, W)

        all_frames = [[] for _ in range(N)]
        cumulative_done_prob = torch.zeros(N, device=self.device)

        for t in range(horizon):
            actions = action_sequences[:, t].to(self.device)

            shape = (N, C, H, W)
            next_obs, reward_logits, done_logits = self.sampler.sample(
                self.world_model, shape, obs, actions, device=self.device,
            )

            # Track done probability
            done_probs = F.softmax(done_logits, dim=-1)[:, 1]
            cumulative_done_prob += done_probs

            for i in range(N):
                all_frames[i].append(next_obs[i].cpu())

            obs = next_obs

        return all_frames, cumulative_done_prob

    # ------------------------------------------------------------------
    # Core: Plan an action
    # ------------------------------------------------------------------

    def plan_action(
        self,
        current_obs: torch.Tensor,
        goal_image: Optional[torch.Tensor] = None,
        method: str = "random_shooting",
    ) -> Tuple[int, List[torch.Tensor], float]:
        """
        Plan the best next action using MPC.

        Args:
            current_obs: (3, H, W) current observation.
            goal_image:  (3, H, W) optional goal observation.
            method:      Planning method: "random_shooting" or "cem".

        Returns:
            best_action:     int — the best action to take NOW.
            best_trajectory: List of imagined frames for the best trajectory.
            best_score:      float — score of the best trajectory.
        """
        if method == "cem":
            return self._plan_cem(current_obs, goal_image)
        else:
            return self._plan_random_shooting(current_obs, goal_image)

    def _plan_random_shooting(
        self,
        current_obs: torch.Tensor,
        goal_image: Optional[torch.Tensor] = None,
    ) -> Tuple[int, List[torch.Tensor], float]:
        """
        Random Shooting MPC: sample N trajectories, pick the best.
        """
        best_score = -float("inf")
        best_action_seq = None
        best_trajectory = None

        candidates = self._sample_action_sequences(
            self.num_candidates, self.planning_horizon,
        )

        for action_seq in candidates:
            frames, reward_logits, done_logits = self.imagine_trajectory(
                current_obs, action_seq,
            )

            score = self._evaluate_trajectory(
                frames, reward_logits, done_logits, goal_image,
            )

            if score > best_score:
                best_score = score
                best_action_seq = action_seq
                best_trajectory = frames

        return best_action_seq[0], best_trajectory, best_score

    def _plan_cem(
        self,
        current_obs: torch.Tensor,
        goal_image: Optional[torch.Tensor] = None,
        num_iterations: int = 3,
        elite_ratio: float = 0.2,
    ) -> Tuple[int, List[torch.Tensor], float]:
        """
        Cross-Entropy Method (CEM) MPC: iteratively refine candidate distribution.

        CEM 比 Random Shooting 更高效:
        - 第1轮: 均匀采样
        - 第2轮: 从 elite 候选中重新采样 (偏向好的动作)
        - 第3轮: 进一步精炼
        """
        H = self.planning_horizon
        N = self.num_candidates
        K = max(1, int(N * elite_ratio))

        # Initialize: uniform action distribution
        # For discrete actions, maintain per-step categorical distributions
        action_probs = torch.ones(H, self.action_dim) / self.action_dim

        best_overall_score = -float("inf")
        best_overall_seq = None
        best_overall_traj = None

        for iteration in range(num_iterations):
            # Sample N sequences from current distribution
            candidates = []
            for _ in range(N):
                seq = []
                for t in range(H):
                    a = torch.multinomial(action_probs[t], 1).item()
                    seq.append(a)
                candidates.append(seq)

            # Evaluate all candidates
            scored = []
            for action_seq in candidates:
                frames, rew_logits, done_logits = self.imagine_trajectory(
                    current_obs, action_seq,
                )
                score = self._evaluate_trajectory(
                    frames, rew_logits, done_logits, goal_image,
                )
                scored.append((score, action_seq, frames))

            # Sort and select elite
            scored.sort(key=lambda x: x[0], reverse=True)
            elites = scored[:K]

            if elites[0][0] > best_overall_score:
                best_overall_score = elites[0][0]
                best_overall_seq = elites[0][1]
                best_overall_traj = elites[0][2]

            # Update distribution from elites
            action_counts = torch.zeros(H, self.action_dim)
            for _, seq, _ in elites:
                for t, a in enumerate(seq):
                    action_counts[t, a] += 1.0
            action_probs = (action_counts + 0.1)  # Laplace smoothing
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        return best_overall_seq[0], best_overall_traj, best_overall_score

    # ------------------------------------------------------------------
    # Trajectory evaluation
    # ------------------------------------------------------------------

    def _evaluate_trajectory(
        self,
        imagined_frames: List[torch.Tensor],
        reward_logits: List[torch.Tensor],
        done_logits: List[torch.Tensor],
        goal_image: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Evaluate a single imagined trajectory.

        Scoring:
            + reward signal from world model
            + goal similarity bonus (if goal_image provided)
            - collision penalty (from done prediction)
            - step cost (encourage shorter paths)

        Args:
            imagined_frames: List of (3, H, W) predicted frames.
            reward_logits:   List of (num_classes,) reward logits.
            done_logits:     List of (2,) done logits.
            goal_image:      Optional (3, H, W) goal observation.

        Returns:
            float score (higher is better).
        """
        score = 0.0

        for t, (frame, rew_log, done_log) in enumerate(
            zip(imagined_frames, reward_logits, done_logits)
        ):
            # --- Collision penalty ---
            done_prob = F.softmax(done_log, dim=-1)[1].item()

            if self.collision_detector is not None:
                coll_prob = self.collision_detector.predict(
                    done_log.unsqueeze(0),
                ).item()
            else:
                coll_prob = done_prob

            if coll_prob > 0.8:
                score -= 100.0
                break  # Stop evaluating after collision
            score -= coll_prob * 10.0

            # --- Reward signal ---
            reward_class = torch.argmax(rew_log).item()
            reward_val = reward_class - 1.0  # {0,1,2} → {-1,0,1}
            score += reward_val * 5.0

            # --- Goal similarity ---
            if goal_image is not None:
                sim = self._compute_similarity(frame, goal_image)
                # Later frames closer to goal get higher weight
                score += sim * 20.0 * (1.0 + t * 0.1)

            # --- Step cost ---
            score -= 1.0

        return score

    def _compute_similarity(
        self,
        frame: torch.Tensor,
        goal_image: torch.Tensor,
    ) -> float:
        """
        Compute visual similarity between predicted frame and goal.

        Uses either a feature extractor (if provided) or simple
        structural similarity proxy.

        Args:
            frame:      (3, H, W) predicted frame.
            goal_image: (3, H, W) goal image.

        Returns:
            float similarity in [0, 1].
        """
        if self.feature_extractor is not None:
            feat1 = self.feature_extractor(frame.unsqueeze(0))
            feat2 = self.feature_extractor(goal_image.unsqueeze(0))
            sim = F.cosine_similarity(feat1, feat2, dim=-1)
            return sim.item()
        else:
            # Simple pixel-level similarity (1 - normalized L2 distance)
            diff = (frame - goal_image).pow(2).mean().sqrt()
            return max(0.0, 1.0 - diff.item())

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action_sequences(
        self, num_candidates: int, horizon: int,
    ) -> List[List[int]]:
        """
        Sample random candidate action sequences.

        Strategies:
        - Pure random
        - Momentum-biased (prefer same action for consecutive steps)
        """
        sequences = []
        for i in range(num_candidates):
            if i < num_candidates // 2:
                # Pure random
                seq = [random.randint(0, self.action_dim - 1) for _ in range(horizon)]
            else:
                # Momentum-biased: 70% chance to repeat last action
                seq = [random.randint(0, self.action_dim - 1)]
                for _ in range(horizon - 1):
                    if random.random() < 0.7:
                        seq.append(seq[-1])
                    else:
                        seq.append(random.randint(0, self.action_dim - 1))
            sequences.append(seq)
        return sequences


# ---------------------------------------------------------------------------
# Full Navigation Loop
# ---------------------------------------------------------------------------

def navigate(
    env,
    world_model: nn.Module,
    diffusion: DiffusionProcess,
    sampler: DDIMSampler,
    goal_image: Optional[torch.Tensor] = None,
    max_steps: int = 200,
    planning_horizon: int = 10,
    num_candidates: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete navigation loop using World Model + MPC.

    Args:
        env:              Navigation environment (GridNavigationEnv or real robot).
        world_model:      Trained DiT World Model.
        diffusion:        DiffusionProcess.
        sampler:          DDIMSampler.
        goal_image:       (3, H, W) target observation.
        max_steps:        Maximum navigation steps.
        planning_horizon: MPC look-ahead steps.
        num_candidates:   Number of candidate trajectories.
        device:           Computation device.
        verbose:          Print progress.

    Returns:
        dict with navigation results:
            trajectory:    List of observations.
            actions:       List of actions taken.
            rewards:       List of rewards.
            success:       bool.
            total_steps:   int.
            imaginations:  List of imagined trajectories (for viz).
    """
    from .sim_env import ACTION_NAMES

    navigator = WorldModelNavigator(
        world_model=world_model,
        diffusion=diffusion,
        sampler=sampler,
        action_dim=4,
        planning_horizon=planning_horizon,
        num_candidates=num_candidates,
        device=device,
    )

    obs = env.reset()
    if goal_image is None:
        goal_image = env.get_goal_image() if hasattr(env, "get_goal_image") else None

    trajectory = [obs]
    actions_taken = []
    rewards_collected = []
    imaginations = []

    for step in range(max_steps):
        if verbose and step % 10 == 0:
            print(f"Step {step}: Planning ...")

        # Plan best action
        best_action, imagined_future, score = navigator.plan_action(
            current_obs=obs,
            goal_image=goal_image,
            method="random_shooting",
        )

        if verbose and step % 10 == 0:
            action_str = ACTION_NAMES[best_action] if best_action < len(ACTION_NAMES) else str(best_action)
            print(f"  → Action: {action_str}, Score: {score:.2f}")

        imaginations.append(imagined_future)

        # Execute
        obs, reward, done, info = env.step(best_action)
        trajectory.append(obs)
        actions_taken.append(best_action)
        rewards_collected.append(reward)

        # Check goal
        success = info.get("reached_goal", False) if isinstance(info, dict) else False
        if success:
            if verbose:
                print(f"✅ Reached goal in {step + 1} steps!")
            break

        if done:
            if verbose:
                print(f"❌ Episode ended at step {step + 1}")
            break

    return {
        "trajectory": trajectory,
        "actions": actions_taken,
        "rewards": rewards_collected,
        "success": success if "success" in dir() else False,
        "total_steps": len(actions_taken),
        "imaginations": imaginations,
    }
