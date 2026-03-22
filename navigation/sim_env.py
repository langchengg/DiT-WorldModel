"""
Grid Navigation Environment — 简易 2D 网格导航仿真环境.

无需任何外部依赖 (Habitat / MuJoCo / etc.)，用于快速验证
World Model + MPC 导航 pipeline 的正确性.

特点:
  - 4 个离散动作: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT
  - 自顶向下 RGB 渲染 (agent + walls + goal)
  - 随机迷宫生成
  - Gym-style API: reset() / step() / render()

用法:
    env = GridNavigationEnv(grid_size=8, img_size=64)
    obs = env.reset()
    obs, reward, done, info = env.step(action=0)  # FORWARD
"""

import math
import random
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch


# Actions
FORWARD = 0
BACKWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

ACTION_NAMES = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"]

# Directions: 0=North, 1=East, 2=South, 3=West
DX = [-1, 0, 1, 0]  # row delta
DY = [0, 1, 0, -1]  # col delta


class GridNavigationEnv:
    """
    2D Grid Navigation Environment with top-down RGB rendering.

    Grid cells:
        0 = free space (white)
        1 = wall (dark gray)
        2 = goal (green)

    Agent state: (row, col, direction)
    direction ∈ {0=N, 1=E, 2=S, 3=W}

    Args:
        grid_size:   Size of the square grid.
        img_size:    Rendered image resolution (square).
        max_steps:   Maximum episode steps.
        wall_ratio:  Fraction of random walls.
        seed:        Random seed for maze generation.
    """

    def __init__(
        self,
        grid_size: int = 8,
        img_size: int = 64,
        max_steps: int = 200,
        wall_ratio: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.img_size = img_size
        self.max_steps = max_steps
        self.wall_ratio = wall_ratio
        self.action_space_n = 4

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # State
        self.grid: Optional[np.ndarray] = None
        self.agent_row = 0
        self.agent_col = 0
        self.agent_dir = 0  # 0=N, 1=E, 2=S, 3=W
        self.goal_row = 0
        self.goal_col = 0
        self.step_count = 0

        # Pre-compute cell size for rendering
        self.cell_px = img_size // grid_size

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Reset environment and return initial observation.

        Returns:
            (3, img_size, img_size) float tensor in [0, 1].
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

        self._generate_maze()
        self.step_count = 0

        return self._render_obs()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take one step.

        Args:
            action: 0=FORWARD, 1=BACKWARD, 2=TURN_LEFT, 3=TURN_RIGHT.

        Returns:
            obs:    (3, H, W) next observation.
            reward: float.
            done:   bool.
            info:   dict with extra info.
        """
        assert 0 <= action < 4, f"Invalid action {action}"
        self.step_count += 1

        collision = False

        if action == FORWARD:
            collision = self._try_move(self.agent_dir)
        elif action == BACKWARD:
            collision = self._try_move((self.agent_dir + 2) % 4)
        elif action == TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Reward
        reached_goal = (self.agent_row == self.goal_row and
                        self.agent_col == self.goal_col)
        timeout = self.step_count >= self.max_steps

        if reached_goal:
            reward = 10.0
        elif collision:
            reward = -1.0
        else:
            # Small penalty per step + distance-based shaping
            dist = abs(self.agent_row - self.goal_row) + abs(self.agent_col - self.goal_col)
            reward = -0.1 - 0.01 * dist

        done = reached_goal or timeout

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "timeout": timeout,
            "agent_pos": (self.agent_row, self.agent_col),
            "agent_dir": self.agent_dir,
            "step": self.step_count,
        }

        return self._render_obs(), reward, done, info

    def sample_action(self) -> int:
        """Sample a random action."""
        return self.rng.randint(0, 3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_maze(self) -> None:
        """Generate a random grid with walls, free space, and a goal."""
        g = self.grid_size
        self.grid = np.zeros((g, g), dtype=np.int32)

        # Border walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Random interior walls
        num_walls = int(self.wall_ratio * (g - 2) ** 2)
        interior = [(r, c) for r in range(1, g - 1) for c in range(1, g - 1)]
        self.rng.shuffle(interior)

        for r, c in interior[:num_walls]:
            self.grid[r, c] = 1

        # Place agent at a random free cell
        free_cells = [(r, c) for r in range(1, g - 1)
                      for c in range(1, g - 1)
                      if self.grid[r, c] == 0]
        self.rng.shuffle(free_cells)

        self.agent_row, self.agent_col = free_cells[0]
        self.agent_dir = self.rng.randint(0, 3)

        # Place goal at a different free cell (prefer far from agent)
        free_cells.sort(
            key=lambda rc: -(abs(rc[0] - self.agent_row) + abs(rc[1] - self.agent_col))
        )
        self.goal_row, self.goal_col = free_cells[0]
        self.grid[self.goal_row, self.goal_col] = 2

    def _try_move(self, direction: int) -> bool:
        """
        Try to move agent in given direction.
        Returns True if collision occurred.
        """
        nr = self.agent_row + DX[direction]
        nc = self.agent_col + DY[direction]

        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            if self.grid[nr, nc] != 1:
                self.agent_row = nr
                self.agent_col = nc
                return False
        return True  # collision

    def _render_obs(self) -> torch.Tensor:
        """
        Render a top-down RGB image of the grid.

        Colors:
            Free space: white (1.0, 1.0, 1.0)
            Wall:       dark gray (0.2, 0.2, 0.2)
            Goal:       green (0.2, 0.8, 0.2)
            Agent body: blue (0.2, 0.4, 0.9)
            Agent nose: red (0.9, 0.2, 0.2) — indicates facing direction

        Returns:
            (3, img_size, img_size) float tensor in [0, 1].
        """
        g = self.grid_size
        cell = self.cell_px
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)

        for r in range(g):
            for c in range(g):
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                if self.grid[r, c] == 1:
                    img[y0:y1, x0:x1] = [0.2, 0.2, 0.2]  # wall
                elif self.grid[r, c] == 2:
                    img[y0:y1, x0:x1] = [0.2, 0.8, 0.2]  # goal

        # Draw agent
        ay0 = self.agent_row * cell
        ax0 = self.agent_col * cell
        pad = max(1, cell // 6)

        # Agent body (blue)
        img[ay0 + pad: ay0 + cell - pad,
            ax0 + pad: ax0 + cell - pad] = [0.2, 0.4, 0.9]

        # Agent nose (red triangle approximation)
        cy, cx = ay0 + cell // 2, ax0 + cell // 2
        nose_len = cell // 3
        ny = cy + DX[self.agent_dir] * nose_len
        nx = cx + DY[self.agent_dir] * nose_len
        # Draw a small red dot at the nose
        ns = max(1, cell // 8)
        ny0, ny1 = max(0, ny - ns), min(self.img_size, ny + ns + 1)
        nx0, nx1 = max(0, nx - ns), min(self.img_size, nx + ns + 1)
        img[ny0:ny1, nx0:nx1] = [0.9, 0.2, 0.2]

        # Add thin grid lines
        for i in range(g + 1):
            pos = min(i * cell, self.img_size - 1)
            img[pos, :] = [0.7, 0.7, 0.7]
            img[:, pos] = [0.7, 0.7, 0.7]

        # (H, W, 3) → (3, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return tensor

    def get_goal_image(self) -> torch.Tensor:
        """
        Render an image showing the goal location (no agent).
        Useful as the target for visual-similarity navigation.
        """
        saved = (self.agent_row, self.agent_col, self.agent_dir)
        self.agent_row, self.agent_col = self.goal_row, self.goal_col
        self.agent_dir = 0
        goal_img = self._render_obs()
        self.agent_row, self.agent_col, self.agent_dir = saved
        return goal_img

    def __repr__(self) -> str:
        return (
            f"GridNavigationEnv(grid_size={self.grid_size}, "
            f"img_size={self.img_size}, max_steps={self.max_steps})"
        )
