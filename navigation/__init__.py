"""
Navigation Module for DiT World Model.

用 World Model + MPC 实现视觉导航:
  观测 → World Model 想象多条轨迹 → 评估 → 选最优动作 → 执行 → 循环

Components:
    - dataset:        公开机器人导航数据集加载器 (RECON, TartanDrive, Synthetic)
    - navigator:      MPC-based WorldModelNavigator
    - collision:      碰撞检测 (CNN / Depth / Done-signal)
    - sim_env:        简易 Grid Navigation 仿真环境
    - visualize_nav:  导航可视化工具
"""

from .navigator import WorldModelNavigator
from .sim_env import GridNavigationEnv
from .collision import DoneSignalCollisionDetector, CNNCollisionDetector

__all__ = [
    "WorldModelNavigator",
    "GridNavigationEnv",
    "DoneSignalCollisionDetector",
    "CNNCollisionDetector",
]
