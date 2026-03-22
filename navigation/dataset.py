"""
Navigation Dataset Loaders — 公开机器人导航数据集.

支持三种数据来源:
1. SyntheticNavigationDataset  — 用 GridNavigationEnv 生成 (无需下载)
2. RECONDataset                — RECON 室外导航数据集 (HDF5)
3. TartanDriveDataset          — TartanDrive 越野导航数据集

所有 loader 输出统一的 dict:
    {obs_history, obs_next, action, reward, done}
与 training.trainer.WorldModelDataset 格式完全一致.

Reference:
    - RECON: "Rapid Exploration via Contrastive Navigation" (Shah et al., RA-L 2021)
    - TartanDrive: "TartanDrive: A Large-Scale Dataset for Learning Off-Road Driving" (Triest et al., 2022)
"""

import os
import random
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# 1. Synthetic Navigation Dataset (zero-download, uses GridNavigationEnv)
# ---------------------------------------------------------------------------

class SyntheticNavigationDataset(Dataset):
    """
    使用 GridNavigationEnv 自动生成导航训练数据.

    特点:
    - 无需下载任何真实数据
    - 可以控制数据量和复杂度
    - 用于验证整个 World Model + MPC pipeline 是否能跑通

    Args:
        num_episodes:    Number of random-policy episodes to collect.
        episode_length:  Maximum steps per episode.
        img_size:        Rendered image size.
        grid_size:       Grid world size.
        obs_history_len: Number of history frames for conditioning.
        seed:            Random seed.
    """

    def __init__(
        self,
        num_episodes: int = 50,
        episode_length: int = 100,
        img_size: int = 64,
        grid_size: int = 8,
        obs_history_len: int = 1,
        seed: int = 42,
    ):
        from .sim_env import GridNavigationEnv

        self.obs_history_len = obs_history_len
        self.observations: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.episode_starts: List[int] = []

        env = GridNavigationEnv(
            grid_size=grid_size, img_size=img_size, max_steps=episode_length, seed=seed,
        )

        print(f"📦 Generating synthetic navigation data: "
              f"{num_episodes} episodes × {episode_length} max steps ...")

        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            self.episode_starts.append(len(self.observations))
            self.observations.append(obs)

            for step in range(episode_length):
                action = env.sample_action()
                next_obs, reward, done, info = env.step(action)

                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.observations.append(next_obs)

                if done:
                    break

        # Mark episode boundaries to avoid sampling across them
        self.episode_starts.append(len(self.observations))
        self._build_valid_indices()

        print(f"  ✅ Generated {len(self.observations)} frames, "
              f"{len(self._valid_indices)} valid transitions")

    def _build_valid_indices(self) -> None:
        """Build list of valid transition indices (not crossing episode boundary)."""
        self._valid_indices = []
        for i in range(len(self.actions)):
            # Check this transition doesn't cross an episode start
            # obs[i] → action[i] → obs[i+1]
            cross = False
            for es in self.episode_starts:
                if es == i + 1:
                    cross = True
                    break
            if not cross:
                self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._valid_indices[idx]

        # Build observation history
        history_frames = []
        for h in range(self.obs_history_len):
            frame_idx = max(0, real_idx - self.obs_history_len + 1 + h)
            history_frames.append(self.observations[frame_idx])
        obs_history = torch.stack(history_frames, dim=0)  # (T, 3, H, W)

        obs_next = self.observations[real_idx + 1]         # (3, H, W)
        action = torch.tensor(self.actions[real_idx], dtype=torch.long)
        reward = self._discretize_reward(self.rewards[real_idx])
        done = int(self.dones[real_idx])

        return {
            "obs_history": obs_history,
            "obs_next": obs_next,
            "action": action,
            "reward": torch.tensor(reward, dtype=torch.long),
            "done": torch.tensor(done, dtype=torch.long),
        }

    @staticmethod
    def _discretize_reward(reward: float) -> int:
        """Map reward → {0: negative, 1: zero, 2: positive}."""
        if reward < -0.01:
            return 0
        elif reward > 0.01:
            return 2
        return 1


# ---------------------------------------------------------------------------
# 2. RECON Dataset (Real outdoor navigation)
# ---------------------------------------------------------------------------

class RECONDataset(Dataset):
    """
    RECON 室外导航数据集 loader.

    RECON 数据集包含真实的室外导航轨迹:
    - RGB 图像 (前视摄像头)
    - 连续动作 (线速度, 角速度)
    - GPS 位置信息

    数据格式: HDF5 文件, 每个文件是一条轨迹.

    下载: https://sites.google.com/view/recon-robot/dataset

    目录结构:
        data_dir/
            traj_0000.hdf5
            traj_0001.hdf5
            ...

    每个 HDF5 文件包含:
        /observations/images:  (T, H, W, 3) uint8
        /actions:              (T, 2) float32   [linear_vel, angular_vel]
        /position:             (T, 2) float32   [x, y]

    Args:
        data_dir:        Root directory containing HDF5 trajectory files.
        img_size:        Target image size (will resize).
        obs_history_len: Number of context frames.
        action_bins:     Number of bins for action discretization.
        max_trajectories: Maximum number of trajectories to load.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        img_size: int = 64,
        obs_history_len: int = 1,
        action_bins: int = 256,
        max_trajectories: Optional[int] = None,
        download_mode: str = "full",
        sample_num_files: int = 3,
    ):
        import h5py
        import cv2

        self.img_size = img_size
        self.obs_history_len = obs_history_len
        self.action_bins = action_bins
        self.download_mode = download_mode

        if download_mode not in {"full", "sample"}:
            raise ValueError(
                f"Unsupported RECON download_mode='{download_mode}'. "
                "Choose from: full, sample"
            )

        # Resolve default storage location. Auto-download happens only if the
        # directory does not already contain trajectory files.
        if data_dir is None:
            data_dir = "data/recon" if download_mode == "full" else "data/recon_sample"
            os.makedirs(data_dir, exist_ok=True)

        data_path = Path(data_dir)
        hdf5_files = sorted(data_path.glob("*.hdf5"))
        if max_trajectories:
            hdf5_files = hdf5_files[:max_trajectories]

        if len(hdf5_files) == 0:
            mode_desc = "full RECON dataset" if download_mode == "full" else "RECON sample"
            print(f"⚠️  No HDF5 files found in {data_dir}. Downloading {mode_desc}...")
            self._download_dataset(
                extract_dir=data_dir,
                max_files=None if download_mode == "full" else sample_num_files,
            )
            hdf5_files = sorted(data_path.glob("*.hdf5"))

        if len(hdf5_files) == 0:
            raise RuntimeError("Failed to load or download RECON dataset.")

        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.episode_starts: List[int] = []

        print(f"📦 Loading RECON dataset from {data_dir} ...")

        for f_path in hdf5_files:
            try:
                with h5py.File(str(f_path), "r") as f:
                    images, actions_raw, collision_any = self._read_recon_file(f)

                    self.episode_starts.append(len(self.observations))

                    for t in range(len(images)):
                        img = self._decode_image(images[t])
                        if img is None:
                            continue
                        img = cv2.resize(img, (img_size, img_size))
                        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        self.observations.append(img_t)

                        if t < len(images) - 1:
                            act = actions_raw[t]
                            act_bin = self._discretize_action(act)
                            self.actions.append(act_bin)
                            self.rewards.append(0.0)
                            done_flag = False
                            if collision_any is not None and t + 1 < len(collision_any):
                                done_flag = bool(collision_any[t + 1])
                            self.dones.append(done_flag)

                    # Mark last step as done
                    if len(self.dones) > 0:
                        self.dones[-1] = True
            except Exception as e:
                print(f"  ⚠️  Skipping {f_path.name}: {e}")

        self.episode_starts.append(len(self.observations))
        self._build_valid_indices()

        if len(self._valid_indices) == 0:
            raise RuntimeError(
                "RECON files were found but no valid transitions could be loaded. "
                "This usually means the HDF5 schema does not match the loader or "
                "the extracted sample files are incomplete."
            )

        print(f"  ✅ Loaded {len(hdf5_files)} trajectories, "
              f"{len(self.observations)} frames, "
              f"{len(self._valid_indices)} valid transitions")

    @staticmethod
    def _decode_image(image_item):
        import cv2

        if isinstance(image_item, np.ndarray) and image_item.ndim == 3:
            if image_item.shape[-1] == 3:
                return image_item
            return None

        if isinstance(image_item, np.ndarray):
            raw = image_item.tobytes()
        else:
            raw = bytes(image_item)

        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == 0:
            return None

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _read_recon_file(h5_file):
        """
        Support both the originally assumed schema and the actual RECON sample
        schema where images are JPEG bytes and controls live under `commands/`.
        """
        if "observations" in h5_file and "images" in h5_file["observations"]:
            images = h5_file["observations/images"][:]
            actions_raw = h5_file["actions"][:]
            collision_any = None
            return images, actions_raw, collision_any

        if (
            "images" in h5_file and "rgb_left" in h5_file["images"]
            and "commands" in h5_file
            and "linear_velocity" in h5_file["commands"]
            and "angular_velocity" in h5_file["commands"]
        ):
            images = h5_file["images/rgb_left"][:]
            linear_velocity = h5_file["commands/linear_velocity"][:]
            angular_velocity = h5_file["commands/angular_velocity"][:]
            actions_raw = np.stack([linear_velocity, angular_velocity], axis=-1)
            collision_any = None
            if "collision" in h5_file and "any" in h5_file["collision"]:
                collision_any = h5_file["collision/any"][:]
            return images, actions_raw, collision_any

        available = list(h5_file.keys())
        raise KeyError(
            "Unsupported RECON HDF5 schema. Top-level keys: "
            f"{available}"
        )

    def _download_dataset(
        self,
        extract_dir: str,
        max_files: Optional[int] = None,
    ) -> None:
        """
        Stream the RECON tarball and extract trajectory HDF5 files.

        If `max_files` is None, extract the full dataset. Otherwise extract only
        the first `max_files` trajectories for quick smoke tests.
        """
        import tarfile
        import requests

        url = "http://rail.eecs.berkeley.edu/datasets/recon-navigation/recon_dataset.tar.gz"
        mode_desc = (
            "the full RECON dataset"
            if max_files is None else
            f"{max_files} RECON sample trajectories"
        )
        print(f"🌐 Streaming from {url} to extract {mode_desc}...")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                    extracted_count = 0
                    skipped_existing = 0
                    for member in tar:
                        if member.name.endswith(".hdf5"):
                            filename = os.path.basename(member.name)
                            dest_path = os.path.join(extract_dir, filename)

                            if os.path.exists(dest_path):
                                skipped_existing += 1
                                extracted_count += 1
                            else:
                                print(f"  ⬇️  Extracting {filename}...")
                                f_in = tar.extractfile(member)
                                if f_in is not None:
                                    with open(dest_path, "wb") as f_out:
                                        f_out.write(f_in.read())
                                    extracted_count += 1

                            if max_files is not None and extracted_count >= max_files:
                                print(f"  ✅ Successfully extracted {max_files} sample files.")
                                break

                    if max_files is None:
                        print(
                            f"  ✅ Finished extracting RECON into {extract_dir} "
                            f"({extracted_count - skipped_existing} new files, "
                            f"{skipped_existing} already present)."
                        )

        except Exception as e:
            print(f"  ❌ Failed to download RECON dataset: {e}")

    def _discretize_action(self, action_2d: np.ndarray) -> torch.Tensor:
        """
        Discretize 2D continuous action (linear_vel, angular_vel) to single bin.

        Maps each dim to a bin, then combines as:
            flat_idx = bin_linear * sqrt(action_bins) + bin_angular
        """
        nbins_per_dim = int(self.action_bins ** 0.5)
        bins = []
        for a in action_2d:
            # Assume actions are roughly in [-1, 1]
            a_clamp = np.clip(a, -1.0, 1.0)
            b = int((a_clamp + 1.0) / 2.0 * (nbins_per_dim - 1))
            b = max(0, min(nbins_per_dim - 1, b))
            bins.append(b)
        flat = bins[0] * nbins_per_dim + bins[1]
        return torch.tensor(flat, dtype=torch.long)

    def _build_valid_indices(self) -> None:
        self._valid_indices = []
        ep_set = set(self.episode_starts)
        for i in range(len(self.actions)):
            if (i + 1) not in ep_set:
                self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._valid_indices[idx]

        history_frames = []
        for h in range(self.obs_history_len):
            fi = max(0, real_idx - self.obs_history_len + 1 + h)
            history_frames.append(self.observations[fi])
        obs_history = torch.stack(history_frames, dim=0)

        return {
            "obs_history": obs_history,
            "obs_next": self.observations[real_idx + 1],
            "action": self.actions[real_idx],
            "reward": torch.tensor(
                0 if abs(self.rewards[real_idx]) < 0.01 else
                (2 if self.rewards[real_idx] > 0 else 0),
                dtype=torch.long,
            ),
            "done": torch.tensor(int(self.dones[real_idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3. TartanDrive Dataset (Off-road driving)
# ---------------------------------------------------------------------------

class TartanDriveDataset(Dataset):
    """
    TartanDrive 越野导航数据集 loader.

    TartanDrive 包含多种传感器的越野驾驶数据:
    - 前视 RGB 相机
    - IMU、GPS、轮速等
    - 控制命令 (线速度, 角速度)

    数据下载: https://github.com/castacks/tartandrive

    目录结构:
        data_dir/
            run_0001/
                image_left/
                    000000.png, 000001.png, ...
                cmd.npy          # (T, 2) control commands
            run_0002/
                ...

    Args:
        data_dir:        Root directory.
        img_size:        Target image size.
        obs_history_len: History context length.
        action_bins:     Action discretization bins.
        max_runs:        Maximum number of runs to load.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 64,
        obs_history_len: int = 1,
        action_bins: int = 256,
        max_runs: Optional[int] = None,
    ):
        import cv2

        self.img_size = img_size
        self.obs_history_len = obs_history_len
        self.action_bins = action_bins

        data_path = Path(data_dir)
        run_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        if max_runs:
            run_dirs = run_dirs[:max_runs]

        if len(run_dirs) == 0:
            raise FileNotFoundError(
                f"No run directories found in {data_dir}. "
                f"Download TartanDrive from: "
                f"https://github.com/castacks/tartandrive"
            )

        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.episode_starts: List[int] = []

        print(f"📦 Loading TartanDrive dataset from {data_dir} ...")

        for run_dir in run_dirs:
            try:
                img_dir = run_dir / "image_left"
                cmd_file = run_dir / "cmd.npy"

                if not img_dir.exists() or not cmd_file.exists():
                    continue

                commands = np.load(str(cmd_file))  # (T, 2)
                img_files = sorted(img_dir.glob("*.png"))

                T = min(len(img_files), len(commands))
                if T < 2:
                    continue

                self.episode_starts.append(len(self.observations))

                for t in range(T):
                    img = cv2.imread(str(img_files[t]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    self.observations.append(img_t)

                    if t < T - 1:
                        act_bin = self._discretize_action(commands[t])
                        self.actions.append(act_bin)
                        self.rewards.append(0.0)
                        self.dones.append(False)

                if len(self.dones) > 0:
                    self.dones[-1] = True

            except Exception as e:
                print(f"  ⚠️  Skipping {run_dir.name}: {e}")

        self.episode_starts.append(len(self.observations))
        self._build_valid_indices()

        print(f"  ✅ Loaded {len(run_dirs)} runs, "
              f"{len(self.observations)} frames, "
              f"{len(self._valid_indices)} valid transitions")

    def _discretize_action(self, action_2d: np.ndarray) -> torch.Tensor:
        nbins_per_dim = int(self.action_bins ** 0.5)
        bins = []
        for a in action_2d:
            a_clamp = np.clip(a, -1.0, 1.0)
            b = int((a_clamp + 1.0) / 2.0 * (nbins_per_dim - 1))
            b = max(0, min(nbins_per_dim - 1, b))
            bins.append(b)
        return torch.tensor(bins[0] * nbins_per_dim + bins[1], dtype=torch.long)

    def _build_valid_indices(self) -> None:
        self._valid_indices = []
        ep_set = set(self.episode_starts)
        for i in range(len(self.actions)):
            if (i + 1) not in ep_set:
                self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._valid_indices[idx]

        history_frames = []
        for h in range(self.obs_history_len):
            fi = max(0, real_idx - self.obs_history_len + 1 + h)
            history_frames.append(self.observations[fi])
        obs_history = torch.stack(history_frames, dim=0)

        return {
            "obs_history": obs_history,
            "obs_next": self.observations[real_idx + 1],
            "action": self.actions[real_idx],
            "reward": torch.tensor(1, dtype=torch.long),   # neutral
            "done": torch.tensor(int(self.dones[real_idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Utility: Create dataset from config
# ---------------------------------------------------------------------------

def get_navigation_action_dim(
    dataset_type: str = "synthetic",
    action_bins: int = 256,
) -> int:
    """
    Resolve the discrete action vocabulary size for each navigation dataset.

    Synthetic grid navigation uses 4 fixed actions.
    RECON and TartanDrive discretize 2D continuous controls into `action_bins`.
    """
    if dataset_type == "synthetic":
        return 4
    return action_bins


def create_navigation_dataset(
    dataset_type: str = "synthetic",
    data_dir: Optional[str] = None,
    img_size: int = 64,
    obs_history_len: int = 1,
    **kwargs,
) -> Dataset:
    """
    Factory function to create navigation datasets.

    Args:
        dataset_type:    "synthetic", "recon", or "tartan".
        data_dir:        Path to data (required for recon/tartan).
        img_size:        Image resolution.
        obs_history_len: History context length.
        **kwargs:        Extra args passed to dataset constructor.

    Returns:
        Dataset instance.
    """
    if dataset_type == "synthetic":
        synthetic_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"num_episodes", "episode_length", "grid_size", "seed"}
        }
        return SyntheticNavigationDataset(
            img_size=img_size, obs_history_len=obs_history_len, **synthetic_kwargs,
        )
    elif dataset_type == "recon":
        recon_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"action_bins", "max_trajectories", "download_mode", "sample_num_files"}
        }
        return RECONDataset(
            data_dir=data_dir, img_size=img_size,
            obs_history_len=obs_history_len, **recon_kwargs,
        )
    elif dataset_type == "tartan":
        if data_dir is None:
            raise ValueError("data_dir required for TartanDrive dataset")
        tartan_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"action_bins", "max_runs"}
        }
        return TartanDriveDataset(
            data_dir=data_dir, img_size=img_size,
            obs_history_len=obs_history_len, **tartan_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                         f"Choose from: synthetic, recon, tartan")


def get_num_trajectories(dataset: Dataset) -> int:
    """
    Return the number of trajectories/episodes if the dataset tracks them.
    """
    episode_starts = getattr(dataset, "episode_starts", None)
    if episode_starts is None:
        return 0
    return max(0, len(episode_starts) - 1)


def split_navigation_dataset_by_trajectory(
    dataset: Dataset,
    eval_ratio: float = 0.1,
    min_eval_trajectories: int = 1,
    seed: int = 42,
) -> Tuple[Subset, Subset, dict]:
    """
    Split a navigation dataset by full trajectories instead of random transitions.

    This avoids leakage where nearly identical adjacent frames from the same
    trajectory appear in both training and evaluation.
    """
    episode_starts = getattr(dataset, "episode_starts", None)
    valid_indices = getattr(dataset, "_valid_indices", None)
    if episode_starts is None or valid_indices is None:
        raise ValueError(
            "Dataset does not expose trajectory boundaries. "
            "Expected `episode_starts` and `_valid_indices`."
        )

    num_trajectories = max(0, len(episode_starts) - 1)
    if num_trajectories == 0:
        raise ValueError("Dataset has no trajectories to split.")

    if num_trajectories == 1:
        all_indices = list(range(len(dataset)))
        info = {
            "num_trajectories": 1,
            "train_trajectories": 1,
            "eval_trajectories": 0,
            "train_samples": len(all_indices),
            "eval_samples": 0,
        }
        return Subset(dataset, all_indices), Subset(dataset, []), info

    traj_ids = list(range(num_trajectories))
    rng = random.Random(seed)
    rng.shuffle(traj_ids)

    eval_traj_count = max(min_eval_trajectories, int(round(num_trajectories * eval_ratio)))
    eval_traj_count = min(eval_traj_count, num_trajectories - 1)

    eval_traj_ids = set(traj_ids[:eval_traj_count])
    train_indices = []
    eval_indices = []

    for subset_idx, real_idx in enumerate(valid_indices):
        traj_idx = bisect_right(episode_starts, real_idx) - 1
        if traj_idx in eval_traj_ids:
            eval_indices.append(subset_idx)
        else:
            train_indices.append(subset_idx)

    info = {
        "num_trajectories": num_trajectories,
        "train_trajectories": num_trajectories - eval_traj_count,
        "eval_trajectories": eval_traj_count,
        "train_samples": len(train_indices),
        "eval_samples": len(eval_indices),
    }

    return Subset(dataset, train_indices), Subset(dataset, eval_indices), info
