# ---
# DiT-WorldModel: 02 - DiT vs U-Net Ablation Experiments
# Kaggle Notebook for systematic ablation study
# ---

"""
# 🔬 DiT vs U-Net 消融实验

## 实验设计

### 主实验: DiT vs U-Net (相近参数量)
| Model    | Params | Hidden | Depth | Heads |
|----------|--------|--------|-------|-------|
| U-Net-S  | ~45M   | 64×4   | 2×4   | -     |
| DiT-S    | ~22M   | 384    | 12    | 6     |
| DiT-S*   | ~38M   | 512    | 12    | 8     |

### 消融 1: DiT Depth
| Depth | Params | FID↓ | LPIPS↓ | RL Reward↑ |
|-------|--------|------|--------|-----------|
| 4     | ~8M    | ?    | ?      | ?         |
| 8     | ~15M   | ?    | ?      | ?         |
| 12    | ~22M   | ?    | ?      | ?         |
| 16    | ~30M   | ?    | ?      | ?         |

### 消融 2: Patch Size
| Patch | Patches | Params | FID↓ | Speed |
|-------|---------|--------|------|-------|
| 2     | 1024    | ~25M   | ?    | ?     |
| 4     | 256     | ~22M   | ?    | ?     |
| 8     | 64      | ~20M   | ?    | ?     |

### 消融 3: Progressive vs Fixed
| Schedule    | Final FID | Time to Converge |
|-------------|-----------|-----------------|
| Fixed 100   | ?         | ?               |
| Prog Linear | ?         | ?               |
| Prog Cosine | ?         | ?               |
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from models.dit_world_model import DiTWorldModel
from models.diffusion import DiffusionProcess
from training.trainer import WorldModelTrainer, WorldModelDataset
from training.progressive_schedule import ProgressiveDiffusionScheduler
from evaluation.metrics import MetricsTracker, SSIMCalculator, PSNRCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Cell 1: 消融实验框架
# ============================================================

class AblationExperiment:
    """消融实验运行器."""
    
    def __init__(self, name: str, output_dir: str = "ablation_results"):
        self.name = name
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_variant(
        self,
        variant_name: str,
        model: DiTWorldModel,
        diffusion: DiffusionProcess,
        dataloader: DataLoader,
        num_epochs: int = 20,
        progressive_scheduler=None,
    ):
        """Run one ablation variant."""
        print(f"\n{'='*50}")
        print(f"  Running: {variant_name}")
        print(f"  Params: {model.get_num_params()/1e6:.1f}M")
        print(f"{'='*50}")
        
        trainer = WorldModelTrainer(
            model=model,
            diffusion=diffusion,
            lr=3e-4,
            warmup_steps=500,
            total_steps=50000,
            use_amp=(device == "cuda"),
            device=device,
            output_dir=str(self.output_dir / variant_name),
            progressive_scheduler=progressive_scheduler,
        )
        
        history = trainer.fit(dataloader, num_epochs=num_epochs)
        
        self.results[variant_name] = {
            "params_M": model.get_num_params() / 1e6,
            "final_loss": history["loss"][-1] if history["loss"] else float("inf"),
            "loss_history": history["loss"],
        }
        
        return history
    
    def plot_comparison(self, filename: str = "comparison.png"):
        """Plot loss curves for all variants."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, result in self.results.items():
            ax.plot(
                result["loss_history"],
                label=f"{name} ({result['params_M']:.1f}M) → {result['final_loss']:.4f}",
                linewidth=2,
            )
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Ablation: {self.name}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / filename), dpi=150)
        plt.show()
        
    def print_results(self):
        """Print results table."""
        print(f"\n📊 Ablation Results: {self.name}")
        print("-" * 60)
        print(f"{'Variant':<25} | {'Params':>8} | {'Final Loss':>10}")
        print("-" * 60)
        for name, r in sorted(self.results.items(), key=lambda x: x[1]["final_loss"]):
            print(f"{name:<25} | {r['params_M']:>7.1f}M | {r['final_loss']:>10.4f}")
        print("-" * 60)


# ============================================================
# Cell 2: 创建测试数据
# ============================================================

def create_test_dataset(size: int = 2000, img_size: int = 64, action_dim: int = 18):
    """Create synthetic dataset for ablation."""
    dataset = WorldModelDataset(obs_history_len=1, max_size=size)
    for i in range(size):
        obs = torch.rand(3, img_size, img_size)
        next_obs = obs + 0.05 * torch.randn_like(obs)
        next_obs = next_obs.clamp(0, 1)
        action = torch.randint(0, action_dim, (1,)).squeeze()
        reward = float(torch.randn(1).item() > 0.5)
        done = i % 200 == 199
        dataset.add(obs, action, next_obs, reward, done)
    return dataset

dataset = create_test_dataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
diffusion = DiffusionProcess(num_timesteps=1000, schedule_type="cosine")


# ============================================================
# Cell 3: 消融 1 — DiT Depth
# ============================================================

print("\n🔬 Ablation 1: DiT Depth")

ablation_depth = AblationExperiment("depth_ablation")

for depth in [4, 8, 12]:
    model = DiTWorldModel(
        img_size=64, patch_size=4, in_channels=6,
        hidden_size=128, depth=depth, num_heads=4,
        action_dim=18,
    )
    ablation_depth.run_variant(
        f"depth_{depth}", model, diffusion, dataloader, num_epochs=10,
    )

ablation_depth.print_results()
ablation_depth.plot_comparison("depth_comparison.png")


# ============================================================
# Cell 4: 消融 2 — Patch Size
# ============================================================

print("\n🔬 Ablation 2: Patch Size")

ablation_patch = AblationExperiment("patch_ablation")

for patch_size in [2, 4, 8]:
    model = DiTWorldModel(
        img_size=64, patch_size=patch_size, in_channels=6,
        hidden_size=128, depth=6, num_heads=4,
        action_dim=18,
    )
    ablation_patch.run_variant(
        f"patch_{patch_size}", model, diffusion, dataloader, num_epochs=10,
    )

ablation_patch.print_results()
ablation_patch.plot_comparison("patch_comparison.png")


# ============================================================
# Cell 5: 消融 3 — Progressive vs Fixed Schedule
# ============================================================

print("\n🔬 Ablation 3: Progressive vs Fixed Diffusion Steps")

ablation_prog = AblationExperiment("progressive_ablation")

# Fixed schedule (baseline)
model_fixed = DiTWorldModel(
    img_size=64, patch_size=4, in_channels=6,
    hidden_size=128, depth=6, num_heads=4, action_dim=18,
)
ablation_prog.run_variant(
    "fixed_100", model_fixed, diffusion, dataloader, num_epochs=15,
)

# Progressive Linear
model_prog_lin = DiTWorldModel(
    img_size=64, patch_size=4, in_channels=6,
    hidden_size=128, depth=6, num_heads=4, action_dim=18,
)
sched_linear = ProgressiveDiffusionScheduler(
    max_steps=100, min_steps=10, warmup_epochs=5,
    total_epochs=15, schedule="linear",
)
ablation_prog.run_variant(
    "progressive_linear", model_prog_lin, diffusion, dataloader,
    num_epochs=15, progressive_scheduler=sched_linear,
)

# Progressive Cosine
model_prog_cos = DiTWorldModel(
    img_size=64, patch_size=4, in_channels=6,
    hidden_size=128, depth=6, num_heads=4, action_dim=18,
)
sched_cosine = ProgressiveDiffusionScheduler(
    max_steps=100, min_steps=10, warmup_epochs=5,
    total_epochs=15, schedule="cosine",
)
ablation_prog.run_variant(
    "progressive_cosine", model_prog_cos, diffusion, dataloader,
    num_epochs=15, progressive_scheduler=sched_cosine,
)

ablation_prog.print_results()
ablation_prog.plot_comparison("progressive_comparison.png")


# ============================================================
# Cell 6: 消融 4 — Hidden Size (Scaling Law)
# ============================================================

print("\n🔬 Ablation 4: Model Scaling")

ablation_scale = AblationExperiment("scaling_ablation")

configs = [
    ("tiny_64d",  64,  4, 4),
    ("small_128d", 128, 6, 4),
    ("medium_256d", 256, 8, 8),
    ("base_384d", 384, 12, 6),
]

for name, hidden, depth, heads in configs:
    model = DiTWorldModel(
        img_size=64, patch_size=4, in_channels=6,
        hidden_size=hidden, depth=depth, num_heads=heads,
        action_dim=18,
    )
    ablation_scale.run_variant(name, model, diffusion, dataloader, num_epochs=10)

ablation_scale.print_results()
ablation_scale.plot_comparison("scaling_comparison.png")


# ============================================================
# Cell 7: 消融 5 — Noise Schedule Comparison
# ============================================================

print("\n🔬 Ablation 5: Noise Schedule")

ablation_noise = AblationExperiment("noise_schedule_ablation")

for schedule in ["linear", "cosine", "sigmoid"]:
    diff = DiffusionProcess(
        num_timesteps=1000, schedule_type=schedule,
        prediction_type="epsilon",
    )
    model = DiTWorldModel(
        img_size=64, patch_size=4, in_channels=6,
        hidden_size=128, depth=6, num_heads=4, action_dim=18,
    )
    ablation_noise.run_variant(
        f"schedule_{schedule}", model, diff, dataloader, num_epochs=10,
    )

ablation_noise.print_results()
ablation_noise.plot_comparison("noise_schedule_comparison.png")


# ============================================================
# Cell 8: 汇总所有消融实验
# ============================================================

print("\n" + "=" * 70)
print("  📊 ALL ABLATION RESULTS SUMMARY")
print("=" * 70)

for ablation in [ablation_depth, ablation_patch, ablation_prog, ablation_scale, ablation_noise]:
    ablation.print_results()

print("\n🎉 Ablation experiments complete!")
print("Next: 03_robotic_transfer.py - Transfer to robotic manipulation")
