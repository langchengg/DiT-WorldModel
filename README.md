# DiT-WorldModel

DiT-WorldModel is an action-conditioned diffusion world model that uses a Diffusion Transformer (DiT) backbone for future-frame generation, next-observation modeling, and short visual rollouts. This repository contains model code, training loops, dataset loaders, action discretization utilities, navigation demos, and visualization helpers for experiments across real navigation datasets, simulator rollouts, and synthetic debugging setups.

## Project Status

This repository is research code in progress, not a finished benchmark release.

The current checkout mixes three data regimes:

- Real public datasets: RECON, TartanDrive
- Simulator rollouts collected online: MetaWorld, Atari
- Synthetic or debug data: GridNavigationEnv, demo random tensors, notebook-generated toy data

Only publish results that you reproduced from your own runs and saved artifacts. This README intentionally does not include benchmark tables, claimed percentage gains, or paper-style result summaries that are not backed by bundled logs and evaluation outputs.

Important scope notes:

- `data/recon_sample/recon_datavis.tar.gz` is a RECON visualization package archive, not a ready-to-train RECON sample dataset.
- `configs/navigation.yaml` defaults to synthetic grid data unless you edit the dataset block.
- `notebooks/04_navigation_demo.py` can train on `recon` or `tartan`, but its imagination, trajectory comparison, and MPC visualization steps still switch to `GridNavigationEnv`.
- The current repository snapshot does not include a root `LICENSE` file, so licensing should be clarified before public release.

## What Is Implemented

- DiT-based world model architecture in `models/dit_world_model.py`
- Diffusion training and DDIM sampling in `models/diffusion.py`
- Progressive diffusion scheduling in `training/progressive_schedule.py`
- Continuous-action discretization in `models/action_discretizer.py`
- Dataset loaders for RECON, TartanDrive, and synthetic navigation in `navigation/dataset.py`
- Training loop, checkpoints, and visualization utilities in `training/` and `evaluation/`
- World-model-driven navigation utilities and video export in `navigation/`

## Supported Data Sources

| Source | Type | How data is obtained | Can be shown as real-data result in README |
| --- | --- | --- | --- |
| RECON | Real public dataset | Loaded from HDF5 trajectories via `RECONDataset`; requires real RECON files or a successful dataset download path | Yes, if the figure or metric comes from held-out RECON sequences |
| TartanDrive | Real public dataset | Loaded from downloaded run directories via `TartanDriveDataset` | Yes, if the figure or metric comes from held-out TartanDrive sequences |
| MetaWorld | Simulator environment | Collected online from `metaworld` environments in `main.py` and `notebooks/03_robotic_transfer.py` | No |
| Atari | Simulator environment | Collected online from `gymnasium` Atari environments in `main.py` | No |
| `GridNavigationEnv` / `SyntheticNavigationDataset` | Synthetic or debug | Generated inside the repository | No |
| `main.py --demo` random data | Synthetic or debug | Random tensors, random actions, and random rewards | No |
| Notebook toy action and sequence examples | Synthetic or debug | Notebook cells synthesize action samples, observation sequences, or augmentation inputs | No |

Notes:

- The RECON loader expects `.hdf5` trajectory files. The bundled `recon_datavis` archive is not that dataset.
- MetaWorld and Atari are useful experimental environments, but they are not real-world datasets and should not be described that way in project-facing results.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/` | Training and environment configurations |
| `models/` | DiT world model, diffusion process, temporal attention, action discretization |
| `training/` | Trainer, augmentation, progressive schedulers |
| `navigation/` | Dataset loaders, grid simulator, navigation logic, visualization |
| `evaluation/` | Metrics and figure/video export utilities |
| `notebooks/` | Reproduction, ablation, robotic transfer, and navigation demo scripts |
| `data/` | Local dataset cache or manually prepared data |
| `outputs/`, `results/` | Checkpoints and generated artifacts |

## Setup

```bash
git clone https://github.com/langchengg/DiT-WorldModel.git
cd DiT-WorldModel

conda create -n wm python=3.10
conda activate wm
pip install -r requirements.txt
```

Dependency notes:

- `metaworld` is only needed for simulator-based robotic runs.
- `gymnasium` and `ale-py` are only needed for Atari runs.
- `h5py` and `opencv-python` are needed for RECON or TartanDrive loading.

## Training and Evaluation Commands

### Smoke test

```bash
python main.py --config configs/dit_small.yaml --demo --epochs 5
```

This is a synthetic debug run. It is useful for checking that the training loop, checkpointing, and plotting code execute, but it is not a real-data experiment.

### Atari simulator rollout training

```bash
python main.py --config configs/dit_small.yaml
```

This collects data online from an Atari environment and trains on those simulator rollouts.

### MetaWorld simulator rollout training

```bash
python main.py --config configs/robotic_env.yaml
```

This collects data online from MetaWorld and trains on simulator rollouts. It is not a real robot dataset pipeline.

### Navigation training pipeline

```bash
python main.py --config configs/navigation.yaml
```

This command exists, but the default `configs/navigation.yaml` uses synthetic grid data. For real navigation data, update the dataset block or use the notebook command surface below.

### RECON navigation demo

```bash
python notebooks/04_navigation_demo.py --dataset recon --data_dir /path/to/recon
```

### TartanDrive navigation demo

```bash
python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path/to/tartan
```

These notebook commands load real navigation datasets for the training stage. The later visualization and MPC stages still switch to the synthetic grid environment, so the resulting navigation figures are not real-dataset evaluation outputs.

## Output Files Produced by the Current Code

### `main.py`

By default, `main.py` writes:

- `outputs/checkpoints/checkpoint_epoch_XXXX.pt`
- `outputs/checkpoints/best_model.pt`
- `outputs/training_curves.png`

### `notebooks/04_navigation_demo.py`

When the corresponding stages complete, the current script writes:

- `outputs/navigation/training_curves.png`
- `outputs/navigation/4_frames_comparison.png`
- `outputs/navigation/navigation_episode.png`
- `outputs/navigation/navigation.mp4`
- `outputs/navigation/navigation_metrics.png`

Notes:

- The current script does not save `imagination_demo.png` or `ground_truth_comparison.png`, even if older files with those names may already exist in a working directory.
- `4_frames_comparison.png` is generated from `GridNavigationEnv` in the current script, not from held-out RECON or TartanDrive sequences.

## What To Show After Full Training

If you want a results section built around real data, keep it restricted to RECON or TartanDrive runs and use held-out sequences.

Recommended result figures:

| Figure | What it should show | Safe to present as real-data result |
| --- | --- | --- |
| Held-out sample grid | Raw frames from the evaluation split | Yes |
| Ground truth vs prediction vs absolute difference | One-step next-observation comparison on held-out real sequences | Yes |
| Multi-step rollout strip | Short rollout on held-out real sequences with ground truth and predicted rows | Yes |
| Training and validation curves | Loss and evaluation metrics from the real-data run | Yes |
| Denoising strip | Intermediate denoising states for one held-out real frame | Yes |
| Dataset statistics plot | Action histograms, trajectory lengths, frame counts, or split summary from the real dataset | Yes |

Method or background figures are still useful, but they belong in a methods section rather than a results section. Good examples:

- Diffusion schedule diagram
- Denoising-process illustration
- Architecture diagram

Do not use the following current repository assets as empirical real-data results:

- `all_images/demo_training_loss.png`
- `all_images/imagination.png`
- `all_images/action_discretization.png`
- `all_images/temporal_augmentation.png`
- `all_images/robot_training.png`

Additional caution:

- The current `outputs/navigation/4_frames_comparison.png`
- The current `outputs/navigation/navigation_episode.png`
- The current `outputs/navigation/navigation.mp4`
- The current `outputs/navigation/navigation_metrics.png`

should be treated as synthetic-environment visuals unless the evaluation path is changed to use held-out RECON or TartanDrive sequences instead of `GridNavigationEnv`.

## Limitations and Honesty Notes

- `notebooks/03_robotic_transfer.py` refers to MetaWorld as if it were a real dataset in some comments, but the code actually collects simulator rollouts online.
- `notebooks/01_reproduction.py` and `notebooks/02_dit_ablation.py` use synthetic data for demos and ablations.
- `main.py --demo` is a random-data smoke test, not a benchmark run.
- `notebooks/04_navigation_demo.py` currently mixes real-data training with synthetic-environment evaluation visuals.
- No reproduced real-data benchmark table is bundled with this repository snapshot.

## Suggested Results Section for Your Own Runs

Once you have completed real-data training, a good README results section should report:

- Dataset and split
- Exact command or config used
- Checkpoint path
- Metrics computed on held-out real sequences
- Figures generated from held-out real sequences only

If you add metrics, make the source split and evaluation script explicit. Do not mix simulator visuals and real-dataset visuals in the same results table without labeling them.

## References

- [DIAMOND: Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399)
- [MetaWorld evaluation documentation](https://metaworld.farama.org/evaluation/evaluation/)
- [RECON dataset page](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive repository](https://github.com/castacks/tartandrive)
