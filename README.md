# DiT-WorldModel

DiT-WorldModel is an action-conditioned diffusion world model with a Diffusion Transformer (DiT) backbone for next-observation modeling, future-frame generation, and short visual rollouts.

This repository now follows a real-data-first navigation workflow:

- Default navigation config: `RECON`
- Optional second real dataset: `TartanDrive`
- Synthetic grid world: debugging and MPC-only
- MetaWorld and Atari: simulator rollouts, not real datasets

## What This Repository Does End to End

From start to finish, the navigation pipeline does the following:

1. Load a dataset of transitions.
2. Convert each transition into the unified training format:
   `obs_history`, `obs_next`, `action`, `reward`, `done`
3. For real datasets (`RECON`, `TartanDrive`), discretize continuous controls into a 256-way action vocabulary.
4. Train a DiT-based diffusion world model to predict the next observation, plus reward and termination heads.
5. Run DDIM sampling on held-out data to generate predicted next frames.
6. Save training curves, side-by-side comparisons, qualitative panels, videos, and evaluation metrics.
7. If you explicitly choose synthetic data, the repo can also run MPC navigation inside `GridNavigationEnv`.

If your goal is a publishable README or a results page, use `RECON` or `TartanDrive`, not synthetic data.

## Real Data vs Simulator vs Synthetic

| Source | Category | How it is used here | Should you present it as real-data evidence? |
| --- | --- | --- | --- |
| RECON | Real public dataset | Real outdoor navigation trajectories loaded from HDF5 files | Yes |
| TartanDrive | Real public dataset | Real off-road driving runs loaded from image folders and control files | Yes |
| MetaWorld | Simulator | Online rollouts collected from simulator environments | No |
| Atari | Simulator | Online rollouts collected from Gymnasium Atari environments | No |
| GridNavigationEnv | Synthetic | Local grid-world generator for debugging and MPC demos | No |
| `main.py --demo` | Synthetic | Random tensors and random rewards for smoke testing | No |
| Notebook toy examples | Synthetic | Simulated action distributions and synthetic image sequences | No |

Important note:

- `data/recon_sample/recon_datavis.tar.gz` is not a ready-to-train RECON sample dataset. It is a visualization package archive.

## Start Here: Real-Data Workflow

### Step 1. Install dependencies

```bash
git clone https://github.com/langchengg/DiT-WorldModel.git
cd DiT-WorldModel

conda create -n wm python=3.10
conda activate wm
pip install -r requirements.txt
```

### Step 2. Decide which real dataset you want to use

#### Option A: RECON

You can use either:

- your own RECON directory containing `.hdf5` trajectory files, or
- the repo's RECON code path with `data_dir: null`, which lets `RECONDataset` try its sample download path

Expected RECON structure:

```text
/path/to/recon/
  traj_0000.hdf5
  traj_0001.hdf5
  ...
```

#### Option B: TartanDrive

Expected TartanDrive structure:

```text
/path/to/tartan/
  run_0001/
    image_left/
      000000.png
      000001.png
      ...
    cmd.npy
  run_0002/
  ...
```

### Step 3. Run the real-data training pipeline

#### Option A: Main training entrypoint, defaulted to real navigation data

`configs/navigation.yaml` now defaults to `RECON` and a 256-bin action space for real navigation data.

```bash
python main.py --config configs/navigation.yaml
```

What this does:

1. Builds the navigation dataset loader.
2. Uses `RECON` by default unless you change the config.
3. Resolves the action vocabulary correctly for real datasets.
4. Trains the DiT world model.
5. Saves checkpoints and training curves.

Main outputs:

- `outputs/checkpoints/checkpoint_epoch_XXXX.pt`
- `outputs/checkpoints/best_model.pt`
- `outputs/training_curves.png`

#### Option B: End-to-end notebook script for real-data training plus held-out evaluation

Default command, real-data-first:

```bash
python notebooks/04_navigation_demo.py
```

Explicit RECON path:

```bash
python notebooks/04_navigation_demo.py --dataset recon --data_dir /path/to/recon
```

Explicit TartanDrive path:

```bash
python notebooks/04_navigation_demo.py --dataset tartan --data_dir /path/to/tartan
```

What this script does on real datasets:

1. Loads `RECON` or `TartanDrive`.
2. Splits the data into train and held-out evaluation subsets.
3. Builds the DiT world model with the correct 256-way action vocabulary.
4. Trains on the training subset.
5. Runs one-step next-frame prediction on held-out samples.
6. Saves qualitative and quantitative evaluation artifacts.

Real-data outputs from `notebooks/04_navigation_demo.py`:

- `outputs/navigation/training_curves.png`
- `outputs/navigation/4_frames_comparison.png`
- `outputs/navigation/navigation_episode.png`
- `outputs/navigation/navigation.mp4`
- `outputs/navigation/navigation_metrics.png`

### Step 4. Understand what each output means

For real datasets (`recon` or `tartan`):

- `training_curves.png`
  Training loss curves from the real-data run
- `4_frames_comparison.png`
  Held-out next-frame comparison: ground truth vs prediction
- `navigation_episode.png`
  Held-out qualitative panel: context frame, ground-truth next frame, predicted next frame
- `navigation.mp4`
  Real-data qualitative video built from held-out predictions
- `navigation_metrics.png`
  Held-out evaluation metrics such as SSIM, PSNR, and LPIPS across samples

For synthetic data only:

- `navigation_episode.png`, `navigation.mp4`, and `navigation_metrics.png`
  represent MPC navigation inside `GridNavigationEnv`, not real-data evaluation

### Step 5. Use only real-data results in your public presentation

If you are writing a README, report, or project page, use figures from held-out `RECON` or `TartanDrive` runs only.

Good result figures:

- held-out sample grid from the raw dataset
- ground truth vs prediction vs absolute difference
- short held-out rollout strip
- training and validation curves
- denoising strip for one held-out frame
- dataset statistics plot from real trajectories

Do not present these current assets as real-data evidence:

- `all_images/demo_training_loss.png`
- `all_images/imagination.png`
- `all_images/action_discretization.png`
- `all_images/temporal_augmentation.png`
- `all_images/robot_training.png`

## Synthetic and Simulator Workflows

These still exist, but they are no longer the main path.

### Synthetic debugging only

```bash
python notebooks/04_navigation_demo.py --dataset synthetic
```

Use this only if you want:

- a CPU-friendly smoke test
- MPC navigation in the grid world
- debugging of the sampler, planner, or visualization code

### Random-data smoke test

```bash
python main.py --config configs/dit_small.yaml --demo --epochs 5
```

This is not a real experiment.

### Simulator-only workflows

Atari:

```bash
python main.py --config configs/dit_small.yaml
```

MetaWorld:

```bash
python main.py --config configs/robotic_env.yaml
```

These are simulator rollouts, not real datasets.

## File and Module Guide

| Path | What it does |
| --- | --- |
| `configs/navigation.yaml` | Real-data-first navigation training config |
| `main.py` | Main training entrypoint |
| `navigation/dataset.py` | RECON, TartanDrive, and synthetic dataset loaders |
| `notebooks/04_navigation_demo.py` | End-to-end navigation demo; real data uses held-out evaluation, synthetic uses MPC |
| `models/dit_world_model.py` | DiT world model |
| `models/diffusion.py` | Diffusion process, DDIM sampler, world model environment |
| `training/trainer.py` | Training loop and checkpointing |
| `evaluation/metrics.py` | SSIM, PSNR, LPIPS, FID utilities |
| `evaluation/visualize.py` | Grid, video, and curve plotting helpers |

## Current Honesty Notes

- The repo mixes real datasets, simulators, and synthetic data, but the default navigation path is now real-data-first.
- `MetaWorld` comments in some notebooks may still read like “real data”, but the code is simulator rollout collection.
- `Atari` is also simulator data.
- `GridNavigationEnv` is useful for MPC demos, but it is not a real dataset.
- This repository snapshot still does not include a root `LICENSE` file.

## Recommended Public Results Section Template

When you finish training on real data, structure your results section like this:

1. Dataset and split
2. Exact command used
3. Checkpoint path
4. Held-out metrics
5. Held-out qualitative figures
6. Clear note that simulator and synthetic runs are not included as evidence

Example fields:

- Dataset: `RECON`
- Split: train / held-out eval
- Command: `python notebooks/04_navigation_demo.py --dataset recon --data_dir /path/to/recon`
- Outputs: `4_frames_comparison.png`, `navigation_episode.png`, `navigation.mp4`, `navigation_metrics.png`

## References

- [DIAMOND: Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399)
- [MetaWorld evaluation documentation](https://metaworld.farama.org/evaluation/evaluation/)
- [RECON dataset page](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive repository](https://github.com/castacks/tartandrive)
