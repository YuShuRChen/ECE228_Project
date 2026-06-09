# Reproducing and Extending "Generalizable Motion Planning via Operator Learning"

**Original paper:** [OpenReview](https://openreview.net/forum?id=UYcUpiULmT) | [arXiv](https://arxiv.org/abs/2410.17547) | [Original Code](https://github.com/ExistentialRobotics/PNO)

## Overview

This project reproduces the results of "Generalizable Motion Planning via Operator Learning" (Matada et al., ICLR 2025) and extends the Planning Neural Operator (PNO) framework to **dynamic environments with moving obstacles**. The original paper proposes PNO for static motion planning — learning to map environments to value functions with zero-shot generalization across resolutions. Our extension explores whether PNO can be used for **online replanning in dynamic settings** by incorporating risk-aware cost maps that account for moving obstacles.

### What We Did

1. **Reproduction (2D):** Independently reimplemented and retrained all models from the paper (FNO, DAFNO, PNO, PNO w/ PINN) on the 2D motion planning task, verifying the reported results across synthetic and city datasets at multiple resolutions.

2. **Extension — Dynamic Obstacle Avoidance:** Extended PNO to handle dynamic environments with moving obstacles using risk-aware planning:
   - **Risk mapping:** Compute a spatial risk map that combines static obstacle proximity (from the SDF) with dynamic obstacle risk modeled as Gaussians around moving agents.
   - **Online replanning:** At each timestep, update the risk map based on current obstacle positions and use PNO to recompute the value function, enabling reactive path planning.
   - **Two planning strategies:** Whole-map replanning (recompute the full value function) and sliding-window replanning (recompute locally around the agent for efficiency).
   - **A\* integration:** Use PNO's predicted value function as a heuristic for A\* search on the risk-weighted cost map, combining learned heuristics with classical search.

3. **Baseline Comparisons (2D):** Added DeepONet and VIN (Value Iteration Network) as additional baselines not included in the original paper, demonstrating PNO's advantages.

4. **3D and 4D:** Included the original 3D iGibson and 4D manipulator experiment code for reference.

## Project Structure

```
ECE228/
├── 2D/                         # 2D experiments (reproduction + baselines)
│   ├── models/                 # Model implementations
│   │   ├── fno.py              # FNO and FNOSDF
│   │   ├── dafno.py            # Domain-Agnostic FNO
│   │   ├── pno.py              # Planning Neural Operator
│   │   ├── deeponet.py         # DeepONet baseline
│   │   └── vin.py              # Value Iteration Network baseline
│   ├── layers/                 # Spectral convolution layers
│   ├── utilities/              # Dataset loader and losses
│   ├── dataset/                # Training and test data
│   ├── results/                # Trained model weights
│   ├── comparison_results/     # Evaluation outputs (tables, plots)
│   ├── train.py                # Training script
│   ├── evaluate.py             # Single-dataset evaluation
│   └── compare.py              # Comprehensive multi-dataset comparison
├── 3D/                         # 3D iGibson experiments (from original repo)
│   ├── models/                 # 3D model implementations
│   ├── train.py                # 3D training script
│   ├── evaluate.py             # 3D evaluation script
│   ├── TestPNO3D.ipynb         # 3D test notebook
│   └── generator/              # Dataset generation scripts
├── 4D/                         # 4D manipulator experiments (from original repo)
│   ├── train.py                # 4D training script
│   ├── planner.py              # 4D path planner
│   ├── TestPlanningOperator4D.ipynb  # 4D test notebook
│   └── generator/              # Dataset generation scripts
└── README.md
```

## 2D Experiments — Reproduction

### Models

| Model | Type | Params | Description |
|-------|------|--------|-------------|
| **FNO** | Neural Operator | 265K | Fourier Neural Operator — baseline operator learning |
| **DAFNO** | Neural Operator | 265K | Domain-Agnostic FNO — masks spectral convolutions by domain |
| **PNO** | Neural Operator | 289K | Planning Neural Operator — cascaded SDF + VF prediction |
| **PNO w/ PINN** | Neural Operator | 289K | PNO with physics-informed Eikonal loss |
| **FNOSDF** | Neural Operator | 265K | FNO for Signed Distance Function (used by PNO) |
| **DeepONet** | Neural Operator | 305K | Branch-trunk architecture baseline |
| **VIN** | Planning Network | 11K | Value Iteration Network baseline |

### Training

All models train on 64×64 synthetic environments for 500 epochs.

```bash
cd 2D

# Paper models
python train.py --model fnosdf --epochs 500
python train.py --model fno --epochs 500
python train.py --model dafno --epochs 500
python train.py --model pno --epochs 500
python train.py --model pno --use_pinn --epochs 500

# Baselines
python train.py --model deeponet --epochs 500
python train.py --model vin --epochs 500
```

### Evaluation

```bash
# Comprehensive comparison across all datasets
python compare.py

# Outputs to comparison_results/:
#   - l2_synthetic.csv, l2_city.csv, summary_64x64.csv
#   - all_results.json
#   - training_curves.png, accuracy_vs_speed.png, visual_comparison.png
```

### Results

#### Relative L2 Error — Synthetic Datasets (Zero-Shot Generalization)

| Model | 64×64 (train) | 256×256 | 512×512 | 1024×1024 |
|-------|:---:|:---:|:---:|:---:|
| FNO | 0.2156 | 0.5501 | 0.5896 | 0.6044 |
| DeepONet | 0.6604 | 0.6613 | 0.6613 | 0.6614 |
| VIN | 0.4322 | 0.4762 | 0.4914 | 0.5000 |
| DAFNO | 0.1255 | 0.3840 | 0.4059 | 0.4127 |
| **PNO** | **0.1250** | **0.1184** | **0.1182** | **0.1185** |
| **PNO+PINN** | **0.1217** | **0.1116** | **0.1116** | **0.1113** |

#### Relative L2 Error — City Datasets (Domain Transfer)

| Model | 256×256 | 512×512 | 1024×1024 |
|-------|:---:|:---:|:---:|
| FNO | 0.6048 | 0.6327 | 0.6489 |
| DeepONet | 0.6398 | 0.6262 | 0.6187 |
| VIN | 0.4680 | 0.4853 | 0.4966 |
| DAFNO | 0.4248 | 0.4516 | 0.4613 |
| **PNO** | **0.1828** | **0.2078** | **0.2262** |
| **PNO+PINN** | **0.1673** | **0.1862** | **0.2026** |

### Key Findings

1. **PNO and PNO+PINN** maintain low error across all resolutions and domains, confirming the paper's core claim of zero-shot generalization. Other neural operators (FNO, DAFNO) degrade significantly at higher resolutions.
2. **PINN regularization helps:** PNO+PINN consistently outperforms PNO by 2-3% across all settings, validating the physics-informed loss.
3. **Baseline comparison:** DeepONet and VIN, despite being established architectures, fail to learn resolution-invariant representations for this task.
4. **Reproduction accuracy:** Our results closely match the paper for DAFNO (0.1255 vs 0.1276), PNO (0.1250 vs 0.1251), and PNO+PINN (0.1217 vs 0.1373, ours is better).

## Extension — Dynamic Obstacle Avoidance

The original PNO framework assumes a static environment. We extend it to handle **dynamic obstacles** — moving agents that the planner must reactively avoid.

### Approach

1. **Risk Map Construction:** At each timestep, a spatial risk map is computed by combining:
   - Static risk from proximity to walls (derived from the SDF)
   - Dynamic risk from Gaussian fields centered on each moving obstacle, parameterized by `sigma_dynamic` and `alpha_dynamic`

2. **Online Replanning with PNO:** The risk map is injected into PNO's input (either replacing the binary occupancy or augmenting the chi function), and the value function is recomputed at each timestep to account for new obstacle positions.

3. **Planning Strategies:**
   - **Whole-map:** Recompute the entire value function at each step — accurate but expensive
   - **Sliding window:** Recompute only a local window around the agent — faster with small accuracy trade-off

4. **Path Execution:** A* search on the risk-weighted cost map uses PNO's value function as a heuristic, combining the learned global structure with local obstacle avoidance.

## 3D iGibson Experiments

The 3D experiments use the [iGibson](https://arxiv.org/abs/1910.14442) benchmark environments with voxelized occupancy grids.

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/sharathmatada/igib-dataset-160-5G)
- **Pre-trained model:** [HuggingFace](https://huggingface.co/sharathmatada/iGibson_3D_Model)

## 4D Manipulator Experiments

The 4D experiments operate in the configuration space of the [ROBOTIS OpenMANIPULATOR](https://www.mathworks.com/help/robotics/ref/loadrobot.html) robot.

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/sharathmatada/4D-OccupancyGrids)
- **Pre-trained model:** [HuggingFace](https://huggingface.co/sharathmatada/4D_Manipulator_Model)

## Citation

```
@inproceedings{
matada2025generalizable,
title={Generalizable Motion Planning via Operator Learning},
author={Sharath Matada and Luke Bhan and Yuanyuan Shi and Nikolay Atanasov},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=UYcUpiULmT}
}
```
