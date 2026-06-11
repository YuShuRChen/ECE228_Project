<div align="center">
  <a href="http://erl.ucsd.edu/">
    <img align="left" src="docs/static/images/erl.png" width="80" alt="erl">
  </a>
  <a href="https://contextualrobotics.ucsd.edu/">
    <img align="center" src="docs/static/images/cri.png" width="150" alt="cri">
  </a>
  <a href="https://ucsd.edu/">
    <img align="right" src="docs/static/images/ucsd.png" width="260" alt="ucsd">
  </a>
</div>

# Risk-Aware Planning Neural Operator (PNO) with Sliding Window Optimization

This repository extends the original **Planning Neural Operator (PNO)** framework to incorporate **dynamic risk awareness** and a **sliding window optimization** for efficient high-resolution path planning.

## Key Features
- **Risk-Aware PNO:** A modified model architecture trained with a dedicated risk channel and PINN-based loss to navigate safely through dynamic, uncertain environments.
- **Sliding Window Optimization:** Scales PNO-based planning to large maps (e.g., 256x256 and beyond) by focusing computational resources on a local window around the agent, drastically reducing node expansions.
- **Continuous Cost-to-Go:** Leverages neural operator learning to predict continuous cost-to-go fields, enabling smoother and safer paths than traditional binary occupancy grids.

---

## Installation

### 1. Environment Setup
We provide Conda environment files for both Linux/Windows and Mac (Silicon).

```bash
# For Mac Silicon:
conda env create -f environment_mac.yml
conda activate pno

# For Linux/Windows:
conda env create -f environment.yml
conda activate pno
```

### 2. Dataset and Models
The project uses a synthetic dataset of 64x64 maps for training and 256x256 maps for evaluation. Pre-trained weights are provided in `examples/models/`.

---

## Reproducing Results

We have provided a suite of scripts in the `examples/` directory to run simulations and evaluate performance.

### 1. Single Map Demonstration
To visualize the planning process on a single map with dynamic obstacles, run:

```bash
# Run the best-performing sliding window method
python examples/demonstration.py --map_size 256 --map_idx 0 --method sliding_window --model new

# Run the binary-only baseline
python examples/demonstration.py --map_size 256 --map_idx 0 --method whole_map --model original --use_binary_cost
```
This will save an animation to `data/{map_size}/{method}/{map_idx}.gif`.

### 2. Batch Experiments
To run a batch of experiments across the entire 256x256 evaluation set (100 maps) for a specific method:

```bash
# Baseline (Binary Hard)
python examples/experiment.py --map_size 256 --method whole_map --model original --use_binary_cost

# Whole Map (Risk-Aware)
python examples/experiment.py --map_size 256 --method whole_map --model new

# Sliding Window (Risk-Aware)
python examples/experiment.py --map_size 256 --method sliding_window --model new
```

### 3. Performance Evaluation
Once the batch experiments are complete, generate a summary table comparing the methods:

```bash
python examples/evaluate_results.py
```
This will output a performance table to the console and save it to `evaluation_summary_256.csv`.

---

## Experimental Results (256x256)

Our evaluation on 100 synthetic maps demonstrates that the **Sliding Window** approach significantly outperforms both the binary baseline and the whole-map risk model in terms of efficiency and safety.

| Method | Success Rate | Avg Nodes Expanded | Node Imp. % | Avg Cumul. Risk | Risk Imp. % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Binary)** | 80% | 1,588,296 | 0.00% | 127.20 | 0.00% |
| **Whole Map (Risk)** | 100% | 3,618,125 | -127.80% | 10.61 | +91.66% |
| **Sliding Window** | **100%** | **51,731** | **+96.74%** | **10.53** | **+91.72%** |

### Key Findings:
- **Efficiency:** The sliding window optimization is **~30x faster** than the baseline and **~70x faster** than whole-map planning.
- **Safety:** Both risk-aware models reduce cumulative risk by over **91%** compared to the binary baseline.
- **Reliability:** Risk-aware planning achieved a **100% success rate**, whereas the binary baseline failed on 20% of the maps due to dynamic obstacle collisions.
