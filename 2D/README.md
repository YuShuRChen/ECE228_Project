# 2D Motion Planning Experiments

This directory contains the full reproduction and extension of the 2D results from "Generalizable Motion Planning via Operator Learning" (ICLR 2025).

See the [project-level README](../README.md) for an overview, results, and instructions.

## Quick Start

```bash
# Train all models (500 epochs each)
python train.py --model fnosdf --epochs 500
python train.py --model fno --epochs 500
python train.py --model dafno --epochs 500
python train.py --model pno --epochs 500
python train.py --model pno --use_pinn --epochs 500
python train.py --model deeponet --epochs 500
python train.py --model vin --epochs 500

# Run comprehensive evaluation
python compare.py
```

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script for all models |
| `evaluate.py` | Quick evaluation on a single dataset |
| `compare.py` | Full comparison across all datasets with tables and plots |
| `plot.py` | Paper-style visualization |
| `models/` | Model implementations (FNO, DAFNO, PNO, DeepONet, VIN) |
| `layers/` | Spectral convolution layers |
| `utilities/` | Dataset loader (`dataset.py`) and losses (`losses.py`) |
| `dataset/` | Training and test data |
| `results/` | Trained weights and loss curves |
| `comparison_results/` | Evaluation outputs (CSV, JSON, PNG) |
