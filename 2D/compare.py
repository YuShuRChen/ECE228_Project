#!/usr/bin/env python3
"""
Comprehensive comparison script for 2D motion planning models.

Evaluates all available models across all datasets and generates:
1. Results tables (printed + CSV)
2. Training curves plot
3. Visual comparison of predictions
4. Accuracy vs Speed scatter plot

All models are trained on synthetic/64x64 and evaluated zero-shot on
higher resolutions and real-world city data.

Usage:
    python compare.py                        # evaluate all, save to comparison_results/
    python compare.py --save_dir my_results  # custom output directory
    python compare.py --no_plot              # skip plot generation
    python compare.py --models fno pno fmm   # evaluate specific models only
"""

import os
import sys
import argparse
import time
import csv
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from functools import reduce
import operator

dirname = os.path.dirname(os.path.abspath(__file__))
if dirname not in sys.path:
    sys.path.insert(0, dirname)

from models.fno import FNO2d, FNO2dMultiGoal
from models.dafno import DAFNO2dMultiGoal
from models.pno import DEEPNORM2dMultiGoal
from utilities.dataset import MotionPlanningDataset
from utilities.losses import LpLoss
from scipy.ndimage import distance_transform_edt

# Optional imports (may not be available yet)
try:
    from models.deeponet import DeepONet2dMultiGoal
    HAS_DEEPONET = True
except ImportError:
    HAS_DEEPONET = False

try:
    from models.vin import VIN2dMultiGoal
    HAS_VIN = True
except ImportError:
    HAS_VIN = False


# ============================================================
# Model registry
# ============================================================

def count_params(model):
    """Count parameters, counting complex params as 2 reals."""
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def smooth_chi(mask, dist, smooth_coef=5):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5


def load_model(name, device, results_dir, width=16, modes=8):
    """
    Load a trained model from results_dir. Returns (model, apply_mask, uses_sdf)
    or (None, None, None) if weights not found.
    """
    weight_map = {
        "FNOSDF": "FNOSDF", "FNO": "FNO", "DeepONet": "DeepONet",
        "VIN": "VIN", "DAFNO": "DAFNO", "PNO": "PNO", "PNO+PINN": "PNOwPINN",
    }
    folder = weight_map.get(name, name)
    weight_path = os.path.join(results_dir, folder, "best_model.pt")

    if not os.path.exists(weight_path):
        return None, None, None

    # apply_mask: True for DAFNO/PNO/PNO+PINN, False for FNO/DeepONet/VIN
    # uses_sdf: True for PNO/PNO+PINN (cascaded evaluation)
    configs = {
        "FNOSDF":   (lambda: FNO2d(1, 1, width, (modes, modes), 4), False, False),
        "FNO":      (lambda: FNO2dMultiGoal(4, 1, modes, modes, width), False, False),
        "DAFNO":    (lambda: DAFNO2dMultiGoal(4, modes, modes, width), True, False),
        "PNO":      (lambda: DEEPNORM2dMultiGoal(4, modes, modes, width), True, True),
        "PNO+PINN": (lambda: DEEPNORM2dMultiGoal(4, modes, modes, width), True, True),
    }

    if name == "DeepONet":
        if not HAS_DEEPONET:
            return None, None, None
        model = DeepONet2dMultiGoal()
        apply_mask, uses_sdf = False, False
    elif name == "VIN":
        if not HAS_VIN:
            return None, None, None
        model = VIN2dMultiGoal()
        apply_mask, uses_sdf = False, False
    elif name in configs:
        factory, apply_mask, uses_sdf = configs[name]
        model = factory()
    else:
        return None, None, None

    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model, apply_mask, uses_sdf


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, name, apply_mask, test_loader, device,
                   sdf_model=None, num_time_inst=100):
    """
    Evaluate a model. Returns dict with l2_loss, sdf_ms, vf_ms, total_ms.
    """
    loss_func = LpLoss(d=2, p=2)

    # --- L2 Loss ---
    test_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for mask, chi, goals, y, dist in test_loader:
            mask, chi, goals, y = mask.to(device), chi.to(device), goals.to(device), y.to(device)

            if sdf_model is not None:
                chi = smooth_chi(mask, sdf_model(mask), 5)

            out = model(chi, goals)

            if apply_mask:
                out = out * mask

            test_loss += loss_func(out, y).item()
            n_samples += y.shape[0]

    l2_loss = test_loss / n_samples

    # --- Timing ---
    time_sdf = 0.0
    time_vf = 0.0
    sample = next(iter(test_loader))
    mask_s = sample[0][0:1].to(device)
    chi_s = sample[1][0:1].to(device)
    goals_s = sample[2][0:1].to(device)
    mask_np = mask_s[0, 0].detach().cpu().numpy()

    with torch.no_grad():
        for _ in range(num_time_inst):
            # SDF time
            if sdf_model is not None:
                t1 = time.time()
                chi_t = smooth_chi(mask_s, sdf_model(mask_s), 5)
                t2 = time.time()
            else:
                t1 = time.time()
                distance_transform_edt(mask_np)
                t2 = time.time()
                chi_t = chi_s
            time_sdf += (t2 - t1)

            # VF time
            t3 = time.time()
            out = model(chi_t, goals_s)
            if apply_mask:
                out = out * mask_s
            t4 = time.time()
            time_vf += (t4 - t3)

    sdf_ms = (time_sdf / num_time_inst) * 1000
    vf_ms = (time_vf / num_time_inst) * 1000
    total_ms = sdf_ms + vf_ms

    return {
        "l2_loss": l2_loss,
        "sdf_ms": sdf_ms,
        "vf_ms": vf_ms,
        "total_ms": total_ms,
    }


def evaluate_sdf(model, test_loader, device, num_time_inst=100):
    """Evaluate the FNOSDF model."""
    loss_func = LpLoss(d=2, p=2)
    test_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for mask, chi, goals, y, dist in test_loader:
            mask, dist = mask.to(device), dist.to(device)
            out = model(mask)
            test_loss += loss_func(out, dist).item()
            n_samples += mask.shape[0]

    l2_loss = test_loss / n_samples

    # Timing
    time_sdf = 0.0
    sample = next(iter(test_loader))
    mask_s = sample[0][0:1].to(device)
    with torch.no_grad():
        for _ in range(num_time_inst):
            t1 = time.time()
            _ = model(mask_s)
            t2 = time.time()
            time_sdf += (t2 - t1)

    sdf_ms = (time_sdf / num_time_inst) * 1000
    return {"l2_loss": l2_loss, "sdf_ms": sdf_ms}


# ============================================================
# Dataset loading
# ============================================================

DATASETS = [
    ("Synthetic 64×64", "synthetic/64x64", True),      # has train split
    ("Synthetic 256×256", "synthetic/256x256", False),
    ("Synthetic 512×512", "synthetic/512x512", False),
    ("Synthetic 1024×1024", "synthetic/1024x1024", False),
    ("City 256×256", "cityData/256x256", False),
    ("City 512×512", "cityData/512x512", False),
    ("City 1024×1024", "cityData/1024x1024", False),
]

# Models that use VF (value function) evaluation
VF_MODELS = ["FNO", "DeepONet", "VIN", "DAFNO", "PNO", "PNO+PINN"]


def load_dataset_for_eval(data_dir, has_train_split):
    """Load dataset for evaluation. For 64x64, use the test split (last 20%).
    For all others, use ALL data (zero-shot generalization).
    Batch size scales down for higher resolutions to avoid OOM."""
    dataset = MotionPlanningDataset(data_dir)
    n = len(dataset)
    if has_train_split:
        n_test = n - int(n * 0.8)
        test_dataset = Subset(dataset, range(n - n_test, n))
    else:
        test_dataset = dataset
        n_test = n

    # Adaptive batch size based on resolution
    sample = dataset[0]
    res = sample[0].shape[-1]  # H dimension
    if res >= 1024:
        batch_size = 1
    elif res >= 512:
        batch_size = 2
    elif res >= 256:
        batch_size = 5
    else:
        batch_size = 20

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False), n_test


# ============================================================
# Output formatting
# ============================================================

def format_loss(val):
    if val is None:
        return "-"
    return f"{val:.4f}"


def format_time(val):
    if val is None:
        return "-"
    return f"{val:.2f}"


def print_table(title, headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_str = "|" + "|".join(h.center(w) for h, w in zip(headers, col_widths)) + "|"

    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "|" + "|".join(str(v).center(w) for v, w in zip(row, col_widths)) + "|"
        print(row_str)
    print(sep)


def save_csv(filepath, headers, rows):
    """Save results as CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)




# ============================================================
# Plotting
# ============================================================

def plot_training_curves(results_dir, save_path, models_to_plot=None):
    """Plot training loss curves for all trained models."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed, skipping training curves plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    model_dirs = {
        "FNO": "FNO", "DAFNO": "DAFNO", "PNO": "PNO",
        "PNO+PINN": "PNOwPINN", "DeepONet": "DeepONet", "VIN": "VIN",
    }

    colors = {
        "FNO": "#e74c3c", "DAFNO": "#3498db", "PNO": "#2ecc71",
        "PNO+PINN": "#9b59b6", "DeepONet": "#e67e22", "VIN": "#1abc9c",
    }

    for name, folder in model_dirs.items():
        if models_to_plot and name.lower() not in [m.lower() for m in models_to_plot]:
            continue
        train_path = os.path.join(results_dir, folder, "training_losses.txt")
        test_path = os.path.join(results_dir, folder, "test_losses.txt")
        if not os.path.exists(train_path):
            continue
        train_losses = np.loadtxt(train_path)
        test_losses = np.loadtxt(test_path)
        color = colors.get(name, None)
        ax1.plot(train_losses, label=name, color=color, linewidth=1.5)
        ax2.plot(test_losses, label=name, color=color, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Relative L2)")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Loss (Relative L2)")
    ax2.set_title("Test Loss")
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {save_path}")


def plot_accuracy_vs_speed(results_64, save_path):
    """Scatter plot: L2 loss vs inference time for 64x64."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed, skipping scatter plot.")
        return

    colors = {
        "FNO": "#e74c3c", "DAFNO": "#3498db",
        "PNO": "#2ecc71", "PNO+PINN": "#9b59b6",
        "DeepONet": "#e67e22", "VIN": "#1abc9c",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results_64.items():
        if name == "FNOSDF" or res is None:
            continue
        l2 = res.get("l2_loss")
        t = res.get("total_ms") or res.get("vf_ms", 0)
        if l2 is None or t is None:
            continue
        ax.scatter(t, l2, s=120, c=colors.get(name, "#333"),
                   label=name, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(name, (t, l2), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Accuracy vs Speed (Synthetic 64×64)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved accuracy vs speed plot: {save_path}")


def plot_visual_comparison(test_loader, models_dict, sdf_model, device, save_path,
                           sample_idx=0):
    """Plot side-by-side predictions for a single sample."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed, skipping visual comparison.")
        return

    # Get sample
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):
        # Subset
        actual_idx = dataset.indices[sample_idx]
        mask, chi, goals, y, dist = dataset.dataset[actual_idx]
    else:
        mask, chi, goals, y, dist = dataset[sample_idx]

    mask = mask.unsqueeze(0).to(device)
    chi = chi.unsqueeze(0).to(device)
    goals = goals.unsqueeze(0).to(device)
    y_np = y.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    # Collect predictions
    preds = {}
    with torch.no_grad():
        for name, (model, apply_mask, uses_sdf) in models_dict.items():
            if model is None:
                continue
            chi_in = chi
            if uses_sdf and sdf_model is not None:
                chi_in = smooth_chi(mask, sdf_model(mask), 5)

            out = model(chi_in, goals)

            if apply_mask:
                out = out * mask
            preds[name] = out.squeeze().cpu().numpy()

    n_plots = 2 + len(preds)  # mask + GT + predictions
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 3.5))

    # Mask
    axes[0].imshow(mask_np, cmap='gray')
    gx, gy = goals.squeeze().cpu().numpy()
    axes[0].plot(gx, gy, 'r*', markersize=12)
    axes[0].set_title("Environment")
    axes[0].axis('off')

    # GT
    vmin, vmax = y_np.min(), y_np.max()
    axes[1].imshow(y_np, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Predictions
    for i, (name, pred) in enumerate(preds.items()):
        axes[i + 2].imshow(pred, cmap='hot', vmin=vmin, vmax=vmax)
        axes[i + 2].set_title(name)
        axes[i + 2].axis('off')

    plt.suptitle("Value Function Predictions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visual comparison: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive model comparison")
    parser.add_argument("--save_dir", type=str, default="comparison_results",
                        help="Directory to save results")
    parser.add_argument("--num_time_inst", type=int, default=100,
                        help="Number of timing iterations")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate (default: all available)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index for visual comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    save_dir = os.path.join(script_dir, args.save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    all_models = ["FNO", "DeepONet", "VIN", "DAFNO", "PNO", "PNO+PINN"]
    if args.models:
        # Map user input to canonical names
        name_map = {m.lower().replace(" ", "").replace("_", ""): m for m in all_models}
        selected = []
        for m in args.models:
            key = m.lower().replace(" ", "").replace("_", "")
            if key in name_map:
                selected.append(name_map[key])
            else:
                print(f"  [WARN] Unknown model: {m}, skipping")
        all_models = selected

    # Load FNOSDF (needed for PNO cascaded evaluation)
    sdf_model, _, _ = load_model("FNOSDF", device, results_dir)
    if sdf_model is not None:
        sdf_result = None  # will be filled per-dataset

    # Load all VF models
    models_loaded = {}
    for name in all_models:
        model, apply_mask, uses_sdf = load_model(name, device, results_dir)
        if model is not None:
            models_loaded[name] = (model, apply_mask, uses_sdf)
            n_params = count_params(model) if hasattr(model, 'parameters') and list(model.parameters()) else "-"
            print(f"  Loaded {name} ({n_params} params)")
        else:
            reason = "weights not found"
            if name == "DeepONet" and not HAS_DEEPONET:
                reason = "module import failed"
            elif name == "VIN" and not HAS_VIN:
                reason = "module import failed"
            print(f"  [SKIP] {name} - {reason}")

    print(f"\nModels loaded: {list(models_loaded.keys())}")

    # ---- Evaluate across all datasets ----
    all_results = {}  # {dataset_name: {model_name: result_dict}}
    sdf_results = {}  # {dataset_name: sdf_result_dict}

    for ds_name, ds_path, has_train in DATASETS:
        full_path = os.path.join(script_dir, "dataset", ds_path)
        if not os.path.exists(full_path):
            print(f"\n  [SKIP] Dataset {ds_name} not found at {full_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f" {ds_name} ({ds_path})")
        print(f"{'=' * 60}")

        loader, n_test = load_dataset_for_eval(full_path, has_train)
        print(f"  Test samples: {n_test}")

        dataset_results = {}

        # Evaluate FNOSDF
        if sdf_model is not None:
            sdf_res = evaluate_sdf(sdf_model, loader, device, args.num_time_inst)
            sdf_results[ds_name] = sdf_res
            print(f"  [FNOSDF] L2: {sdf_res['l2_loss']:.4f} | SDF: {sdf_res['sdf_ms']:.2f}ms")

        # Evaluate VF models
        for name, (model, apply_mask, uses_sdf) in models_loaded.items():
            sdf_for_eval = sdf_model if uses_sdf else None
            try:
                res = evaluate_model(model, name, apply_mask, loader, device,
                                     sdf_model=sdf_for_eval,
                                     num_time_inst=args.num_time_inst)
                dataset_results[name] = res
                print(f"  [{name}] L2: {res['l2_loss']:.4f} | "
                      f"SDF: {res['sdf_ms']:.2f}ms | VF: {res['vf_ms']:.2f}ms | "
                      f"Total: {res['total_ms']:.2f}ms")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                dataset_results[name] = None

        all_results[ds_name] = dataset_results

    # ---- Generate Tables ----
    print("\n" + "=" * 80)
    print(" RESULTS TABLES")
    print("=" * 80)

    # Table 1: L2 Loss (Synthetic datasets)
    synthetic_ds = [d for d in DATASETS if "synthetic" in d[1].lower()]
    synthetic_names = [d[0] for d in synthetic_ds if d[0] in all_results]
    if synthetic_names:
        headers = ["Model"] + synthetic_names
        rows = []
        for name in models_loaded:
            row = [name]
            for ds in synthetic_names:
                res = all_results.get(ds, {}).get(name)
                row.append(format_loss(res["l2_loss"] if res else None))
            rows.append(row)
        print_table("Table 1: Relative L2 Error (Synthetic)", headers, rows)
        save_csv(os.path.join(save_dir, "l2_synthetic.csv"), headers, rows)

    # Table 2: L2 Loss (City datasets)
    city_ds = [d for d in DATASETS if "city" in d[1].lower()]
    city_names = [d[0] for d in city_ds if d[0] in all_results]
    if city_names:
        headers = ["Model"] + city_names
        rows = []
        for name in models_loaded:
            row = [name]
            for ds in city_names:
                res = all_results.get(ds, {}).get(name)
                row.append(format_loss(res["l2_loss"] if res else None))
            rows.append(row)
        print_table("Table 2: Relative L2 Error (City)", headers, rows)
        save_csv(os.path.join(save_dir, "l2_city.csv"), headers, rows)

    # Table 3: Inference time (64x64)
    ds64_name = "Synthetic 64×64"
    if ds64_name in all_results:
        headers = ["Model", "Params", "L2 Loss", "SDF (ms)", "VF (ms)", "Total (ms)"]
        rows = []
        for name in models_loaded:
            model = models_loaded[name][0]
            n_params = count_params(model) if hasattr(model, 'parameters') and list(model.parameters()) else "-"
            res = all_results[ds64_name].get(name)
            if res:
                rows.append([
                    name,
                    f"{n_params:,}" if isinstance(n_params, int) else n_params,
                    format_loss(res["l2_loss"]),
                    format_time(res["sdf_ms"]),
                    format_time(res["vf_ms"]),
                    format_time(res["total_ms"]),
                ])
        print_table("Table 3: Model Summary (Synthetic 64×64)", headers, rows)
        save_csv(os.path.join(save_dir, "summary_64x64.csv"), headers, rows)

    # ---- Save raw results as JSON ----
    json_results = {}
    for ds_name, ds_res in all_results.items():
        json_results[ds_name] = {}
        for model_name, res in ds_res.items():
            json_results[ds_name][model_name] = res
    with open(os.path.join(save_dir, "all_results.json"), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved raw results: {os.path.join(save_dir, 'all_results.json')}")

    # ---- Generate Plots ----
    if not args.no_plot:
        print("\nGenerating plots...")

        plot_training_curves(
            results_dir,
            os.path.join(save_dir, "training_curves.png"),
            models_to_plot=args.models,
        )

        if ds64_name in all_results:
            plot_accuracy_vs_speed(
                all_results[ds64_name],
                os.path.join(save_dir, "accuracy_vs_speed.png"),
            )

            # Visual comparison on 64x64
            full_path_64 = os.path.join(script_dir, "dataset", "synthetic/64x64")
            loader_64, _ = load_dataset_for_eval(full_path_64, True)
            plot_visual_comparison(
                loader_64, models_loaded, sdf_model, device,
                os.path.join(save_dir, "visual_comparison.png"),
                sample_idx=args.sample_idx,
            )

    print(f"\n{'=' * 60}")
    print(f" All results saved to: {save_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
