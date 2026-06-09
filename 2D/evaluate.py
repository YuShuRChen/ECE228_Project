import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
if dirname not in sys.path:
    sys.path.insert(0, dirname)

import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from models.fno import FNO2d, FNO2dMultiGoal
from models.dafno import DAFNO2dMultiGoal
from models.pno import DEEPNORM2dMultiGoal
from utilities.dataset import MotionPlanningDataset
from utilities.losses import LpLoss
from scipy.ndimage import distance_transform_edt

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


def smooth_chi(mask, dist, smooth_coef=5):
    """Smooth occupancy function, matching original notebook."""
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5


def test_model(model, model_name, apply_mask, test_loader, device,
               sdf_model=None, num_time_inst=100):
    """
    Test a model, matching the original notebook's test_model() function.
    
    Returns: (test_loss, sdf_time_ms, vf_time_ms, total_time_ms)
    
    Key details matching original:
    - Loss: LpLoss(d=2, p=2), accumulated per-sample then averaged
    - If sdf_model is provided: chi = smooth_chi(mask, sdf_model(mask), 5)
    - Output masking: out * mask for DAFNO/PNO, no-op for FNO
    - Timing: average over num_time_inst single-sample inferences
    """
    loss_func = LpLoss(d=2, p=2)
    test_loss = 0.0
    samples_test = 0

    # First pass: compute test loss
    with torch.no_grad():
        for mask, chi, goals, y, dist in test_loader:
            mask = mask.to(device)
            chi = chi.to(device)
            goals = goals.to(device)
            y = y.to(device)

            # For cascaded models, recompute chi from SDF prediction
            if sdf_model is not None:
                chi = smooth_chi(mask, sdf_model(mask), 5)

            out = model(chi, goals)
            if apply_mask:
                out = out * mask

            # LpLoss with reduce_dims=0, reductions='sum' returns sum over batch
            # Accumulate directly, divide by total samples later
            test_loss += loss_func(out, y).item()
            samples_test += y.shape[0]

    test_loss /= samples_test

    # Second pass: timing (matching original's single-sample timing loop)
    time_sdf = 0.0
    time_vf = 0.0
    with torch.no_grad():
        # Get a sample for timing
        sample_batch = next(iter(test_loader))
        mask_sample = sample_batch[0][0:1].to(device)
        chi_sample = sample_batch[1][0:1].to(device)
        goals_sample = sample_batch[2][0:1].to(device)
        mask_np = mask_sample[0, 0].detach().cpu().numpy()

        for _ in range(num_time_inst):
            # SDF computation time
            if sdf_model is not None:
                t1 = time.time()
                chi_timed = smooth_chi(mask_sample, sdf_model(mask_sample), 5)
                t2 = time.time()
            else:
                t1 = time.time()
                distance_transform_edt(mask_np)
                t2 = time.time()
                chi_timed = chi_sample
            time_sdf += (t2 - t1)

            # VF computation time
            t3 = time.time()
            out = model(chi_timed, goals_sample)
            if apply_mask:
                out = out * mask_sample
            t4 = time.time()
            time_vf += (t4 - t3)

    time_sdf = (time_sdf / num_time_inst) * 1000  # ms
    time_vf = (time_vf / num_time_inst) * 1000  # ms
    total_time = time_sdf + time_vf

    print(f"  [{model_name}] L2 Loss: {test_loss:.4f} | "
          f"SDF: {time_sdf:.2f}ms | VF: {time_vf:.2f}ms | Total: {total_time:.2f}ms")

    return test_loss, time_sdf, time_vf, total_time


def test_model_sdf(model, test_loader, device, num_time_inst=100):
    """Test the SDF model."""
    loss_func = LpLoss(d=2, p=2)
    test_loss = 0.0
    samples_test = 0

    with torch.no_grad():
        for mask, chi, goals, y, dist in test_loader:
            mask = mask.to(device)
            dist = dist.to(device)

            out = model(mask)
            test_loss += loss_func(out, dist).item()
            samples_test += mask.shape[0]

    test_loss /= samples_test

    # Timing
    time_sdf = 0.0
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        mask_sample = sample_batch[0][0:1].to(device)
        for _ in range(num_time_inst):
            t1 = time.time()
            _ = model(mask_sample)
            t2 = time.time()
            time_sdf += (t2 - t1)

    time_sdf = (time_sdf / num_time_inst) * 1000

    print(f"  [FNOSDF] L2 Loss: {test_loss:.4f} | SDF: {time_sdf:.2f}ms")
    return test_loss, time_sdf


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic/64x64")
    parser.add_argument("--num_time_inst", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Evaluating on device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_clean = args.data_dir
    if data_dir_clean.startswith("2D/"):
        data_dir_clean = data_dir_clean[3:]
    full_data_dir = data_dir_clean if os.path.isabs(data_dir_clean) else os.path.join(script_dir, "dataset", data_dir_clean)

    dataset = MotionPlanningDataset(full_data_dir)
    n_data = len(dataset)
    n_test = n_data - int(n_data * 0.8)
    test_dataset = Subset(dataset, range(n_data - n_test, n_data))
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    print("=" * 60)
    print(f"Dataset: {args.data_dir}")
    print("=" * 60)
    print(f"Test size: {n_test}")

    width = 16
    modes = 8

    # ---- FNOSDF ----
    sdf_model = None
    sdf_weight_path = os.path.join(script_dir, "results", "FNOSDF", "best_model.pt")
    if os.path.exists(sdf_weight_path):
        sdf_model = FNO2d(in_channels=1, out_channels=1, width=width,
                          modes=(modes, modes), num_blocks=4).to(device)
        sdf_model.load_state_dict(torch.load(sdf_weight_path, map_location=device, weights_only=True))
        sdf_model.eval()
        test_model_sdf(sdf_model, test_loader, device, args.num_time_inst)
    else:
        print("  [FNOSDF] Weights not found, skipping.")

    # ---- FNO ----
    # Matching train.py: FNO uses padding=1
    fno_path = os.path.join(script_dir, "results", "FNO", "best_model.pt")
    if os.path.exists(fno_path):
        model_fno = FNO2dMultiGoal(num_layers=4, padding=1, modes1=modes,
                                   modes2=modes, width=width).to(device)
        model_fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
        model_fno.eval()
        # FNO: no output masking (mask_func = lambda a: 1), no SDF model
        test_model(model_fno, "FNO", apply_mask=False, test_loader=test_loader,
                   device=device, sdf_model=None, num_time_inst=args.num_time_inst)
    else:
        print("  [FNO] Weights not found, skipping.")

    # ---- DAFNO ----
    dafno_path = os.path.join(script_dir, "results", "DAFNO", "best_model.pt")
    if os.path.exists(dafno_path):
        model_dafno = DAFNO2dMultiGoal(num_layers=4, modes1=modes,
                                       modes2=modes, width=width).to(device)
        model_dafno.load_state_dict(torch.load(dafno_path, map_location=device, weights_only=True))
        model_dafno.eval()
        # DAFNO: output masking, no SDF model (uses GT chi)
        test_model(model_dafno, "DAFNO", apply_mask=True, test_loader=test_loader,
                   device=device, sdf_model=None, num_time_inst=args.num_time_inst)
    else:
        print("  [DAFNO] Weights not found, skipping.")

    # ---- PNO ----
    pno_path = os.path.join(script_dir, "results", "PNO", "best_model.pt")
    if os.path.exists(pno_path):
        model_pno = DEEPNORM2dMultiGoal(num_layers=4, modes1=modes,
                                        modes2=modes, width=width).to(device)
        model_pno.load_state_dict(torch.load(pno_path, map_location=device, weights_only=True))
        model_pno.eval()
        # PNO: output masking, uses SDF model for cascaded evaluation
        test_model(model_pno, "PNO", apply_mask=True, test_loader=test_loader,
                   device=device, sdf_model=sdf_model, num_time_inst=args.num_time_inst)
    else:
        print("  [PNO] Weights not found, skipping.")

    # ---- PNO w/ PINN ----
    pnopinn_path = os.path.join(script_dir, "results", "PNOwPINN", "best_model.pt")
    if os.path.exists(pnopinn_path):
        model_pnopinn = DEEPNORM2dMultiGoal(num_layers=4, modes1=modes,
                                            modes2=modes, width=width).to(device)
        model_pnopinn.load_state_dict(torch.load(pnopinn_path, map_location=device, weights_only=True))
        model_pnopinn.eval()
        # PNOwPINN: same as PNO, with SDF model
        test_model(model_pnopinn, "PNOwPINN", apply_mask=True, test_loader=test_loader,
                   device=device, sdf_model=sdf_model, num_time_inst=args.num_time_inst)
    else:
        print("  [PNOwPINN] Weights not found, skipping.")

    # ---- DeepONet ----
    deeponet_path = os.path.join(script_dir, "results", "DeepONet", "best_model.pt")
    if HAS_DEEPONET and os.path.exists(deeponet_path):
        model_deeponet = DeepONet2dMultiGoal().to(device)
        model_deeponet.load_state_dict(torch.load(deeponet_path, map_location=device, weights_only=True))
        model_deeponet.eval()
        test_model(model_deeponet, "DeepONet", apply_mask=False, test_loader=test_loader,
                   device=device, sdf_model=None, num_time_inst=args.num_time_inst)
    else:
        print("  [DeepONet] Weights not found or module unavailable, skipping.")

    # ---- VIN ----
    vin_path = os.path.join(script_dir, "results", "VIN", "best_model.pt")
    if HAS_VIN and os.path.exists(vin_path):
        model_vin = VIN2dMultiGoal(k_iterations=20).to(device)
        model_vin.load_state_dict(torch.load(vin_path, map_location=device, weights_only=True))
        model_vin.eval()
        test_model(model_vin, "VIN", apply_mask=False, test_loader=test_loader,
                   device=device, sdf_model=None, num_time_inst=args.num_time_inst)
    else:
        print("  [VIN] Weights not found or module unavailable, skipping.")


if __name__ == "__main__":
    evaluate()
