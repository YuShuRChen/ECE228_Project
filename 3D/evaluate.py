import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
if dirname not in sys.path:
    sys.path.insert(0, dirname)

import argparse
import time
import torch
from torch.utils.data import DataLoader, Subset

from models.fno import FNO3dMultiGoal
from models.dafno import DAFNO3dMultiGoal
from models.pno import PNO3dMultiGoal
from utilities.dataset import MotionPlanningDataset3D
from utilities.losses import LpLoss

def get_model(model_name, device):
    width = 8
    modes = 3
    nlayers = 1
    
    if model_name == "fno":
        return FNO3dMultiGoal(num_layers=nlayers, modes1=modes, modes2=modes, modes3=modes, width=width).to(device)
    elif model_name == "fnosdf":
        from models.fno import FNO3d
        # FNOSDF predicts 1 channel SDF from 1 channel mask
        return FNO3d(in_channels=1, out_channels=1, modes1=modes, modes2=modes, modes3=modes, width=width).to(device)
    elif model_name == "dafno":
        return DAFNO3dMultiGoal(num_layers=nlayers, modes1=modes, modes2=modes, modes3=modes, width=width).to(device)
    elif model_name == "pno" or model_name == "pnowpinn":
        return PNO3dMultiGoal(num_layers=nlayers, modes1=modes, modes2=modes, modes3=modes, width=width).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def smooth_chi(mask, dist, smooth_coef=5):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/dataset/igib-dataset-160-5G")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Evaluating on device: {device}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_clean = args.data_dir
    if data_dir_clean.startswith("3D/"):
        data_dir_clean = data_dir_clean[3:]
    full_data_dir = data_dir_clean if os.path.isabs(data_dir_clean) else os.path.join(script_dir, data_dir_clean)

    dataset = MotionPlanningDataset3D(full_data_dir)
    n_data = len(dataset)
    n_test = n_data - int(n_data * 0.8)
    test_dataset = Subset(dataset, range(n_data - n_test, n_data))
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    l2_loss_fn = LpLoss(d=3, p=2)

    model_names = ["fno", "fnosdf", "dafno", "pno", "pnowpinn"]
    save_names = ["FNO3D", "FNOSDF3D", "DAFNO3D", "PNO3D", "PNO3DwPINN"]

    print("="*60)
    print(f"Dataset: {args.data_dir}")
    print("="*60)
    print(f"Test size: {n_test}")

    sdf_model = get_model("fnosdf", device)
    sdf_weight_path = os.path.join(script_dir, "results", "FNOSDF3D", "best_model.pt")
    if os.path.exists(sdf_weight_path):
        sdf_model.load_state_dict(torch.load(sdf_weight_path, map_location=device))
        sdf_model.eval()
    else:
        sdf_model = None

    for m_name, s_name in zip(model_names, save_names):
        model = get_model(m_name, device)
        weight_path = os.path.join(script_dir, "results", s_name, "best_model.pt")
        
        if not os.path.exists(weight_path):
            print(f"  [{s_name}] Weights not found, skipping.")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        
        test_loss_oracle = 0.0
        test_loss_cascade = 0.0
        total_vf_time = 0.0
        total_sdf_time = 0.0
        n_samples = 0
        
        with torch.no_grad():
            # 3D dataset returns 4 elements usually. We assume dataset yields mask, chi, goals, gt, dist
            # Or if it doesn't have dist, we can't evaluate fnosdf.
            # Let's check what MotionPlanningDataset3D yields. It yields (mask, chi, goals, y)
            for batch in test_loader:
                # To support both 4-element and 5-element loaders
                if len(batch) == 5:
                    mask, chi, goals, gt, dist = batch
                else:
                    mask, chi, goals, gt = batch
                    dist = None  # dist not returned by current 3D dataset?

                mask, chi, goals, gt = mask.to(device), chi.to(device), goals.to(device), gt.to(device)
                if dist is not None: dist = dist.to(device)
                
                bs = mask.shape[0]
                n_samples += bs
                
                if m_name == "fnosdf":
                    if dist is None: continue
                    start_sdf = time.time()
                    out = model(mask)
                    total_sdf_time += (time.time() - start_sdf)
                    loss = l2_loss_fn(out, dist)
                    test_loss_oracle += loss.item() * bs
                    test_loss_cascade += loss.item() * bs
                else:
                    # 1. ORACLE
                    out_oracle = model(chi, goals)
                    if m_name != "fno": out_oracle = out_oracle * mask
                    test_loss_oracle += l2_loss_fn(out_oracle, gt).item() * bs

                    # 2. CASCADE
                    start_sdf = time.time()
                    if sdf_model is not None:
                        pred_dist = sdf_model(mask)
                        pred_chi = smooth_chi(mask, pred_dist, 5)
                    else:
                        pred_chi = chi
                    total_sdf_time += (time.time() - start_sdf)

                    start_vf = time.time()
                    out_cascade = model(pred_chi, goals)
                    total_vf_time += (time.time() - start_vf)
                    
                    if m_name != "fno": out_cascade = out_cascade * mask
                    test_loss_cascade += l2_loss_fn(out_cascade, gt).item() * bs
                    
        if n_samples == 0: continue
        
        test_loss_oracle /= n_samples
        test_loss_cascade /= n_samples
        
        avg_sdf_time = (total_sdf_time / n_test) * 1000
        avg_vf_time = (total_vf_time / n_test) * 1000
        total_time = avg_sdf_time + avg_vf_time
        
        if m_name == "fnosdf":
            print(f"  [{s_name}] L2 Loss: {test_loss_oracle:.4f} | SDF Time: {avg_sdf_time:.2f}ms | VF Time: {avg_vf_time:.2f}ms | Total Time: {total_time:.2f}ms")
        else:
            print(f"  [{s_name}] Oracle L2: {test_loss_oracle:.4f} | Cascaded L2: {test_loss_cascade:.4f} | SDF Time: {avg_sdf_time:.2f}ms | VF Time: {avg_vf_time:.2f}ms | Total Time: {total_time:.2f}ms")

if __name__ == "__main__":
    evaluate()
