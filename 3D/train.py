import os
import sys

# Ensure robust imports regardless of execution directory
dirname = os.path.dirname(os.path.abspath(__file__))
if dirname not in sys.path:
    sys.path.insert(0, dirname)

import argparse
import time
import torch
from torch.utils.data import DataLoader, Subset

from models.fno import FNO3d, FNO3dMultiGoal
from models.dafno import DAFNO3dMultiGoal
from models.pno import PNO3dMultiGoal
from utilities.dataset import MotionPlanningDataset3D
from utilities.losses import LpLoss, PINNLoss3d

def get_model(args, device):
    if args.model == "fno":
        return FNO3dMultiGoal(num_layers=args.nlayers, modes1=args.modes, modes2=args.modes, modes3=args.modes, width=args.width).to(device)
    elif args.model == "dafno":
        return DAFNO3dMultiGoal(num_layers=args.nlayers, modes1=args.modes, modes2=args.modes, modes3=args.modes, width=args.width).to(device)
    elif args.model == "pno":
        return PNO3dMultiGoal(num_layers=args.nlayers, modes1=args.modes, modes2=args.modes, modes3=args.modes, width=args.width).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Train 3D Operator Learning Models for Motion Planning")
    parser.add_argument("--data_dir", type=str, default="/dataset/igib-dataset-160-5G", help="Dataset directory")
    parser.add_argument("--model", type=str, choices=["fno", "dafno", "pno"], default="pno")
    parser.add_argument("--epochs", type=int, default=401)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=3e-6)
    parser.add_argument("--width", type=int, default=8, help="Hidden dimension width")
    parser.add_argument("--modes", type=int, default=3, help="Number of Fourier modes")
    parser.add_argument("--nlayers", type=int, default=1, help="Number of layers (blocks)")
    parser.add_argument("--use_pinn", action="store_true", help="Use Physics-Informed Neural Network (PINN) loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model_name_print = f"{args.model.upper()}3D w/ PINN" if args.use_pinn else args.model.upper() + "3D"
    print(f"Using device: {device}, Model: {model_name_print}")
    
    args.save_model_name = "PNO3DwPINN" if args.use_pinn and args.model == "pno" else args.model.upper() + "3D"

    # Make data_dir robust
    data_dir_clean = args.data_dir
    if data_dir_clean.startswith("3D/"):
        data_dir_clean = data_dir_clean[3:]
    full_data_dir = data_dir_clean if os.path.isabs(data_dir_clean) else os.path.join(script_dir, data_dir_clean)

    dataset = MotionPlanningDataset3D(full_data_dir)
    n_data = len(dataset)
    # Based on original script, first 320 train, last 80 test out of 400 total
    n_train = int(n_data * 0.8)
    n_test = n_data - n_train
    
    train_dataset = Subset(dataset, range(0, n_train))
    test_dataset = Subset(dataset, range(n_data - n_test, n_data))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)
    
    l2_loss_fn = LpLoss(d=3, p=2)
    pinn_loss_fn = PINNLoss3d(l2_loss_fn) if args.use_pinn else None
    
    best_test_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for mask, chi, goals, gt in train_loader:
            mask, chi, goals, gt = mask.to(device), chi.to(device), goals.to(device), gt.to(device)
            optimizer.zero_grad()
            
            # Predict
            out = model(chi, goals)
            
            # Apply mask to output and ground truth exactly like original 3D script:
            # out = out*mm, yy = yy*mm
            out = out * mask
            gt = gt * mask
            
            loss = l2_loss_fn(out, gt)
            
            if args.use_pinn:
                loss = loss + pinn_loss_fn(out, gt)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= n_train

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for mask, chi, goals, gt in test_loader:
                mask, chi, goals, gt = mask.to(device), chi.to(device), goals.to(device), gt.to(device)
                
                out = model(chi, goals)
                out = out * mask
                gt = gt * mask
                
                loss = l2_loss_fn(out, gt)
                if args.use_pinn:
                    loss = loss + pinn_loss_fn(out, gt)
                    
                test_loss += loss.item()
                
        test_loss /= n_test
        epoch_time = time.time() - start_time

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_dir = os.path.join(script_dir, "results", args.save_model_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | Train L2: {train_loss:.4f} | Test L2: {test_loss:.4f}")

    print(f"Training complete. Best Test L2 Error: {best_test_loss:.4f}")

if __name__ == "__main__":
    train()
