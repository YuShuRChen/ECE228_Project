import os
import sys

# Ensure robust imports regardless of execution directory
dirname = os.path.dirname(os.path.abspath(__file__))
if dirname not in sys.path:
    sys.path.insert(0, dirname)

import argparse
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from functools import reduce
import operator

from models.fno import FNO2d, FNO2dMultiGoal
from models.dafno import DAFNO2dMultiGoal
from models.pno import DEEPNORM2dMultiGoal
from models.deeponet import DeepONet2dMultiGoal
from models.vin import VIN2dMultiGoal
from utilities.dataset import MotionPlanningDataset
from utilities.losses import LpLoss, PINNLoss


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


def get_model(model_name, device, modes=8, width=16):
    """
    Instantiate models matching the original notebook:
      FNO2dMultiGoal(4, 1, 8, 8, 16)
      DAFNO2dMultiGoal(4, 8, 8, 16)
      DEEPNORM2dMultiGoal(4, 8, 8, 16)
      FNO2d (for SDF): same FNO2d backbone with in_channels=1, out_channels=1
    """
    if model_name == "fno":
        return FNO2dMultiGoal(num_layers=4, padding=1, modes1=modes, modes2=modes, width=width).to(device)
    elif model_name == "fnosdf":
        return FNO2d(in_channels=1, out_channels=1, width=width, modes=(modes, modes), num_blocks=4).to(device)
    elif model_name == "dafno":
        return DAFNO2dMultiGoal(num_layers=4, modes1=modes, modes2=modes, width=width).to(device)
    elif model_name == "pno":
        return DEEPNORM2dMultiGoal(num_layers=4, modes1=modes, modes2=modes, width=width).to(device)
    elif model_name == "deeponet":
        return DeepONet2dMultiGoal().to(device)
    elif model_name == "vin":
        return VIN2dMultiGoal(k_iterations=20).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model_vf(model, model_name, train_loader, test_loader, n_train, n_test,
                   use_pinn, device, save_path,
                   lr=1e-3, gamma=0.5, wd=1e-5, epochs=500, scheduler_step=50):
    """
    Train a Value Function model (FNO, DAFNO, PNO, or PNO w/ PINN).
    Matches the original notebook's train_model() function exactly.
    
    Key details matching original:
    - mask_func: lambda a: 1 for FNO (no output masking), lambda a: a for others
    - Loss: relative L2 (LpLoss d=2 p=2)
    - PINN: adds 0.05 * Eikonal gradient loss
    - Loss normalization: divide by number of samples (not batches)
    - Optimizer: Adam with weight_decay=1e-5
    - Scheduler: StepLR step_size=50 gamma=0.5
    """
    loss_func = LpLoss(d=2, p=2)
    pinn_loss_fn = PINNLoss(loss_func) if use_pinn else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=gamma)

    # Mask function: FNO does NOT mask output, DAFNO/PNO do
    apply_mask = (model_name not in ("fno", "deeponet", "vin"))

    training_losses = []
    test_losses = []
    train_time_arr = []
    best_model_weights = None
    best_training_loss = np.inf
    best_test_loss = np.inf
    best_epoch = None

    model_label = "PNOwPINN" if use_pinn else model_name.upper()
    print(f"Training model {save_path}")
    print("Epoch", "Epoch time", "Train loss", "Test loss")

    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_loss = 0.0
        samples_train = 0

        for mask, chi, goals, y, dist in train_loader:
            mask = mask.to(device)
            chi = chi.to(device)
            goals = goals.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(chi, goals)

            # Apply mask function: out * mask for DAFNO/PNO, no-op for FNO
            if apply_mask:
                out = out * mask

            # Compute loss (matching original: loss on views without channel dim)
            loss = loss_func(out, y)

            # PINN loss (Eikonal constraint)
            if use_pinn:
                loss = loss + pinn_loss_fn(out, y)

            train_loss += loss.item()
            samples_train += y.shape[0]

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        samples_test = 0
        with torch.no_grad():
            for mask, chi, goals, y, dist in test_loader:
                mask = mask.to(device)
                chi = chi.to(device)
                goals = goals.to(device)
                y = y.to(device)

                out = model(chi, goals)
                if apply_mask:
                    out = out * mask

                loss = loss_func(out, y)

                # Original notebook does NOT use PINN loss during test evaluation
                # (PINN loss is only used during training for the gradient)
                # But the notebook code DOES compute PINN loss in test loop too.
                # Matching original: include PINN in test loss
                if use_pinn:
                    loss = loss + pinn_loss_fn(out, y)

                test_loss += loss.item()
                samples_test += y.shape[0]

        # Normalize by number of samples (matching original)
        train_loss /= samples_train
        test_loss /= samples_test

        training_losses.append(train_loss)
        test_losses.append(test_loss)
        t2 = time.time()
        train_time_arr.append(t2 - t1)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_training_loss = train_loss
            best_epoch = ep
            best_model_weights = model.state_dict()

        if ep % 10 == 0:
            print(ep, t2 - t1, train_loss, test_loss)

    # Save results (matching original)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.savetxt(save_path + "/training_losses.txt", training_losses)
    np.savetxt(save_path + "/test_losses.txt", test_losses)
    np.savetxt(save_path + "/training_times.txt", train_time_arr)
    torch.save(best_model_weights, save_path + "/best_model.pt")
    with open(save_path + "/aggregate_results.txt", "w") as f:
        f.write(f"ModelParameters: {count_params(model)}\n")
        f.write(f"BestL2TrainingError: {best_training_loss}\n")
        f.write(f"BestL2TestingError: {best_test_loss}\n")
        f.write(f"BestEpoch: {best_epoch}\n")
        f.write(f"TotalTrainingTime: {np.sum(train_time_arr)}\n")

    print(f"Training complete. Best Test L2 Error: {best_test_loss:.4f} at epoch {best_epoch}")


def train_model_sdf(model, train_loader, test_loader, n_train, n_test,
                    device, save_path,
                    lr=1e-3, gamma=0.5, wd=3e-6, epochs=500, scheduler_step=50):
    """
    Train the FNO SDF model.
    Matches the original notebook's train_model_SDF() function.
    
    Key differences from VF training:
    - weight_decay = 3e-6 (not 1e-5)
    - Input: mask only, Output: dist (SDF)
    - No goals, no mask function, no PINN
    """
    loss_func = LpLoss(d=2, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=gamma)

    training_losses = []
    test_losses = []
    train_time_arr = []
    best_model_weights = None
    best_training_loss = np.inf
    best_test_loss = np.inf
    best_epoch = None

    print(f"Training model {save_path}")
    print("Epoch", "Epoch time", "Train loss", "Test loss")

    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_loss = 0.0
        samples_train = 0

        for mask, chi, goals, y, dist in train_loader:
            mask = mask.to(device)
            dist = dist.to(device)

            optimizer.zero_grad()
            out = model(mask)
            loss = loss_func(out, dist)

            train_loss += loss.item()
            samples_train += mask.shape[0]

            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        test_loss = 0.0
        samples_test = 0
        with torch.no_grad():
            for mask, chi, goals, y, dist in test_loader:
                mask = mask.to(device)
                dist = dist.to(device)

                out = model(mask)
                loss = loss_func(out, dist)

                test_loss += loss.item()
                samples_test += mask.shape[0]

        train_loss /= samples_train
        test_loss /= samples_test

        training_losses.append(train_loss)
        test_losses.append(test_loss)
        t2 = time.time()
        train_time_arr.append(t2 - t1)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_training_loss = train_loss
            best_epoch = ep
            best_model_weights = model.state_dict()

        if ep % 10 == 0:
            print(ep, t2 - t1, train_loss, test_loss)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.savetxt(save_path + "/training_losses.txt", training_losses)
    np.savetxt(save_path + "/test_losses.txt", test_losses)
    np.savetxt(save_path + "/training_times.txt", train_time_arr)
    torch.save(best_model_weights, save_path + "/best_model.pt")
    with open(save_path + "/aggregate_results.txt", "w") as f:
        f.write(f"ModelParameters: {count_params(model)}\n")
        f.write(f"BestL2TrainingError: {best_training_loss}\n")
        f.write(f"BestL2TestingError: {best_test_loss}\n")
        f.write(f"BestEpoch: {best_epoch}\n")
        f.write(f"TotalTrainingTime: {np.sum(train_time_arr)}\n")

    print(f"Training complete. Best Test L2 Error: {best_test_loss:.4f} at epoch {best_epoch}")


def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Train Operator Learning Models for Motion Planning")
    parser.add_argument("--data_dir", type=str, default="dataset/synthetic/64x64/", help="Dataset directory")
    parser.add_argument("--model", type=str, choices=["fno", "fnosdf", "dafno", "pno", "deeponet", "vin"], default="pno")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16, help="Hidden dimension width (default 16 per paper)")
    parser.add_argument("--modes", type=int, default=8, help="Number of Fourier modes (default 8 per paper)")
    parser.add_argument("--use_pinn", action="store_true", help="Use PINN (Eikonal) loss (only for pno)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Resolve data directory
    data_dir_clean = args.data_dir
    if data_dir_clean.startswith("2D/"):
        data_dir_clean = data_dir_clean[3:]
    full_data_dir = data_dir_clean if os.path.isabs(data_dir_clean) else os.path.join(script_dir, data_dir_clean)

    # Load dataset
    dataset = MotionPlanningDataset(full_data_dir)
    n_data = len(dataset)
    n_train = int(n_data * 0.8)
    n_test = n_data - n_train

    # Deterministic split: first 80% train, last 20% test (matching original)
    train_dataset = Subset(dataset, range(0, n_train))
    test_dataset = Subset(dataset, range(n_data - n_test, n_data))

    # Original notebook: shuffle=False for VF training, shuffle=True for SDF
    is_sdf = (args.model == "fnosdf")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=is_sdf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model save path
    save_name_map = {
        "fno": "FNO", "fnosdf": "FNOSDF", "dafno": "DAFNO",
        "pno": "PNO", "deeponet": "DeepONet", "vin": "VIN",
    }
    if args.use_pinn and args.model == "pno":
        save_name = "PNOwPINN"
    else:
        save_name = save_name_map.get(args.model, args.model.upper())
    save_path = os.path.join(script_dir, "results", save_name)

    model = get_model(args.model, device, modes=args.modes, width=args.width)
    print(f"Model: {save_name}, Parameters: {count_params(model)}")

    if is_sdf:
        # SDF training: weight_decay=3e-6 (matching original)
        train_model_sdf(
            model, train_loader, test_loader, n_train, n_test,
            device, save_path,
            lr=args.lr, wd=3e-6, epochs=args.epochs, scheduler_step=50)
    else:
        # VF training: weight_decay=1e-5 (matching original)
        train_model_vf(
            model, args.model, train_loader, test_loader, n_train, n_test,
            args.use_pinn, device, save_path,
            lr=args.lr, wd=1e-5, epochs=args.epochs, scheduler_step=50)


if __name__ == "__main__":
    train()