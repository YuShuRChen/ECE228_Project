import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import *

# Output directory for images and animations
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

risk_cmap = get_risk_colormap()

METHOD = "sliding_window"
MODEL = "new"

# --- 1. Load Map & Weights ---
MAP_SIZE = 64
open_cell_dist = {64:3.0, 256: 15.0}
data_dir = f"dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"
if not os.path.exists(data_dir):
    data_dir = f"examples/dataset/synthetic/{MAP_SIZE}x{MAP_SIZE}/"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset not found at {data_dir}")

MAP_IDX = 2
actual_masks = np.load(os.path.join(data_dir, "mask.npy"))
dist_maps = np.load(os.path.join(data_dir, "dist_in.npy"))
binary_map = actual_masks[MAP_IDX]
static_dist = dist_maps[MAP_IDX]
H, W = binary_map.shape
print(f"Map size: {H}x{W}")

# Precompute Static Risk
SIGMA_STATIC = 1.5
static_risk = np.exp(-(static_dist ** 2) / (2 * SIGMA_STATIC ** 2))
static_risk = (static_risk - static_risk.min()) / (static_risk.max() - static_risk.min() + 1e-8)

# --- 2. Initialize Scenario ---
PERC_OPEN_CELLS = 0.02
SIGMA_DYNAMIC = 1.0
ALPHA_DYNAMIC = 1.0
DYNAMIC_SPEED = 1.0

# Obstacle movement directions
DIRECTIONS = np.array([
    [0, DYNAMIC_SPEED], [DYNAMIC_SPEED, 0], [0, -DYNAMIC_SPEED], [-DYNAMIC_SPEED, 0],
    [DYNAMIC_SPEED, DYNAMIC_SPEED], [DYNAMIC_SPEED, -DYNAMIC_SPEED],
    [-DYNAMIC_SPEED, DYNAMIC_SPEED], [-DYNAMIC_SPEED, -DYNAMIC_SPEED]]).astype(float)

SEED = 0
np.random.seed(SEED)

open_cells = np.argwhere(static_dist > open_cell_dist[MAP_SIZE])

num_open_cells = open_cells.shape[0]
print(num_open_cells, "open cells")
NUM_OBSTACLES = math.floor(num_open_cells * PERC_OPEN_CELLS)
print(NUM_OBSTACLES, "obstacles")

# Spawn points
random_open_cells = np.random.choice(len(open_cells), NUM_OBSTACLES+1, replace=False)

agent_pos_init = open_cells[random_open_cells[0]]
goal_maps = np.load(os.path.join(data_dir, "goal.npy"))
goal_data = goal_maps[MAP_IDX][::-1] # Flip (col, row) to (row, col)

obs_indices = random_open_cells[1:]
obs_positions = [open_cells[idx].astype(float) for idx in obs_indices]
obs_velocities = [DIRECTIONS[np.random.choice(len(DIRECTIONS))] for _ in range(NUM_OBSTACLES)]


# --- . Global PNO Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Models
modelSDF, modelPNO = load_new_model(device)

# Pre-compute CHI (Smoothed SDF) and mask tensor
mask_tensor = torch.tensor(binary_map, dtype=torch.float).reshape(1, H, W, 1).to(device)
with torch.no_grad():
    sdf_pred = modelSDF(mask_tensor)
    chi_tensor = smooth_chi(mask_tensor, sdf_pred, 5.0)
print("Global setup complete.")

# --- 3. Static Path Planning (Global Guide) ---
print("Planning global static path...")
goal_coord_static = torch.tensor(np.array([goal_data]), dtype=torch.int).to(device)
cost_map_static = run_pno_inference(modelPNO, static_risk, chi_tensor, goal_coord_static, mask_tensor, device)
global_path, _ = plan_path_astar(agent_pos_init, goal_data, static_risk, cost_map_static, risk_weight=10.0)

if len(global_path) == 0:
    print("Warning: No global static path found! The goal may be unreachable.")

# --- 4. Run Advanced Dynamic Window Simulation ---
print("Running Dynamic Window Simulation (Path-Guided)...")
plan_data = run_dynamic_window_simulation(obs_positions, obs_velocities, agent_pos_init, goal_data,
                                          modelPNO, chi_tensor, mask_tensor, device,
                                          static_risk, binary_map, H, W,
                                          SIGMA_DYNAMIC, ALPHA_DYNAMIC, global_path=global_path)

# Save Evaluation Metrics
output_dir = os.path.join(OUTPUT_DIR, str(MAP_SIZE), f"{METHOD}_{MODEL}")
save_metrics(plan_data['metrics'], output_dir, f"{MAP_IDX}.json")

# --- 5. Animation ---
fig, (ax_bin, ax_risk) = plt.subplots(1, 2, figsize=(14, 6))

# Binary Plot - Matching demo.py last plot style (BLUE)
ax_bin.imshow(binary_map, origin='lower', cmap='gray')
gp_line_bin, = ax_bin.plot([], [], 'b--', alpha=0.5, label='Global Plan')
obs_dots_bin, = ax_bin.plot([], [], 'ko', markersize=6)
agent_dot_bin, = ax_bin.plot([], [], 'bs', markersize=6)
ax_bin.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_bin.set_title("Binary Environment")
ax_bin.axis('off')

# Risk Plot - Matching demo.py last plot style (BLUE)
im_risk_dyn = ax_risk.imshow(plan_data['risk'][0], origin='lower', cmap=risk_cmap, vmin=0, vmax=1.0)
path_line, = ax_risk.plot([], [], 'b-', linewidth=2)
obs_dots_risk, = ax_risk.plot([], [], 'ko', markersize=6)
agent_dot_risk, = ax_risk.plot([], [], 'bs', markersize=6)
ax_risk.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_risk.set_title("Dynamic Neural Planning")
ax_risk.axis('off')

artists = {
    'im_risk': im_risk_dyn,
    'obs_dots_bin': obs_dots_bin,
    'obs_dots_risk': obs_dots_risk,
    'agent_dot_bin': agent_dot_bin,
    'agent_dot_risk': agent_dot_risk,
    'path_line': path_line,
    'global_path_bin': gp_line_bin
}

update_fn = partial(animation_step, data_dict=plan_data, artists=artists)

ani = FuncAnimation(fig, update_fn, frames=len(plan_data['risk']), interval=100, blit=False)
gif_path = os.path.join(output_dir, f"{MAP_IDX}.gif")
print(f"Saving advanced planning animation to {gif_path}...")
ani.save(gif_path, writer=PillowWriter(fps=10))
plt.close()

print(f"Done! Result saved to {gif_path}")
