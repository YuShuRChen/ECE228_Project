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

METHOD = "whole_map"
MODEL = "original"

# --- 1. Load Static Map ---
MAP_SIZE = 64
open_cell_dist = {64: 3.0, 256: 15.0}
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

# Add risk to static map
SIGMA_STATIC = 1.0
static_risk = np.exp(-(static_dist ** 2) / (2 * SIGMA_STATIC ** 2))
static_risk = (static_risk - static_risk.min()) / (static_risk.max() - static_risk.min() + 1e-8)

# --- 2. Settings & Initializations ---
PERC_OPEN_CELLS = 0.02
TOTAL_STEPS = 50
SIGMA_DYNAMIC = 1.0
ALPHA_DYNAMIC = 0.3
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
goal_data = goal_maps[MAP_IDX][::-1]

obs_indices = random_open_cells[1:]
obs_positions = [open_cells[idx].astype(float) for idx in obs_indices]
obs_velocities = [DIRECTIONS[np.random.choice(len(DIRECTIONS))] for _ in range(NUM_OBSTACLES)]

# --- 4. Global PNO Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Models
modelSDF, modelPNO = load_pno_models(device)

# Pre-compute SDF and mask tensor
mask_tensor = torch.tensor(binary_map, dtype=torch.float).reshape(1, H, W, 1).to(device)
with torch.no_grad():
    sdf_pred = modelSDF(mask_tensor)
    chi_tensor = smooth_chi(mask_tensor, sdf_pred, 5.0)
print("Global setup complete.")

# --- 5. Static Planning with Pre-trained PNO ---

# 2. A* Planning
print("Planning global static path...")
goal_coord_static = torch.tensor(np.array([goal_data]), dtype=torch.int).to(device)
cost_map_static = run_pno_inference(modelPNO, static_risk, chi_tensor, goal_coord_static, mask_tensor, device)
global_path, _ = plan_path_astar(agent_pos_init, goal_data, static_risk, cost_map_static, risk_weight=10.0)

if len(global_path) == 0:
    print("Warning: No global static path found! The goal may be unreachable.")

# --- 6. Dynamic Planning Simulation ---
plan_data = run_planning_simulation(obs_positions, obs_velocities, agent_pos_init, goal_data,
                                    modelPNO, chi_tensor, mask_tensor, device,
                                    static_risk, binary_map, H, W,
                                    SIGMA_DYNAMIC, ALPHA_DYNAMIC, global_path=global_path)

# Save Evaluation Metrics
output_dir = os.path.join(OUTPUT_DIR, str(MAP_SIZE), f"{METHOD}_{MODEL}")
save_metrics(plan_data['metrics'], output_dir, f"{MAP_IDX}.json")

# Animation Setup
fig_dyn, (ax_bin, ax_risk) = plt.subplots(1, 2, figsize=(14, 6))

ax_bin.imshow(binary_map, origin='lower', cmap='gray')
gp_line_bin, = ax_bin.plot([], [], 'b--', alpha=0.5, label='Global Plan')
obs_dots_bin, = ax_bin.plot([], [], 'ko', markersize=6)
agent_dot_bin, = ax_bin.plot([], [], 'bs', markersize=6)
ax_bin.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_bin.set_title("Binary Environment")
ax_bin.axis('off')

im_risk_dyn = ax_risk.imshow(plan_data['risk'][0], origin='lower', cmap=risk_cmap, vmin=0, vmax=1.0)
path_line, = ax_risk.plot([], [], 'b-', linewidth=2)
obs_dots_risk, = ax_risk.plot([], [], 'ko', markersize=6)
agent_dot_risk, = ax_risk.plot([], [], 'bs', markersize=6)
ax_risk.plot(goal_data[1], goal_data[0], 'b*', markersize=10)
ax_risk.set_title("Dynamic Neural Planning")
ax_risk.axis('off')

artists_dyn = {
    'im_risk': im_risk_dyn,
    'obs_dots_bin': obs_dots_bin,
    'obs_dots_risk': obs_dots_risk,
    'agent_dot_bin': agent_dot_bin,
    'agent_dot_risk': agent_dot_risk,
    'path_line': path_line,
    'global_path_bin': gp_line_bin
}

update_dyn_fn = partial(animation_step, data_dict=plan_data, artists=artists_dyn)

ani_dyn = FuncAnimation(fig_dyn, update_dyn_fn, frames=len(plan_data['risk']), interval=150, blit=False)
gif_path = os.path.join(output_dir, f"{MAP_IDX}.gif")
print(f"Saving planning animation to {gif_path}...")
ani_dyn.save(gif_path, writer=PillowWriter(fps=7))
plt.close()
print("Dynamic planning simulation complete.")
