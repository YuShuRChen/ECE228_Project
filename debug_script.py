import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt
import os

# Mocking data if not present
H, W = 64, 64
binary_map = np.ones((H, W))
static_dist = np.full((H, W), 10.0)
static_risk = np.zeros((H, W))

NUM_OBSTACLES = 3
TOTAL_STEPS = 5
SIGMA_DYNAMIC = 4.0
ALPHA_DYNAMIC = 1.5

DIRECTIONS = np.array([[0, 2], [2, 0], [0, -2], [-2, 0]])
obs_positions = [np.array([32.0, 32.0]) for _ in range(NUM_OBSTACLES)]
obs_velocities = [DIRECTIONS[0] for _ in range(NUM_OBSTACLES)]

colors = ["blue", "green", "yellow", "red"]
risk_cmap = LinearSegmentedColormap.from_list("risk_map", colors)

def compute_dynamic_risk(obs_pos, obs_vel, H, W, sigma, alpha):
    grid = np.zeros((H, W))
    grid[int(obs_pos[0]), int(obs_pos[1])] = 1
    dyn_dist = distance_transform_edt(1 - grid)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    dx, dy = xs - obs_pos[1], ys - obs_pos[0]
    vel_norm = np.linalg.norm(obs_vel) + 1e-6
    d_norm   = np.sqrt(dx**2 + dy**2) + 1e-6
    cos_theta = (dx * obs_vel[1] + dy * obs_vel[0]) / (vel_norm * d_norm)
    w = 1 + alpha * cos_theta
    dir_dist = dyn_dist / w
    risk = np.exp(-(dir_dist**2) / (2 * sigma**2))
    return (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)

def compute_total_risk(obs_positions, obs_velocities, static_risk, H, W):
    total = static_risk.copy()
    for pos, vel in zip(obs_positions, obs_velocities):
        total = np.clip(total + compute_dynamic_risk(pos, vel, H, W, SIGMA_DYNAMIC, ALPHA_DYNAMIC), 0, 1)
    return total

def step_obstacles(obs_positions, obs_velocities, H, W):
    for i in range(len(obs_positions)):
        next_pos = obs_positions[i] + obs_velocities[i]
        if next_pos[0] < 0 or next_pos[0] >= H: obs_velocities[i][0] *= -1
        if next_pos[1] < 0 or next_pos[1] >= W: obs_velocities[i][1] *= -1
        obs_positions[i] = np.clip(obs_positions[i] + obs_velocities[i], 0, [H-1, W-1])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(binary_map, origin='lower', cmap='gray')
obs_dots_binary, = ax1.plot([], [], 'ro')
initial_risk = compute_total_risk(obs_positions, obs_velocities, static_risk, H, W)
im_risk = ax2.imshow(initial_risk, origin='lower', cmap=risk_cmap, vmin=-0.2, vmax=1.0)
obs_dots_risk, = ax2.plot([], [], 'wo')
contour_set = [ax2.contour(initial_risk, colors='white', alpha=0.2, levels=8, origin='lower')]

def update(frame):
    global obs_positions, obs_velocities
    risk_data = compute_total_risk(obs_positions, obs_velocities, static_risk, H, W)
    pos_array = np.array(obs_positions)
    obs_dots_binary.set_data(pos_array[:, 1], pos_array[:, 0])
    im_risk.set_data(risk_data)
    obs_dots_risk.set_data(pos_array[:, 1], pos_array[:, 0])
    for c in contour_set[0].collections: c.remove()
    contour_set[0] = ax2.contour(risk_data, colors='white', alpha=0.2, levels=8, origin='lower')
    step_obstacles(obs_positions, obs_velocities, H, W)
    return [im_risk, obs_dots_binary, obs_dots_risk] + contour_set[0].collections

ani = FuncAnimation(fig, update, frames=TOTAL_STEPS, interval=100, blit=False)
print("Animation initialization successful")
