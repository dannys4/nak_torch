# %%
import torch
from datetime import datetime

from viz_tools import animate_trajectories_box
from functions import gaussian_kde
from nak_torch.algorithms import msip_greedy

# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(314159)

# ── Config ────────────────────────────────────────────────────────────────────
save_gif = True
now_stamp = datetime.now()
algorithm_name = "msip_greedy"
function_name = "sqexp_kde"

sigma = 0.3

# ── Dataset ───────────────────────────────────────────────────────────────────
import numpy as np
npz_file = np.load('datasets/two_bananas.npz')
dataset = 10 * npz_file['X'].T - np.mean(10 * npz_file['X'].T)

log_density = gaussian_kde(sigma, dataset)
bounds = (-4.0, 4.0)

# ── Shared hyper-parameters ───────────────────────────────────────────────────
n_steps = 40
n_particles = 25
lr = 0.8
kernel_bandwidth = 0.3
bandwidth_factor = 0.001

# ── Run algorithm ─────────────────────────────────────────────────────────────
print("=== MSIP-Greedy ===")
trajectories = msip_greedy(
    log_density,
    n_particles=n_particles,
    n_steps=n_steps,
    dim=2,
    gradient_informed=True,
    lr=lr,
    noise=0.05,
    kernel_bandwidth=kernel_bandwidth,
    bandwidth_factor=bandwidth_factor,
    inner_tol=1e-6,
    max_inner_steps=5,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

# ── Save GIF ──────────────────────────────────────────────────────────────────
if save_gif:
    fpath = f"results/gif/{algorithm_name}_{function_name}_particles_{now_stamp}.gif"
    animate_trajectories_box(
        log_density, trajectories, 50, bounds,
        save_path=fpath
    )