"""
experiment_vs_M_gi_msip_fredholm_qg.py
───────────────────────────────────────
Sweeps over M (number of particles) ∈ M_VALUES.
For each M, runs R independent initialisations of each algorithm.
Every algorithm is run to convergence (N_STEPS).
Metrics are evaluated at the final step only.

Algorithms
──────────
  SVGD | GI-ALDI | MSIP-Fredholm | MSIP-QG

Metrics  (all scalar)
─────────────────────
  MMD
  KSD-RBF, KSD-IMQ
  |Ê[f] − E_true[f]|  for f ∈ {x₁, x₂, x₁², x₂², x₁x₂}

Output
──────
  One median+IQR figure per metric  →  vsM_{metric}_median_iqr_sigma_*.pdf
"""

from functools import partial

import numpy as np
import torch
#import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

import matplotlib.pyplot as plt
font_dir = Path("/Users/ayoubbelhadji/Library/Fonts")

for font_file in font_dir.glob("*.otf"):
    fm.fontManager.addfont(str(font_file))

plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'NewComputerModern10'
plt.rcParams['font.size'] = 12

import matplotlib.pyplot as plt

#plt.rcParams.update({
#    "font.family": "serif",
#    "font.serif": ["Computer Modern Roman"],  # or "CMU Serif"
#    "mathtext.fontset": "cm",
#})

from nak_torch.algorithms import grad_aldi, msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre

# ── Device / dtype ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

M_VALUES  = [5, 10, 20, 50]
#M_VALUES  = [5, 10]
#M_VALUES  = [ 50]
R_RUNS    = 20
N_STEPS   = 1000

LR_SVGD   = 0.8
LR_ALDI   = 0.005
LR_MSIP   = 0.8

KERNEL_LS      = 0.5
KERNEL_DIAG    = 1e-8
GRADIENT_DECAY = 1.0
BOUNDS_MSIP    = (-1000.0, 1000.0)
N_QUAD         = 10

INIT_MEAN = torch.tensor([18.0, 18.0])
INIT_STD  = 1.0
BASE_SEED = 314159
SIGMA_TAG = str(KERNEL_LS)

# ══════════════════════════════════════════════════════════════════════════════
# TARGET: 5-COMPONENT ANISOTROPIC GMM
# ══════════════════════════════════════════════════════════════════════════════

gmm_weights = torch.tensor([1 / 5] * 5)

gmm_means = torch.stack([
    2 * torch.tensor([ 6.2, -6.0]),
    2 * torch.tensor([-4.0,  5.0]),
    2 * torch.tensor([ 7.0,  3.0]),
    2 * torch.tensor([-6.5, -4.5]),
    2 * torch.tensor([ 1.0,  7.0]),
])

gmm_covs = torch.stack([
    torch.tensor([[1.5,  0.1],
                  [0.1,  0.5]]),
    torch.tensor([[2.0, -0.6],
                  [-0.6, 0.5]]),
    torch.tensor([[0.7,  0.4],
                  [0.4,  1.2]]),
    torch.tensor([[1.3, -0.5],
                  [-0.5, 0.9]]),
    torch.tensor([[0.6,  0.35],
                  [0.35, 1.6]]),
])

gmm_precisions = torch.linalg.inv(gmm_covs)
gmm_logdets    = torch.stack([torch.logdet(cov) for cov in gmm_covs])

LOG_2PI = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.get_default_dtype()))


def post_log_dens(pt):
    """
    Correct normalized log density of the Gaussian mixture.
    Supports pt of shape (..., 2).
    """
    d = gmm_means.shape[1]
    log_probs = []
    for mean, prec, logdet, w in zip(gmm_means, gmm_precisions, gmm_logdets, gmm_weights):
        diff = pt - mean
        quad = torch.einsum("...i,ij,...j->...", diff, prec, diff)
        lp = torch.log(w) - 0.5 * (quad + logdet + d * LOG_2PI)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1)


post_log_dens_grad_val       = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# ── Quadrature rules ────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    dimension = dim
    N_spherical = 2 * dimension
    N_radial = int(N_quad / 4)
    pts, wts = spherical_MC_radial_Laguerre(
        batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
    )
    return pts, wts

# ══════════════════════════════════════════════════════════════════════════════
# TRUE GMM INTEGRALS  (closed form)
# ══════════════════════════════════════════════════════════════════════════════

TRUE_INTEGRALS = {
    "x1":    sum(gmm_weights[k].item() * gmm_means[k][0].item()
                 for k in range(5)),
    "x2":    sum(gmm_weights[k].item() * gmm_means[k][1].item()
                 for k in range(5)),
    "x1_sq": sum(gmm_weights[k].item() * (gmm_covs[k][0, 0] + gmm_means[k][0] ** 2).item()
                 for k in range(5)),
    "x2_sq": sum(gmm_weights[k].item() * (gmm_covs[k][1, 1] + gmm_means[k][1] ** 2).item()
                 for k in range(5)),
    "x1x2":  sum(gmm_weights[k].item() * (gmm_covs[k][0, 1] + gmm_means[k][0] * gmm_means[k][1]).item()
                 for k in range(5)),
}

TEST_FUNCTIONS = {
    "x1":    lambda pts: pts[:, 0],
    "x2":    lambda pts: pts[:, 1],
    "x1_sq": lambda pts: pts[:, 0] ** 2,
    "x2_sq": lambda pts: pts[:, 1] ** 2,
    "x1x2":  lambda pts: pts[:, 0] * pts[:, 1],
}

# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_nonnegative(weights):
    w = weights.clamp(min=0.0)
    return w / w.sum().clamp_min(1e-300)


def gaussian_rbf_expectation(mu1, cov1, mu2, cov2, bw):
    """
    E[exp(-||X-Y||² / (2 bw²))]
    for independent X ~ N(mu1,cov1), Y ~ N(mu2,cov2).
    """
    d = mu1.numel()
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=mu1.device, dtype=mu1.dtype)

    S = cov1 + cov2
    diff = mu1 - mu2

    det_term = torch.linalg.det(eye + S / sigma_sq).clamp_min(1e-300) ** (-0.5)
    quad_term = torch.exp(-0.5 * diff @ torch.linalg.solve(S + sigma_sq * eye, diff))
    return det_term * quad_term


def _gmm_rbf_cross(particles, bw):
    d = particles.shape[1]
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=particles.device, dtype=particles.dtype)

    Epp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for j in range(len(gmm_weights)):
        for l in range(len(gmm_weights)):
            Epp = Epp + gmm_weights[j] * gmm_weights[l] * gaussian_rbf_expectation(
                gmm_means[j], gmm_covs[j], gmm_means[l], gmm_covs[l], bw
            )

    Exp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for k in range(len(gmm_weights)):
        diff = particles - gmm_means[k]
        A = gmm_covs[k] + sigma_sq * eye

        det_term = torch.linalg.det(eye + gmm_covs[k] / sigma_sq).clamp_min(1e-300) ** (-0.5)
        quad = torch.einsum("ni,ij,nj->n", diff, torch.linalg.inv(A), diff)
        Exp = Exp + gmm_weights[k] * (det_term * torch.exp(-0.5 * quad)).mean()

    return Epp, Exp


def compute_mmd(particles, bw=KERNEL_LS):
    Kxx = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2)).mean()
    Epp, Exp = _gmm_rbf_cross(particles, bw)
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def compute_mmd_weighted(particles, weights, bw=KERNEL_LS):
    w = normalize_nonnegative(weights)

    K = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2))
    Kxx = (K * w[:, None] * w[None, :]).sum()

    d = particles.shape[1]
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=particles.device, dtype=particles.dtype)

    Epp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for j in range(len(gmm_weights)):
        for l in range(len(gmm_weights)):
            Epp = Epp + gmm_weights[j] * gmm_weights[l] * gaussian_rbf_expectation(
                gmm_means[j], gmm_covs[j], gmm_means[l], gmm_covs[l], bw
            )

    Exp_w = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for k in range(len(gmm_weights)):
        diff = particles - gmm_means[k]
        A = gmm_covs[k] + sigma_sq * eye

        det_term = torch.linalg.det(eye + gmm_covs[k] / sigma_sq).clamp_min(1e-300) ** (-0.5)
        quad = torch.einsum("ni,ij,nj->n", diff, torch.linalg.inv(A), diff)
        kvals = det_term * torch.exp(-0.5 * quad)
        Exp_w = Exp_w + gmm_weights[k] * (w * kvals).sum()

    return (Kxx + Epp - 2 * Exp_w).clamp(min=0.0).sqrt().item()


def _rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * KERNEL_LS ** 2))


def _imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.5)


def compute_ksd(particles, kernel_fn):
    n = particles.shape[0]
    grads, _ = post_log_dens_grad_val_batch(particles)

    xi = particles.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    xj = particles.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    si = grads.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    sj = grads.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)

    k_vals    = torch.vmap(kernel_fn)(xi, xj)
    grad_xi_k = torch.vmap(torch.func.grad(kernel_fn, argnums=0))(xi, xj)
    grad_xj_k = torch.vmap(torch.func.grad(kernel_fn, argnums=1))(xi, xj)

    def mixed_trace(a, b):
        return torch.func.jacfwd(
            torch.func.grad(kernel_fn, argnums=1), argnums=0
        )(a, b).diagonal().sum()

    trace_mixed = torch.vmap(mixed_trace)(xi, xj)

    h = (
        (si * sj).sum(-1) * k_vals
        + (sj * grad_xi_k).sum(-1)
        + (si * grad_xj_k).sum(-1)
        + trace_mixed
    )

    mask = ~torch.eye(n, dtype=torch.bool, device=particles.device)
    return h.reshape(n, n)[mask].sum().div(n * (n - 1)).clamp(min=0.0).sqrt().item()


def integral_error(fname, particles, weights=None):
    vals = TEST_FUNCTIONS[fname](particles)
    if weights is not None:
        w = normalize_nonnegative(weights)
        est = (w * vals).sum().item()
    else:
        est = vals.mean().item()
    return abs(est - TRUE_INTEGRALS[fname])

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

ALGO_NAMES = ["SVGD", "ALDI", "MSIP-Fredholm"]


#, "MSIP-GI-1"

def run_one(M: int, seed: int) -> dict:
    torch.manual_seed(seed)
    init_p = INIT_MEAN + INIT_STD * torch.randn((M, 2))
    results = {}

    traj = svgd(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_SVGD, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["SVGD"] = (traj[-1], None)

    traj = grad_aldi(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_ALDI, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["ALDI"] = (traj[-1], None)

    msip_fredholm = MSIPFredholm(
        GRADIENT_DECAY,
        post_log_dens_grad_val_batch,
    )
    traj, wts = msip(
        msip_fredholm, M, N_STEPS, dim=2,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        gradient_decay=GRADIENT_DECAY,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-Fredholm"] = (traj[-1], None)

    msip_qg = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=N_QUAD),
        GRADIENT_DECAY,
    )
    traj, wts = msip(
        msip_qg, M, N_STEPS, dim=2,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-GI-1"] = (traj[-1], None)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════

raw_results = {}
for M in M_VALUES:
    print(f"\n{'='*60}\n  M = {M} particles\n{'='*60}")
    raw_results[M] = []
    for r in range(R_RUNS):
        seed = BASE_SEED + M * 10_000 + r * 997
        print(f"  run {r + 1}/{R_RUNS}  (seed={seed})", flush=True)
        raw_results[M].append(run_one(M, seed))

print("\nSimulations done. Computing metrics …")

INTEGRAL_KEYS   = list(TEST_FUNCTIONS.keys())
DIVERGENCE_KEYS = ["mmd", "ksd_rbf", "ksd_imq"]
ALL_METRIC_KEYS = DIVERGENCE_KEYS + INTEGRAL_KEYS

metric_data = {
    mk: {algo: np.full((len(M_VALUES), R_RUNS), np.nan) for algo in ALGO_NAMES}
    for mk in ALL_METRIC_KEYS
}

for mi, M in enumerate(M_VALUES):
    for r in range(R_RUNS):
        run = raw_results[M][r]
        for algo in ALGO_NAMES:
            pts, wts = run[algo]
            is_w = wts is not None

            metric_data["mmd"][algo][mi, r] = (
                compute_mmd_weighted(pts, wts) if is_w else compute_mmd(pts)
            )
            metric_data["ksd_rbf"][algo][mi, r] = compute_ksd(pts, _rbf_fn)
            metric_data["ksd_imq"][algo][mi, r] = compute_ksd(pts, _imq_fn)

            for fk in INTEGRAL_KEYS:
                metric_data[fk][algo][mi, r] = integral_error(
                    fk, pts, wts if is_w else None
                )

        print(f"  M={M}, run {r + 1}/{R_RUNS} metrics done.", flush=True)

print("All metrics computed. Generating figures …")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING  —  one median+IQR figure per metric
# ══════════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLORS = {a: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, a in enumerate(ALGO_NAMES)}
ALGO_LS     = {
    "SVGD": "-",
    "ALDI": "-",
    "MSIP-Fredholm": "--",
    "MSIP-GI-1": "-.",
}
ALGO_MARKER = {
    "SVGD": "o",
    "ALDI": "s",
    "MSIP-Fredholm": "D",
    "MSIP-GI-1": "^",
}

M_arr = np.array(M_VALUES, dtype=float)
PLOT_EPS = 1e-16

METRIC_META = {
    "mmd"    : ("MMD",                      "MMD"),
    "ksd_rbf": ("KSD (RBF kernel)",                      "KSD_RBF"),
    "ksd_imq": ("KSD (IMQ kernel)",                      "KSD_IMQ"),
    "x1"     : (r"Integration error",          "integral_x1"),
    "x2"     : (r"$|\hat{E}[x_2] - E[x_2]|$",          "integral_x2"),
    "x1_sq"  : (r"$|\hat{E}[x_1^2] - E[x_1^2]|$",      "integral_x1sq"),
    "x2_sq"  : (r"$|\hat{E}[x_2^2] - E[x_2^2]|$",      "integral_x2sq"),
    "x1x2"   : (r"$|\hat{E}[x_1 x_2] - E[x_1 x_2]|$", "integral_x1x2"),
}

for mk, (ylabel, filetag) in METRIC_META.items():
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in ALGO_NAMES:
        arr = metric_data[mk][algo]
        med = np.maximum(np.median(arr, axis=1), PLOT_EPS)
        lo  = np.maximum(np.percentile(arr, 10, axis=1), PLOT_EPS)
        hi  = np.maximum(np.percentile(arr, 90, axis=1), PLOT_EPS)

        ax.loglog(
            M_arr, med,
            color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
            linewidth=2.0, marker=ALGO_MARKER[algo], markersize=6,
            label=algo,
        )
        ax.fill_between(M_arr, lo, hi, alpha=0.20, color=ALGO_COLORS[algo])

    ax.set_xlabel("$M$  (number of particles)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.set_title(
    #     f"{ylabel}  vs  $M$",
        
    #     fontsize=16,
    # )
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.legend(fontsize=12, ncol=2)
    plt.tight_layout()
    fname = f"vsM_{filetag}_median_iqr_sigma_{SIGMA_TAG}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

print("\nAll done.")
#\n(median ± IQR over {R_RUNS} runs)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE FINAL PARTICLES — last run, largest M
# ══════════════════════════════════════════════════════════════════════════════

try:
    M = M_VALUES[-1]
    run_idx = R_RUNS - 1

    # recycle already-computed trajectories
    final_run = raw_results[M][run_idx]

    grid_res = 250
    x_np = np.linspace(-18, 18, grid_res)
    y_np = np.linspace(-18, 18, grid_res)

    x_range = torch.tensor(x_np, dtype=torch.get_default_dtype())
    y_range = torch.tensor(y_np, dtype=torch.get_default_dtype())

    XX, YY = torch.meshgrid(x_range, y_range, indexing="ij")
    grid_pts = torch.stack([XX.flatten(), YY.flatten()], dim=1)

    log_dens_grid = torch.vmap(post_log_dens)(grid_pts).reshape(grid_res, grid_res)
    dens_grid = torch.exp(
        log_dens_grid - log_dens_grid.max()
    ).detach().cpu().numpy()

    n_algos = len(ALGO_NAMES)

    fig, axes = plt.subplots(
        1,
        n_algos,
        figsize=(5 * n_algos, 4.5),
        sharex=True,
        sharey=True,
    )

    if n_algos == 1:
        axes = [axes]

    for ax, algo_name in zip(axes, ALGO_NAMES):

        pts, wts = final_run[algo_name]
        xy = pts.detach().cpu().numpy()

        # very light background density
        ax.contourf(
            x_np,
            y_np,
            dens_grid.T,
            levels=40,
            cmap="Greys",
            alpha=0.24,
            zorder=0,
        )

        # subtle contours
        ax.contour(
            x_np,
            y_np,
            dens_grid.T,
            levels=12,
            colors="grey",
            linewidths=0.7,
            alpha=0.7,
            zorder=1,
        )

        # particles
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=38,
            c="#c1121f",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            zorder=3,
        )

        ax.set_facecolor("white")

        ax.set_xlim(-18, 18)
        ax.set_ylim(-18, 18)

        ax.set_title(
            algo_name,
            fontsize=12,
            fontweight="bold",
        )

#        ax.set_xlabel(r"$x_1$")

#    axes[0].set_ylabel(r"$x_2$")

    # fig.suptitle(
    #     f"Final particles — GMM 2D, M={M}, T={N_STEPS}, run={R_RUNS}",
    #     fontsize=14,
    #     y=1.03,
    # )

    plt.tight_layout()

    fname = (
        f"final_particles_four_algorithms_"
        f"M_{M}_T_{N_STEPS}_run_{R_RUNS}_sigma_{SIGMA_TAG}.pdf"
    )

    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    print(f"Saved {fname}")

except Exception as exc:
    print(f"Skipped final particle visualization because of: {exc}")