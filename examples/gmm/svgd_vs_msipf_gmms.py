# %%
# ─────────────────────────────────────────────────────────────────────────────
# MSIP-Fredholm vs SVGD on a rotated 3-component GMM.
#
# Structure mirrors examples/gmm/v0/msip-vs-msipgs_on_gmms.py (same target, same
# closed-form MMD-to-GMM diagnostic and 2D particle plots), but the two methods
# being compared are MSIP-Fredholm and SVGD, instantiated exactly as in
# examples/gmm/all-gi-vs-msip_algos_comparison.py.
# ─────────────────────────────────────────────────────────────────────────────
from functools import partial

import math
import torch
import matplotlib

matplotlib.use("Agg")  # file output only; drop this line for interactive use
import matplotlib.pyplot as plt

from nak_torch.algorithms import svgd, msip
from nak_torch.algorithms.msip import MSIPFredholm

# ── Device / dtype ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(1023921)

# ── Target: rotated 3-component GMM (identical to the v0 file) ────────────────


gmm_weights = torch.tensor([1 / 5] * 5, dtype=torch.get_default_dtype())

gmm_means = torch.stack([
    2 * torch.tensor([ 6.2, -6.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([-4.0,  5.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([ 7.0,  3.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([-6.5, -4.5], dtype=torch.get_default_dtype()),
    2 * torch.tensor([ 1.0,  7.0], dtype=torch.get_default_dtype()),
])

gmm_covs = torch.stack([
    torch.tensor([[1.5,  0.1], [0.1,  0.5]], dtype=torch.get_default_dtype()),
    torch.tensor([[2.0, -0.6], [-0.6, 0.5]], dtype=torch.get_default_dtype()),
    torch.tensor([[0.7,  0.4], [0.4,  1.2]], dtype=torch.get_default_dtype()),
    torch.tensor([[1.3, -0.5], [-0.5, 0.9]], dtype=torch.get_default_dtype()),
    torch.tensor([[0.6,  0.35], [0.35, 1.6]], dtype=torch.get_default_dtype()),
])


# gmm_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
# gmm_means = torch.stack(
#     [
#         torch.tensor([6.2, -6.0]),
#         torch.tensor([-4.0, 5.0]),
#         torch.tensor([7.0, 0.0]),
#     ]
# )
#cov_rng = torch.Generator(torch.get_default_device())
#cov_rng.manual_seed(0)
#gmm_cov_matrices = torch.stack(
#    [torch.nn.init.orthogonal_(torch.empty((2, 2)), generator=cov_rng) for _ in range(3)]
#
#gmm_cov_eigvals = torch.stack([torch.diag(torch.as_tensor([2.0, 1.0])) for _ in range(3)])
#gmm_covs = torch.einsum(
#    "kij,kjl,kml->kim", gmm_cov_matrices, gmm_cov_eigvals, gmm_cov_matrices
#)
gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


# Batched (grad, value) of log-density — required by MSIP-Fredholm.
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
n_steps = 10000
n_particles = 1000

lr_svgd = 0.8  # SVGD step size (as in all-gi-vs-msip_algos_comparison.py)
lr_msip = 0.8  # MSIP-Fredholm step size
kernel_length_scale = 0.1  # kernel bandwidth for the *dynamics* (both methods)
kernel_diag_infl = 1e-8
gradient_decay = 1.0  # gamma in MSIP-Fredholm (multiplies the score)
bounds = (-100.0, 100.0)

# Bandwidth for the MMD *metric*, kept separate from the dynamics bandwidth.
# At the well-separated modes of this target, the dynamics value (0.1) makes the
# unweighted RBF-MMD^2 go negative and clamp to 0 (a degenerate, misleading
# diagnostic); ~ the within-mode scale is far more informative. Set this equal to
# kernel_length_scale to reproduce the exact single-bandwidth v0 convention.
mmd_bandwidth = 0.1

# Shared initial cloud so the two methods start from exactly the same particles.
init_mean = torch.tensor([0.0, 0.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))

# ══════════════════════════════════════════════════════════════════════════════
# Run the two algorithms
# ══════════════════════════════════════════════════════════════════════════════
print("=== SVGD ===")
trajectories_svgd = svgd(
    post_log_dens,
    n_particles,
    n_steps,
    dim=2,
    lr=lr_svgd,
    init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm,
    n_particles,
    n_steps,
    dim=2,
    lr=lr_msip,
    init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Closed-form MMD to the target GMM (RBF kernel), identical to the v0 file.
#
# Note on weights: MSIP-Fredholm returns *signed* particle weights. By default
# this diagnostic treats every method's particles uniformly (weights=None), so
# the MMD is directly comparable to the other examples/gmm scripts, which all use
# exactly this unweighted estimator. To score MSIP-Fredholm's actual signed
# empirical measure instead, pass its per-step weights (see the commented line in
# the loop below).
# ══════════════════════════════════════════════════════════════════════════════
def gmm_rbf_expectations(particles, weights, means, covs, bandwidth, p_wts=None):
    """Return (E_{y,y'~pi}[k], E_{x~mu, y~pi}[k]) for the RBF kernel k.

    p_wts: optional particle weights for mu (defaults to uniform 1/N).
    """
    sigma_sq = bandwidth**2
    K, D = means.shape
    
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    log_C = 0.5 * D * math.log(2 * math.pi * sigma_sq)   
    
    
    # E_{y,y'~pi}[k(y,y')] — analytic, independent of the particles.
    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for ell in range(K):
            cov_jl = covs[j] + covs[ell] + sigma_sq * eye
            diff_jl = means[j] - means[ell]
            log_k = -0.5 * (
                diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                + torch.logdet(2 * torch.pi * cov_jl)
            ) + log_C 
            Epp = Epp + weights[j] * weights[ell] * log_k.exp()

    # E_{x~mu, y~pi}[k(x,y)] — mixes particles with the target.
    smoothed_covs = covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - means[k].unsqueeze(0)
        log_k = -0.5 * (
            torch.einsum(
                "ni,ij,nj->n", diff_nk, torch.linalg.inv(smoothed_covs[k]), diff_nk
            )
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        ) + log_C  
        contrib = log_k.exp()
        contrib = (contrib * p_wts).sum() if p_wts is not None else contrib.mean()
        Exp = Exp + weights[k] * contrib
    return Epp, Exp


def compute_mmd(particles, bandwidth, weights=None):
    """sqrt(MMD^2) between the particle measure and the target GMM.

    weights=None  -> uniform particles (default; matches the other examples).
    weights given -> uses that (possibly signed) empirical measure; normalized
                     to sum to 1 first.
    """
    if weights is not None:
        w = weights / weights.sum()
        Epp, Exp = gmm_rbf_expectations(
            particles, gmm_weights, gmm_means, gmm_covs, bandwidth, p_wts=w
        )
        Kmat = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bandwidth**2))
        Kxx = w @ Kmat @ w
    else:
        Epp, Exp = gmm_rbf_expectations(
            particles, gmm_weights, gmm_means, gmm_covs, bandwidth
        )
        Kxx = torch.exp(
            -torch.cdist(particles, particles).pow(2) / (2 * bandwidth**2)
        ).mean()
    return (Kxx + Epp - 2 * Exp).sqrt().item()
#.clamp(min=0.0).sqrt().item()


# ── MMD vs iteration ─────────────────────────────────────────────────────────
mmd_every = 10
T = min(len(trajectories_svgd), len(trajectories_msip_f))
steps_recorded = list(range(0, T, mmd_every))

mmd_svgd, mmd_msip_f = [], []
print("Computing MMD …")
for i in steps_recorded:
    mmd_svgd.append(compute_mmd(trajectories_svgd[i], mmd_bandwidth))
    mmd_msip_f.append(compute_mmd(trajectories_msip_f[i], mmd_bandwidth))
    # Weighted (signed-measure) variant for MSIP-Fredholm instead of the line above:
    # mmd_msip_f.append(compute_mmd(trajectories_msip_f[i], mmd_bandwidth,
    #                               weights=traj_wts_msip_f[i]))
print(f"final MMD  SVGD={mmd_svgd[-1]:.4e}   MSIP-Fredholm={mmd_msip_f[-1]:.4e}")

# ── Target contour grid ──────────────────────────────────────────────────────
Ngrid = 200
xgrid = torch.linspace(-25, 25, Ngrid)
ygrid = torch.linspace(-25, 25, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

# ── Final particle clouds (MSIP-Fredholm coloured by |weight|) ───────────────
pts_svgd = trajectories_svgd[-1]
pts_msip_f = trajectories_msip_f[-1]

wts_msip_f = traj_wts_msip_f[-1]
wts_msip_f_n = wts_msip_f / wts_msip_f.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20, alpha=0.6)
axes[0].scatter(pts_svgd[:, 0].cpu(), pts_svgd[:, 1].cpu(), s=25, alpha=0.6, color="C0")
axes[0].set_title(f"SVGD – step {n_steps}")

sc = axes[1].scatter(
    pts_msip_f[:, 0].cpu(),
    pts_msip_f[:, 1].cpu(),
    c=wts_msip_f_n.abs().cpu(),
    cmap="viridis",
    s=25,
    alpha=0.7,
)
axes[1].contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20, alpha=0.6)
plt.colorbar(sc, ax=axes[1], shrink=0.8, label="|weight|")
axes[1].set_title(f"MSIP-Fredholm – step {n_steps}")

for ax in axes:
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect(1.0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
plt.tight_layout()
plt.savefig("2dplot_GMM_MSIP-Fredholm-vs-SVGD.pdf")
plt.close()
print("Saved 2dplot_GMM_MSIP-Fredholm-vs-SVGD.pdf")

# ── MMD curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(steps_recorded, mmd_svgd, label="SVGD", marker="o", ms=3)
ax.semilogy(steps_recorded, mmd_msip_f, label="MSIP-Fredholm", marker="s", ms=3)
ax.set_xlabel("Iteration")
ax.set_ylabel("MMD to target GMM (RBF kernel)")
ax.set_title(f"MMD vs iteration — MSIP-Fredholm vs SVGD (metric σ={mmd_bandwidth})")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("MMD_GMM_MSIP-Fredholm-vs-SVGD.pdf")
plt.close()
print("Saved MMD_GMM_MSIP-Fredholm-vs-SVGD.pdf")
# %%