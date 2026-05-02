# %%
"""
Benchmark particle algorithms on Neal's funnel distribution.

This is the funnel analogue of the 5-GMM benchmark:
  - SVGD
  - GI-ALDI
  - EKS, optional / diagnostic only
  - MSIP-Fredholm
  - MSIP-QG
  - MSIP-GS-QG
  - MSIP-QGF
  - MSIP-GS-QGF

Important:
  The old MSIP-GMM and MSIP-GS-GMM variants are GMM-specific and are not
  available for a generic funnel target unless you derive closed-form v0/v1
  formulas for that target.

Diagnostics:
  - MMD to a large reference sample from the funnel, using empirical MMD.
  - KSD using the funnel score.
  - particle plots over funnel reference samples.
"""

from functools import partial

import torch
import torch._dynamo
torch._dynamo.disable()

import matplotlib.pyplot as plt

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)
#from nak_torch.tools.quadrature import (
#    spherical_MC_radial_Laguerre,
#    spherical_struct_radial_Laguerre,
#)

from nak_torch.tools.metrics import CrossEntropy

from functions import funnel


# ── Device / dtype ────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(314159)


# ── Target: funnel distribution ───────────────────────────────────────────────

DIM = 2

funnel_log_density = funnel.logpdf_factory(DIM)
funnel_sampler = funnel.sample_factory(DIM)

cross_entropy_metric = CrossEntropy(
    funnel_log_density,
    is_log_dens_vectorized=True,
)

funnel_grad_val = torch.func.grad_and_value(funnel_log_density)
funnel_grad_val_batch = torch.vmap(funnel_grad_val)

rng_ref = torch.Generator(device=torch.get_default_device())
rng_ref.manual_seed(0)

# Reference samples for plots and empirical MMD.
# Increase this if you want a more accurate MMD estimate.
N_REF = 1000
ref_samples = funnel_sampler(rng_ref, N_REF)


# ── Shared hyper-parameters ───────────────────────────────────────────────────

n_steps = 5000
n_particles = 100

lr = 0.01
lr_msip = 0.4

kernel_length_scale = 0.1
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-1000.0, 1000.0)

init_particles = torch.randn((n_particles, DIM))


# ── Optional Gaussian model wrapper for EKS ───────────────────────────────────
#
# Warning:
# EKS here does not target the funnel distribution. This is only kept so that
# the script can compare the same list of algorithm names as in the GMM file.
# For a fair funnel benchmark, set RUN_EKS = False.

RUN_EKS = False

if RUN_EKS:
    obs_op = torch.eye(DIM)
    forward_model = lambda particles: particles @ obs_op
    true_obs = torch.zeros(DIM)

    model = nak_torch.GaussianModel(
        forward_model,
        likelihood_precision=1.0,
        prior_precision=1.0,
        true_obs=true_obs,
        is_vectorized=True,
    )


# ── Quadrature rules ──────────────────────────────────────────────────────────

def mc_quad_rule(batch_size: int, N_quad: int = 40, dim: int = DIM):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


# def spherical_quad_mc(batch_size: int, N_quad: int = 40, dim: int = DIM):
#     N_spherical = max(1, N_quad)
#     N_radial = 1
#     pts, wts = spherical_MC_radial_Laguerre(
#         batch_size,
#         N_spherical,
#         dim,
#         N_radial,
#         dtype=torch.float64,
#     )
#     return pts, wts


# def spherical_quad_struct(batch_size: int, N_quad: int = 40, dim: int = DIM):
#     N_spherical = 2 * dim
#     N_radial = max(1, int(N_quad / N_spherical))
#     pts, wts = spherical_struct_radial_Laguerre(
#         batch_size,
#         N_spherical,
#         dim,
#         N_radial,
#         dtype=torch.float64,
#     )
#     return pts, wts


quad_rule = partial(mc_quad_rule, N_quad=5)


# ══════════════════════════════════════════════════════════════════════════════
# Run algorithms
# ══════════════════════════════════════════════════════════════════════════════

print("=== SVGD ===")
trajectories_svgd = svgd(
    funnel_log_density,
    n_particles,
    n_steps,
    dim=DIM,
    lr=lr,
    init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

print("=== GI-ALDI ===")
trajectories_galdi = grad_aldi(
    funnel_log_density,
    n_particles,
    n_steps,
    dim=DIM,
    lr=lr / 3,
    init_particles=init_particles,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

if RUN_EKS:
    print("=== EKS diagnostic only, not funnel-targeting ===")
    trajectories_eks = eks(
        model,
        n_particles=n_particles,
        n_steps=n_steps,
        dim=DIM,
        lr=lr / 3,
        init_particles=init_particles,
        keep_all=True,
        compile_step=False,
        verbose=True,
    )

print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(
    gradient_decay=gradient_decay,
    log_dens_grad_val=funnel_grad_val_batch,
)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm,
    n_particles,
    n_steps,
    dim=DIM,
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

print("=== MSIP-QG ===")
msip_quadgrad = MSIPQuadGradientInformed(
    funnel_grad_val_batch,
    quad_rule,
    gradient_decay,
)
trajectories_msip_qg, traj_wts_msip_qg = msip(
    msip_quadgrad,
    n_particles,
    n_steps,
    dim=DIM,
    lr=lr_msip,
    init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

# print("=== MSIP-GS-QG ===")
# msip_quadgrad_gs = MSIPQuadGradientInformed(
#     funnel_grad_val_batch,
#     quad_rule,
#     gradient_decay,
# )
# trajectories_msip_gs_qg, traj_wts_msip_gs_qg = msip_gs(
#     msip_quadgrad_gs,
#     n_particles,
#     n_steps,
#     dim=DIM,
#     lr=lr_msip,
#     init_particles=init_particles,
#     kernel_length_scale=kernel_length_scale,
#     kernel_diag_infl=kernel_diag_infl,
#     bounds=bounds,
#     keep_all=True,
#     compile_step=False,
#     verbose=True,
# )


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════════════════

algo_trajs = {
    "SVGD": trajectories_svgd,
    #"GI-ALDI": trajectories_galdi,
    "MSIP-Fredholm": trajectories_msip_f,
    "MSIP-QG": trajectories_msip_qg,
    #"MSIP-GS-QG": trajectories_msip_gs_qg,
}

if RUN_EKS:
    algo_trajs["EKS diagnostic"] = trajectories_eks

algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
    "MSIP-QG": traj_wts_msip_qg[-1],
    #"MSIP-GS-QG": traj_wts_msip_gs_qg[-1],
}


algo_wts_trajs = {
    "MSIP-Fredholm": traj_wts_msip_f,
    "MSIP-QG": traj_wts_msip_qg,
    #"MSIP-GS-QG": traj_wts_msip_gs_qg,
}

#metrics = {name: {"mmd": [], "ksd_rbf": [], "ksd_imq": []} for name in algo_trajs}

metrics = {
    name: {"mmd": [], "ksd_rbf": [], "ksd_imq": [], "cross_entropy": []}
    for name in algo_trajs
}

metric_every = 20
steps_recorded = list(range(0, n_steps, metric_every))


# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def subsample_reference(ref, n_max=2000):
    if ref.shape[0] <= n_max:
        return ref
    idx = torch.randperm(ref.shape[0], device=ref.device)[:n_max]
    return ref[idx]


ref_mmd = subsample_reference(ref_samples, n_max=2000)


def compute_mmd_empirical(particles, reference, bandwidth):
    """Biased empirical MMD using the same unnormalized RBF kernel everywhere."""
    Kxx = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bandwidth**2)
    ).mean()

    Kyy = torch.exp(
        -torch.cdist(reference, reference).pow(2) / (2 * bandwidth**2)
    ).mean()

    Kxy = torch.exp(
        -torch.cdist(particles, reference).pow(2) / (2 * bandwidth**2)
    ).mean()

    return (Kxx + Kyy - 2 * Kxy).clamp(min=0.0).sqrt().item()


def rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * kernel_length_scale**2))


def imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.1)


def compute_ksd(particles, log_p_grad_fn, kernel_fn):
    """Vectorized KSD U-statistic over all n*(n-1) off-diagonal pairs."""
    n = particles.shape[0]
    grads, _ = log_p_grad_fn(particles)

    xi_flat = particles.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    xj_flat = particles.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    si_flat = grads.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    sj_flat = grads.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)

    k_vals = torch.vmap(kernel_fn)(xi_flat, xj_flat)
    grad_xi_k = torch.vmap(torch.func.grad(kernel_fn, argnums=0))(xi_flat, xj_flat)
    grad_xj_k = torch.vmap(torch.func.grad(kernel_fn, argnums=1))(xi_flat, xj_flat)

    def mixed_trace(a, b):
        return torch.func.jacfwd(
            torch.func.grad(kernel_fn, argnums=1),
            argnums=0,
        )(a, b).diagonal().sum()

    trace_mixed = torch.vmap(mixed_trace)(xi_flat, xj_flat)

    h_flat = (
        (si_flat * sj_flat).sum(-1) * k_vals
        + (sj_flat * grad_xi_k).sum(-1)
        + (si_flat * grad_xj_k).sum(-1)
        + trace_mixed
    )

    mask = ~torch.eye(n, dtype=torch.bool, device=particles.device)
    return h_flat.reshape(n, n)[mask].sum().div(n * (n - 1)).clamp(min=0.0).sqrt().item()


# ══════════════════════════════════════════════════════════════════════════════
# Metric loops
# ══════════════════════════════════════════════════════════════════════════════

print("Computing MMD …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["mmd"].append(
            compute_mmd_empirical(traj[step_idx], ref_mmd, kernel_length_scale)
        )
print("MMD done.")

# print("Computing cross entropy …")
# for step_idx in steps_recorded:
#     for name, traj in algo_trajs.items():
#         pts = traj[step_idx]
#         wts = None

#         # Use MSIP weights when available.
#         if name in algo_last_wts:
#             # Careful: this only uses final weights.
#             # Better version below if you store full weight trajectories.
#             wts = algo_last_wts[name]

#         metrics[name]["cross_entropy"].append(
#             cross_entropy_metric(pts, wts=wts).item()
#         )
# print("Cross entropy done.")


print("Computing cross entropy …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        pts = traj[step_idx]
        wts = None

        #if name in algo_wts_trajs:
        #    wts = algo_wts_trajs[name][step_idx]

        metrics[name]["cross_entropy"].append(
            cross_entropy_metric(pts, wts=wts).item()
        )
print("Cross entropy done.")

print("Computing KSD (RBF) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_rbf"].append(
            compute_ksd(traj[step_idx], funnel_grad_val_batch, rbf_fn)
        )
print("KSD (RBF) done.")

print("Computing KSD (IMQ) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_imq"].append(
            compute_ksd(traj[step_idx], funnel_grad_val_batch, imq_fn)
        )
print("KSD (IMQ) done.")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def normalize_marker_weights(wts, min_size=10.0, max_size=80.0):
    w = torch.abs(wts.detach())
    w = w / (w.max() + 1e-30)
    n = w.shape[0]
    w = (1/n)*torch.ones(n)
    #return min_size + (max_size - min_size) * w
    return w

def plot_particles(ax, pts, wts=None, title=""):
    ax.scatter(
        ref_samples[:, 0].detach().cpu(),
        ref_samples[:, 1].detach().cpu(),
        alpha=0.08,
        s=5,
        label="reference funnel samples",
    )

    sizes = 35
    colors = None
    if wts is not None:
        sizes = normalize_marker_weights(wts).detach().cpu()
        colors = torch.abs(wts.detach()).cpu()

    sc = ax.scatter(
        pts[:, 0].detach().cpu(),
        pts[:, 1].detach().cpu(),
        c=colors,
        s=sizes,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.4,
        label="particles",
    )

    if wts is not None:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="|weight|")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    #ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7)


for name, traj in algo_trajs.items():
    pts = traj[-1]
    wts = algo_last_wts.get(name, None)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_particles(ax, pts, wts, title=f"{name} – step {n_steps}")
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("/", "-")
    fname = f"funnel_particles_{safe_name}_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.plot(steps_recorded, metrics[name]["cross_entropy"], label=name)

ax.set_xlabel("Iteration")
ax.set_ylabel("Cross entropy  -E_mu[log pi]")
ax.set_title("Funnel: cross entropy vs iteration")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"funnel_cross_entropy_all_algorithms_sigma_{kernel_length_scale}.pdf")
plt.close()
print(f"Saved funnel_cross_entropy_all_algorithms_sigma_{kernel_length_scale}.pdf")

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["mmd"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("Empirical MMD")
ax.set_title("Funnel: empirical MMD vs iteration")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"funnel_MMD_all_algorithms_sigma_{kernel_length_scale}.pdf")
plt.close()
print(f"Saved funnel_MMD_all_algorithms_sigma_{kernel_length_scale}.pdf")


fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("Funnel: KSD [RBF] vs iteration")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"funnel_KSD_RBF_all_algorithms_sigma_{kernel_length_scale}.pdf")
plt.close()
print(f"Saved funnel_KSD_RBF_all_algorithms_sigma_{kernel_length_scale}.pdf")


fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("Funnel: KSD [IMQ] vs iteration")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"funnel_KSD_IMQ_all_algorithms_sigma_{kernel_length_scale}.pdf")
plt.close()
print(f"Saved funnel_KSD_IMQ_all_algorithms_sigma_{kernel_length_scale}.pdf")



print("All done.")