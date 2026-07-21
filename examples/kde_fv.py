from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from nak_torch.algorithms import grad_aldi, msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm
from functions import gaussian_kde
from nak_torch.tools.metrics import CrossEntropy

# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(314159)

# ── Dataset / target ──────────────────────────────────────────────────────────
sigma = 0.1
npz_file = np.load('datasets/two_bananas.npz')
dataset = 10 * npz_file['X'].T - np.mean(10 * npz_file['X'].T)

#post_log_dens = gaussian_kde(sigma, dataset)

#_kde = gaussian_kde(sigma, dataset)
#post_log_dens = lambda x: _kde(x).squeeze(0)

_kde = gaussian_kde(sigma, dataset)
post_log_dens = _kde  # no squeeze needed anymore

post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_batch = torch.vmap(torch.func.grad(post_log_dens))

cross_entropy_metric = CrossEntropy(
    post_log_dens_batch,
    is_log_dens_vectorized=True,
)

# ── Shared hyper-parameters ───────────────────────────────────────────────────
n_steps = 1000
n_particles = 25
lr = 0.5
lr_msip = 0.5

kernel_length_scale = 0.1
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-4.0, 4.0)

init_mean = torch.tensor([0.0, 0.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))

# ══════════════════════════════════════════════════════════════════════════════
# Run algorithms
# ══════════════════════════════════════════════════════════════════════════════



print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds, gradient_decay=gradient_decay,
    keep_all=True, compile_step=False, verbose=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════════════════

algo_trajs = {
#    "SVGD": trajectories_svgd,
#    "GI-ALDI": trajectories_galdi,
    "MSIP-Fredholm": trajectories_msip_f,
}

algo_few_trajs = {
#    "SVGD": trajectories_svgd,
#    "GI-ALDI": trajectories_galdi,
}

algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
}

algo_wts_trajs = {
    "MSIP-Fredholm": traj_wts_msip_f,
}

metrics = {
    name: {"ksd_rbf": [], "ksd_imq": [], "cross_entropy": []}
    for name in algo_trajs
}

metric_every = 10
steps_recorded = list(range(0, n_steps, metric_every))

# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * kernel_length_scale ** 2))


def imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.1)


def compute_ksd(particles, log_p_grad_fn, kernel_fn):
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
            torch.func.grad(kernel_fn, argnums=1), argnums=0
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
# Cross entropy loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing cross entropy …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        pts = traj[step_idx]
        wts = None

        if name in algo_wts_trajs:
            wts = algo_wts_trajs[name][step_idx]
            wts = wts / wts.sum()

        metrics[name]["cross_entropy"].append(
            cross_entropy_metric(pts, wts=wts).item()
        )
print("Cross entropy done.")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (RBF) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_rbf"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, rbf_fn)
        )
print("KSD (RBF) done.")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (IMQ) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_imq"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, imq_fn)
        )
print("KSD (IMQ) done.")

# ══════════════════════════════════════════════════════════════════════════════
# Contour grid
# ══════════════════════════════════════════════════════════════════════════════

Ngrid = 200
xgrid = torch.linspace(-4, 4, Ngrid)
ygrid = torch.linspace(-4, 4, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens_batch(grid_pts).reshape(Ngrid, Ngrid)

z_min = Z.max() - 20
levels = torch.linspace(z_min, Z.max(), 20).detach().cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# Individual particle plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_particles(ax, pts, wts=None, title=""):
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=levels, alpha=0.6)
    c = torch.abs(wts.detach().cpu()) if wts is not None else None
    sc = ax.scatter(
        pts[:, 0].detach().cpu(), pts[:, 1].detach().cpu(),
        c=c, cmap="viridis", alpha=0.7, s=60,
        edgecolors="k", linewidths=0.4,
    )
    if wts is not None:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect(1.0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")


for name, traj in algo_trajs.items():
    pts = traj[-1]
    wts = algo_last_wts.get(name, None)
    if wts is not None:
        wts = wts / wts.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_particles(ax, pts, wts, title=f"{name} – step {n_steps}")
    plt.tight_layout()
    fname = (
        f"kde_particles_{name.replace(' ', '_').replace('/', '-')}_"
        f"sigma_{kernel_length_scale}.pdf"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# Cross entropy vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.plot(steps_recorded, metrics[name]["cross_entropy"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Cross entropy  $-\mathbb{E}_{\mu}[\log \pi]$")
ax.set_title("Cross entropy vs Iteration – KDE")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("kde_cross_entropy_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved kde_cross_entropy_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – KDE")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("kde_KSD_RBF_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved kde_KSD_RBF_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – KDE")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("kde_KSD_IMQ_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved kde_KSD_IMQ_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD plots - few algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – KDE")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("kde_KSD_RBF_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – KDE")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("kde_KSD_IMQ_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()

print("All done.")