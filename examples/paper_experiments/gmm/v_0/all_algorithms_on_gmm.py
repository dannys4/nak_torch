
import torch
from nak_torch.algorithms import msip, msip_gs
from nak_torch.algorithms.msip import MSIPGMMGaussianKernel, MSIPQuadGradientInformed
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(1023921)



# ======================================================================
# ???
# ======================================================================


def gmm_rbf_expectations(particles, weights, means, covs, bandwidth):
    sigma_sq = bandwidth ** 2
    K, D = means.shape
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for l in range(K):
            cov_jl = covs[j] + covs[l] + 2 * sigma_sq * eye
            diff_jl = means[j] - means[l]
            log_k = -0.5 * (diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                            + torch.logdet(2 * torch.pi * cov_jl))
            Epp = Epp + weights[j] * weights[l] * log_k.exp()
    smoothed_covs = covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - means[k].unsqueeze(0)
        log_k = -0.5 * (
            torch.einsum("ni,ij,nj->n", diff_nk, torch.linalg.inv(smoothed_covs[k]), diff_nk)
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        )
        Exp = Exp + weights[k] * log_k.exp().mean()
    return Epp, Exp

# ======================================================================
# The log-density
# ======================================================================

gmm_weights = torch.tensor([1/3, 1/3, 1/3])
gmm_means = torch.stack([
    torch.tensor([6.2,  -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([11.0,   5.0]),
])
#gmm_covs = torch.stack([torch.eye(2) / 5.0] * 3)

gmm_covs = torch.stack([
    torch.tensor([[0.5,  0.3],
                  [0.3,  0.4]]),   # tilted
    torch.tensor([[0.6, -0.2],
                  [-0.2, 0.3]]),   # anti-tilted
    torch.tensor([[0.1,  0.0],
                  [0.0,  0.8]]),   # axis-aligned elongated
])

def gmm_generator(weights, means, covs):
    log_weights = torch.log(weights)
    precisions  = torch.linalg.inv(covs)

    def log_density(pt):
        log_probs = []
        for k in range(len(log_weights)):
            diff = pt - means[k]
            maha = (diff * (diff @ precisions[k])).sum(dim=-1)
            log_probs.append(log_weights[k] - 0.5 * maha)
        return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()

    return log_density

log_density = gmm_generator(gmm_weights, gmm_means, gmm_covs)



# ======================================================================
# The parameters of the experiment
# ======================================================================

bounds = (-100., 100.)

n_steps, n_particles = 500, 25
lr_msip = 100e-3
kernel_length_scale = 0.9
kernel_diag_infl = 1e-8

# ======================================================================
# The initialization
# ======================================================================


init_mean = torch.tensor([18.0, 18.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))



@torch.compile(dynamic=False)
def mc_quad_rule(batch_size: int, N_quad: int = 10, dim: int = 2):
    # dim = dimension
    pts = torch.randn((batch_size, N_quad, dim), dtype=torch.float64)
    wts = torch.ones((batch_size, N_quad), dtype=torch.float64).div_(N_quad)
    return pts, wts



exact_MSIP_estimator = MSIPGMMGaussianKernel(
    weights=gmm_weights,
    means=gmm_means,
    covariances=gmm_covs,
    bandwidth=kernel_length_scale
)

GRADIENT_DECAY = 1.0

quadgrad_MSIP_estimator = MSIPQuadGradientInformed(
    torch.func.vmap(torch.func.grad_and_value(log_density)),
    mc_quad_rule,
    gradient_decay=GRADIENT_DECAY
)

trajectories_msip, traj_wts_msip = msip(
    exact_MSIP_estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)

trajectories_msip_gs, traj_wts_qgmsip_gs = msip_gs(
    exact_MSIP_estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)


trajectories_qgmsip, traj_wts_qgmsip = msip(
    quadgrad_MSIP_estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)

trajectories_qgmsip_gs, traj_wts_msip_gs = msip_gs(
    quadgrad_MSIP_estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)


mmd_every = 10
steps_recorded = list(range(0, n_steps, mmd_every))
mmd_msip, mmd_msip_gs = [], []

for i in steps_recorded:
    for traj, mmd_list in [
        (trajectories_msip,    mmd_msip),
        (trajectories_msip_gs, mmd_msip_gs),
    ]:
        pts = traj[i]
        Epp, Exp = gmm_rbf_expectations(pts, gmm_weights, gmm_means, gmm_covs, kernel_length_scale)
        Kxx = torch.exp(-torch.cdist(pts, pts).pow(2) / (2 * kernel_length_scale ** 2)).mean()
        mmd_val = (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()
        mmd_list.append(mmd_val)

Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = log_density(grid_pts).reshape(Ngrid, Ngrid)

pts_msip    = trajectories_msip[-1]
wts_msip    = traj_wts_msip[-1];    wts_msip    = wts_msip    / wts_msip.sum()
pts_msip_gs = trajectories_msip_gs[-1]
wts_msip_gs = traj_wts_msip_gs[-1]; wts_msip_gs = wts_msip_gs / wts_msip_gs.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contour(X, Y, Z, levels=20)
axes[0].scatter(pts_msip[:, 0],    pts_msip[:, 1],    c=wts_msip,    alpha=0.5)
axes[0].set_title("MSIP")
axes[0].set_aspect(1.0)
axes[1].contour(X, Y, Z, levels=20)
axes[1].scatter(pts_msip_gs[:, 0], pts_msip_gs[:, 1], c=wts_msip_gs, alpha=0.5)
axes[1].set_title("MSIP-GS")
axes[1].set_aspect(1.0)
plt.tight_layout()
plt.show()


Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = log_density(grid_pts).reshape(Ngrid, Ngrid)

pts_msip    = trajectories_msip[-1]
wts_msip    = traj_wts_msip[-1];    wts_msip    = wts_msip    / wts_msip.sum()
pts_msip_gs = trajectories_msip_gs[-1]
wts_msip_gs = traj_wts_msip_gs[-1]; wts_msip_gs = wts_msip_gs / wts_msip_gs.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contour(X, Y, Z, levels=20)
axes[0].scatter(pts_msip[:, 0],    pts_msip[:, 1],    c=wts_msip,    alpha=0.5)
axes[0].set_title("MSIP")
axes[0].set_xlim(-10, 15)
axes[0].set_ylim(-10, 15)
axes[0].set_aspect(1.0)
axes[1].contour(X, Y, Z, levels=20)
axes[1].scatter(pts_msip_gs[:, 0], pts_msip_gs[:, 1], c=wts_msip_gs, alpha=0.5)
axes[1].set_title("MSIP-GS")
axes[1].set_xlim(-10, 15)
axes[1].set_ylim(-10, 15)
axes[1].set_aspect(1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps_recorded, mmd_msip,    label="MSIP")
ax.plot(steps_recorded, mmd_msip_gs, label="MSIP-GS")
ax.set_xlabel("Step")
ax.set_ylabel("MMD to target GMM")
ax.legend()
plt.tight_layout()
plt.show()


mmd_every = 10
steps_recorded = list(range(0, n_steps, mmd_every))
mmd_msip, mmd_msip_gs, mmd_qgmsip, mmd_qgmsip_gs = [], [], [], []

for i in steps_recorded:
    for traj, mmd_list in [
        (trajectories_msip,       mmd_msip),
        (trajectories_msip_gs,    mmd_msip_gs),
        (trajectories_qgmsip,     mmd_qgmsip),
        (trajectories_qgmsip_gs,  mmd_qgmsip_gs),
    ]:
        pts = traj[i]
        Epp, Exp = gmm_rbf_expectations(pts, gmm_weights, gmm_means, gmm_covs, kernel_length_scale)
        Kxx = torch.exp(-torch.cdist(pts, pts).pow(2) / (2 * kernel_length_scale ** 2)).mean()
        mmd_val = (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()
        mmd_list.append(mmd_val)


Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = log_density(grid_pts).reshape(Ngrid, Ngrid)

pts_msip       = trajectories_msip[-1]
wts_msip       = traj_wts_msip[-1];       wts_msip       = wts_msip       / wts_msip.sum()
pts_msip_gs    = trajectories_msip_gs[-1]
wts_msip_gs    = traj_wts_msip_gs[-1];    wts_msip_gs    = wts_msip_gs    / wts_msip_gs.sum()
pts_qgmsip     = trajectories_qgmsip[-1]
wts_qgmsip     = traj_wts_qgmsip[-1];     wts_qgmsip     = wts_qgmsip     / wts_qgmsip.sum()
pts_qgmsip_gs  = trajectories_qgmsip_gs[-1]
wts_qgmsip_gs  = traj_wts_qgmsip_gs[-1];  wts_qgmsip_gs  = wts_qgmsip_gs  / wts_qgmsip_gs.sum()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, pts, wts, title in [
    (axes[0, 0], pts_msip,      wts_msip,      "MSIP"),
    (axes[0, 1], pts_msip_gs,   wts_msip_gs,   "MSIP-GS"),
    (axes[1, 0], pts_qgmsip,    wts_qgmsip,    "QGrad-MSIP"),
    (axes[1, 1], pts_qgmsip_gs, wts_qgmsip_gs, "QGrad-MSIP-GS"),
]:
    ax.contour(X, Y, Z, levels=20)
    ax.scatter(pts[:, 0], pts[:, 1], c=wts, alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 15)
    ax.set_aspect(1.0)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps_recorded, mmd_msip,      label="MSIP")
ax.plot(steps_recorded, mmd_msip_gs,   label="MSIP-GS")
ax.plot(steps_recorded, mmd_qgmsip,    label="QGrad-MSIP")
ax.plot(steps_recorded, mmd_qgmsip_gs, label="QGrad-MSIP-GS")
ax.set_xlabel("Step")
ax.set_ylabel("MMD to target GMM")
ax.legend()
plt.tight_layout()
plt.show()




