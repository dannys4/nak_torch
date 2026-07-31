# %%
# ─────────────────────────────────────────────────────────────────────────────
# MSIP-Fredholm vs SVGD on a rotated GMM — sweep over the number of particles.
#
# Same target, dynamics, and closed-form MMD-to-GMM diagnostic as
# msip-fredholm-vs-svgd_on_gmm.py, but the run is wrapped in a loop over
#     M ∈ {5, 25, 100, 250, 1000}.
# For each M we record, at the LAST iteration only:
#   • MMD of MSIP-Fredholm using its signed weights          (weighted)
#   • MMD of MSIP-Fredholm treating particles uniformly       (unweighted)
#   • whether every MSIP-Fredholm weight is > 0               (+ min weight)
#   • MMD of SVGD (inherently uniform)
# and print a table with one row per M (also CSV + an MMD-vs-M plot).
#
# Only final-iteration MMDs are computed (no per-step MMD sweep, no per-M 2D
# scatter plots) — those were the expensive parts and aren't needed for the
# table; re-add them if you want the trajectories back.
# ─────────────────────────────────────────────────────────────────────────────
import math
import torch
import matplotlib

matplotlib.use("Agg")  # file output only; drop this line for interactive use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from nak_torch.algorithms import svgd, msip
from nak_torch.algorithms.msip import MSIPFredholm

# ── Device / dtype ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

SEED = 1023921  # re-applied before each M, so the smaller particle clouds are
                # exact prefixes of the larger ones (reproducible, nested init).

print(f"device={torch.empty(0).device}  dtype={torch.get_default_dtype()}")

# ── Target: rotated 5-component GMM (identical to the source script) ─────────
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

# ── Hyper-parameters (shared across the sweep) ───────────────────────────────
particle_counts = [5, 25, 100, 250, 1000]  # <-- the sweep

n_steps = 100

lr_svgd = 0.8              # SVGD step size
lr_msip = 0.8             # MSIP-Fredholm step size
kernel_length_scale = 0.1  # kernel bandwidth for the *dynamics* (both methods)
kernel_diag_infl = 1e-8
gradient_decay = 1.0       # gamma in MSIP-Fredholm (multiplies the score)
bounds = (-100.0, 100.0)

# Bandwidth for the MMD *metric*, kept separate from the dynamics bandwidth.
# Set this equal to kernel_length_scale to reproduce the single-bandwidth v0
# convention.
mmd_bandwidth = 0.1

init_mean = torch.tensor([0.0, 0.0])
init_std = 1.0

VERBOSE = False  # per-step progress bars off for the sweep; short status per M.

# ══════════════════════════════════════════════════════════════════════════════
# Closed-form MMD to the target GMM (RBF kernel) — verbatim from the source.
#
# MSIP-Fredholm returns *signed* particle weights. weights=None scores the
# uniform particle measure (comparable to the other examples/gmm scripts);
# passing per-step weights scores its actual signed empirical measure.
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

    weights=None  -> uniform particles (matches the other examples).
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
    # .clamp(min=0.0).sqrt().item()


# ══════════════════════════════════════════════════════════════════════════════
# Sweep over the number of particles
# ══════════════════════════════════════════════════════════════════════════════
results = []
for M in particle_counts:
    print(f"\n===== M = {M} particles =====")

    # Shared initial cloud for the two methods; nested across M via fixed seed.
    torch.manual_seed(SEED)
    init_particles = init_mean + init_std * torch.randn((M, 2))

    # ── SVGD ─────────────────────────────────────────────────────────────────
    print("  SVGD …")
    traj_svgd = svgd(
        post_log_dens,
        M,
        n_steps,
        dim=2,
        lr=lr_svgd,
        init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        keep_all=True,
        compile_step=False,
        verbose=VERBOSE,
    )

    # ── MSIP-Fredholm ────────────────────────────────────────────────────────
    print("  MSIP-Fredholm …")
    msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
    traj_msip, wts_msip = msip(
        msip_fredholm,
        M,
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
        verbose=VERBOSE,
    )

    # ── Final-iteration diagnostics ──────────────────────────────────────────
    pts_svgd = traj_svgd[-1]
    pts_msip = traj_msip[-1]
    w_msip = wts_msip[-1]

    mmd_svgd = compute_mmd(pts_svgd, mmd_bandwidth)
    mmd_msip_unwt = compute_mmd(pts_msip, mmd_bandwidth)
    mmd_msip_wt = compute_mmd(pts_msip, mmd_bandwidth, weights=w_msip)
    all_pos = bool((w_msip > 0).all().item())
    min_w = w_msip.min().item()

    results.append(
        dict(
            M=M,
            mmd_msip_wt=mmd_msip_wt,
            mmd_msip_unwt=mmd_msip_unwt,
            all_pos=all_pos,
            min_w=min_w,
            mmd_svgd=mmd_svgd,
        )
    )
    print(
        f"    MMD  MSIP(wt)={mmd_msip_wt:.4e}  MSIP(unwt)={mmd_msip_unwt:.4e}  "
        f"SVGD={mmd_svgd:.4e}  |  all wts>0={all_pos}  min_w={min_w:.2e}"
    )

    del traj_svgd, traj_msip, wts_msip  # free before the next (larger) M

# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════
title = (
    f"Final-iteration MMD to target GMM  (RBF metric σ = {mmd_bandwidth}, "
    f"n_steps = {n_steps})"
)
hdr = (
    f"{'M':>6} | {'MMD MSIP (wt)':>14} | {'MMD MSIP (unwt)':>16} | "
    f"{'all wts > 0':>11} | {'min weight':>12} | {'MMD SVGD':>14}"
)
rule = "=" * len(hdr)

print("\n" + rule)
print(title)
print(rule)
print(hdr)
print("-" * len(hdr))
for r in results:
    print(
        f"{r['M']:>6} | {r['mmd_msip_wt']:>14.4e} | {r['mmd_msip_unwt']:>16.4e} | "
        f"{('yes' if r['all_pos'] else 'no'):>11} | {r['min_w']:>12.2e} | "
        f"{r['mmd_svgd']:>14.4e}"
    )
print(rule)

# ── CSV ──────────────────────────────────────────────────────────────────────
csv_path = "mmd_vs_nparticles_MSIP-Fredholm-vs-SVGD.csv"
with open(csv_path, "w") as f:
    f.write(
        "n_particles,mmd_msip_weighted,mmd_msip_unweighted,"
        "msip_all_weights_positive,msip_min_weight,mmd_svgd\n"
    )
    for r in results:
        f.write(
            f"{r['M']},{r['mmd_msip_wt']:.8e},{r['mmd_msip_unwt']:.8e},"
            f"{int(r['all_pos'])},{r['min_w']:.8e},{r['mmd_svgd']:.8e}\n"
        )
print(f"Saved {csv_path}")

# ── Summary plot: MMD vs M ───────────────────────────────────────────────────
Ms = [r["M"] for r in results]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(Ms, [r["mmd_msip_wt"] for r in results], "s-", label="MSIP-Fredholm (weighted)")
ax.plot(Ms, [r["mmd_msip_unwt"] for r in results], "^--", label="MSIP-Fredholm (unweighted)")
ax.plot(Ms, [r["mmd_svgd"] for r in results], "o-", label="SVGD")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(Ms)
ax.set_xticklabels([str(m) for m in Ms])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("number of particles  M")
ax.set_ylabel(f"final MMD to target GMM  (RBF, σ={mmd_bandwidth})")
ax.set_title("MMD vs particle count — MSIP-Fredholm vs SVGD")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("MMD_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")
plt.close()
print("Saved MMD_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")
# %%