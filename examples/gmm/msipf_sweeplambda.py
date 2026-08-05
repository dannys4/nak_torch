# %%
# ─────────────────────────────────────────────────────────────────────────────
# MSIP-Fredholm on a rotated GMM — sweep the diagonal regularization lambda.
#
# For each
#     M in {5, 25, 100, 250}
# and each value of lambda, run MSIP-Fredholm for 1000 iterations and record:
#   • final weighted MMD (using the Fredholm weights),
#   • final unweighted MMD (uniform weights on the same particles),
#   • whether all final Fredholm weights are positive, and
#   • the smallest final Fredholm weight.
#
# The output figure contains all MMD-vs-lambda curves in one graph:
#   • color identifies M,
#   • solid lines are weighted MMD,
#   • dashed lines are unweighted MMD.
# ─────────────────────────────────────────────────────────────────────────────
import csv
import math

import matplotlib

matplotlib.use("Agg")  # file output only; remove for an interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from nak_torch.algorithms import msip
from nak_torch.algorithms.msip import MSIPFredholm

# ── Device / dtype ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

SEED = 1023921

print(f"device={torch.empty(0).device}  dtype={torch.get_default_dtype()}")

# ── Target: rotated five-component GMM ───────────────────────────────────────
gmm_weights = torch.tensor([1 / 5] * 5, dtype=torch.get_default_dtype())

gmm_means = torch.stack(
    [
        2 * torch.tensor([6.2, -6.0], dtype=torch.get_default_dtype()),
        2 * torch.tensor([-4.0, 5.0], dtype=torch.get_default_dtype()),
        2 * torch.tensor([7.0, 3.0], dtype=torch.get_default_dtype()),
        2 * torch.tensor([-6.5, -4.5], dtype=torch.get_default_dtype()),
        2 * torch.tensor([1.0, 7.0], dtype=torch.get_default_dtype()),
    ]
)

gmm_covs = torch.stack(
    [
        torch.tensor([[1.5, 0.1], [0.1, 0.5]], dtype=torch.get_default_dtype()),
        torch.tensor([[2.0, -0.6], [-0.6, 0.5]], dtype=torch.get_default_dtype()),
        torch.tensor([[0.7, 0.4], [0.4, 1.2]], dtype=torch.get_default_dtype()),
        torch.tensor([[1.3, -0.5], [-0.5, 0.9]], dtype=torch.get_default_dtype()),
        torch.tensor([[0.6, 0.35], [0.35, 1.6]], dtype=torch.get_default_dtype()),
    ]
)

gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    log_probs = []
    for mean, precision, weight in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        log_prob = torch.log(weight) - 0.5 * torch.einsum(
            "...i,ij,...j", diff, precision, diff
        )
        log_probs.append(log_prob)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


# Batched (gradient, value) of the log-density, required by MSIP-Fredholm.
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# ── Sweep configuration ──────────────────────────────────────────────────────
particle_counts = [50, 100, 250]

# One point per decade, including the original value 1e-8.
# Edit this list for a denser or narrower sweep.
lambda_values = [10.0**exponent for exponent in range(-10, 1)]

n_steps = 1000
n_trials = 50  # Increase for error bars; cost scales linearly with this value.

lr_msip = 0.8
kernel_length_scale = 0.1
gradient_decay = 1.0
bounds = (-100.0, 100.0)

# Bandwidth used only to evaluate MMD.
mmd_bandwidth = 0.1

init_mean = torch.tensor([0.0, 0.0])
init_std = 1.0
VERBOSE = False


# ══════════════════════════════════════════════════════════════════════════════
# Closed-form MMD to the target GMM for an RBF kernel
# ══════════════════════════════════════════════════════════════════════════════
def gmm_rbf_expectations(
    particles,
    mixture_weights,
    means,
    covariances,
    bandwidth,
    particle_weights=None,
):
    """Return E_pi,pi[k] and E_mu,pi[k] for the RBF kernel."""
    sigma_sq = bandwidth**2
    n_components, dimension = means.shape

    eye = torch.eye(
        dimension, device=particles.device, dtype=particles.dtype
    )
    log_constant = 0.5 * dimension * math.log(2 * math.pi * sigma_sq)

    target_target = torch.tensor(
        0.0, device=particles.device, dtype=particles.dtype
    )
    for j in range(n_components):
        for ell in range(n_components):
            covariance = covariances[j] + covariances[ell] + sigma_sq * eye
            difference = means[j] - means[ell]
            log_kernel_expectation = -0.5 * (
                difference @ torch.linalg.solve(covariance, difference)
                + torch.logdet(2 * torch.pi * covariance)
            ) + log_constant
            target_target = (
                target_target
                + mixture_weights[j]
                * mixture_weights[ell]
                * log_kernel_expectation.exp()
            )

    smoothed_covariances = covariances + sigma_sq * eye.unsqueeze(0)
    particle_target = torch.tensor(
        0.0, device=particles.device, dtype=particles.dtype
    )
    for component in range(n_components):
        difference = particles - means[component].unsqueeze(0)
        log_kernel_expectation = -0.5 * (
            torch.einsum(
                "ni,ij,nj->n",
                difference,
                torch.linalg.inv(smoothed_covariances[component]),
                difference,
            )
            + torch.logdet(2 * torch.pi * smoothed_covariances[component])
        ) + log_constant

        contribution = log_kernel_expectation.exp()
        if particle_weights is None:
            contribution = contribution.mean()
        else:
            contribution = (contribution * particle_weights).sum()

        particle_target = (
            particle_target + mixture_weights[component] * contribution
        )

    return target_target, particle_target


def compute_mmd(particles, bandwidth, weights=None):
    """Compute MMD between the particle measure and the target GMM."""
    kernel_matrix = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bandwidth**2)
    )

    if weights is None:
        target_target, particle_target = gmm_rbf_expectations(
            particles,
            gmm_weights,
            gmm_means,
            gmm_covs,
            bandwidth,
        )
        particle_particle = kernel_matrix.mean()
    else:
        normalized_weights = weights / weights.sum()
        target_target, particle_target = gmm_rbf_expectations(
            particles,
            gmm_weights,
            gmm_means,
            gmm_covs,
            bandwidth,
            particle_weights=normalized_weights,
        )
        particle_particle = (
            normalized_weights @ kernel_matrix @ normalized_weights
        )

    mmd_squared = particle_particle + target_target - 2 * particle_target
    return mmd_squared.clamp(min=0.0).sqrt().item()


def mean_std(values):
    """Return sample mean and standard deviation."""
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, variance**0.5


def run_msip(init_particles, n_particles, regularization):
    """Run MSIP-Fredholm with kernel diagonal inflation lambda."""
    msip_fredholm = MSIPFredholm(
        gradient_decay, post_log_dens_grad_val_batch
    )
    return msip(
        msip_fredholm,
        n_particles,
        n_steps,
        dim=2,
        lr=lr_msip,
        init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        kernel_diag_infl=regularization,
        bounds=bounds,
        gradient_decay=gradient_decay,
        keep_all=True,
        compile_step=False,
        verbose=VERBOSE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sweep over M, lambda, and trial
# ══════════════════════════════════════════════════════════════════════════════
records = []

for trial in range(n_trials):
    # The same master cloud is reused across every lambda, and each smaller M is
    # a prefix of the larger cloud. This removes initialization differences from
    # the lambda comparison.
    torch.manual_seed(SEED + trial)
    master_init = init_mean + init_std * torch.randn((max(particle_counts), 2))

    for n_particles in particle_counts:
        init_particles = master_init[:n_particles].clone()
        print(f"\n===== trial={trial}, M={n_particles} =====")

        for regularization in lambda_values:
            trajectories, weight_trajectories = run_msip(
                init_particles,
                n_particles,
                regularization,
            )

            final_particles = trajectories[-1]
            final_weights = weight_trajectories[-1]

            record = {
                "M": n_particles,
                "trial": trial,
                "seed": SEED + trial,
                "lambda": regularization,
                "all_positive": bool((final_weights > 0).all().item()),
                "min_weight": final_weights.min().item(),
                "mmd_weighted": compute_mmd(
                    final_particles,
                    mmd_bandwidth,
                    weights=final_weights,
                ),
                "mmd_unweighted": compute_mmd(
                    final_particles,
                    mmd_bandwidth,
                ),
            }
            records.append(record)

            print(
                f"lambda={regularization:>9.1e} | "
                f"MMD weighted={record['mmd_weighted']:.4e} | "
                f"unweighted={record['mmd_unweighted']:.4e} | "
                f"all weights > 0: {record['all_positive']} | "
                f"min weight={record['min_weight']:.3e}"
            )

            del trajectories, weight_trajectories


# ── Aggregate across trials ──────────────────────────────────────────────────
summary = []
for n_particles in particle_counts:
    for regularization in lambda_values:
        rows = [
            record
            for record in records
            if record["M"] == n_particles
            and record["lambda"] == regularization
        ]

        weighted_mean, weighted_std = mean_std(
            [row["mmd_weighted"] for row in rows]
        )
        unweighted_mean, unweighted_std = mean_std(
            [row["mmd_unweighted"] for row in rows]
        )

        summary.append(
            {
                "M": n_particles,
                "lambda": regularization,
                "n_trials": len(rows),
                "all_positive_count": sum(
                    row["all_positive"] for row in rows
                ),
                "worst_min_weight": min(row["min_weight"] for row in rows),
                "mmd_weighted_mean": weighted_mean,
                "mmd_weighted_std": weighted_std,
                "mmd_unweighted_mean": unweighted_mean,
                "mmd_unweighted_std": unweighted_std,
            }
        )


# ── Save raw and aggregated CSV files ────────────────────────────────────────
raw_csv = "per_trial_MSIP-Fredholm_lambda_sweep.csv"
with open(raw_csv, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
print(f"\nSaved {raw_csv}")

summary_csv = "summary_MSIP-Fredholm_lambda_sweep.csv"
with open(summary_csv, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print(f"Saved {summary_csv}")


# ── Plot every curve in one graph ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.0, 6.0))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for color_index, n_particles in enumerate(particle_counts):
    rows = [row for row in summary if row["M"] == n_particles]
    color = colors[color_index % len(colors)]
    lambdas = [row["lambda"] for row in rows]

    weighted_means = [row["mmd_weighted_mean"] for row in rows]
    weighted_stds = [row["mmd_weighted_std"] for row in rows]
    unweighted_means = [row["mmd_unweighted_mean"] for row in rows]
    unweighted_stds = [row["mmd_unweighted_std"] for row in rows]

    ax.plot(
        lambdas,
        weighted_means,
        marker="o",
        linestyle="-",
        color=color,
        label=f"M={n_particles}, weighted",
    )
    # ax.plot(
    #     lambdas,
    #     unweighted_means,
    #     marker="s",
    #     linestyle="--",
    #     color=color,
    #     label=f"M={n_particles}, unweighted",
    # )

    if n_trials > 1:
        weighted_lower = [
            max(mean - std, torch.finfo(torch.get_default_dtype()).tiny)
            for mean, std in zip(weighted_means, weighted_stds)
        ]
        weighted_upper = [
            mean + std for mean, std in zip(weighted_means, weighted_stds)
        ]
        unweighted_lower = [
            max(mean - std, torch.finfo(torch.get_default_dtype()).tiny)
            for mean, std in zip(unweighted_means, unweighted_stds)
        ]
        unweighted_upper = [
            mean + std for mean, std in zip(unweighted_means, unweighted_stds)
        ]

        ax.fill_between(
            lambdas,
            weighted_lower,
            weighted_upper,
            color=color,
            alpha=0.12,
        )
        # ax.fill_between(
        #     lambdas,
        #     unweighted_lower,
        #     unweighted_upper,
        #     color=color,
        #     alpha=0.06,
        # )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"regularization $\lambda$")
ax.set_ylabel(rf"MMD at iteration {n_steps} (RBF, $\sigma={mmd_bandwidth}$)")
ax.set_title(
    rf"Sensitivity to regularization for MSIP-Fredholm"
)
ax.grid(True, which="both", alpha=0.3)

# Two compact legends: color encodes M, line style encodes weighting.
particle_legend = [
    Line2D(
        [0],
        [0],
        color=colors[index % len(colors)],
        marker="o",
        linestyle="-",
        label=rf"$M={n_particles}$",
    )
    for index, n_particles in enumerate(particle_counts)
]
weighting_legend = [
    Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linestyle="-",
        label="weighted MMD",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        marker="s",
        linestyle="--",
        label="unweighted MMD",
    ),
]

first_legend = ax.legend(
    handles=particle_legend,
    loc="lower left",
)
ax.add_artist(first_legend)
# ax.legend(
#     title="MMD evaluation",
#     loc="lower left",
# )
#handles=weighting_legend,

plt.tight_layout()
figure_path = "MMD_vs_lambda_MSIP-Fredholm_all_M.pdf"
plt.savefig(figure_path, bbox_inches="tight")
plt.close()
print(f"Saved {figure_path}")
# %%