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
import time

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

n_steps = 1000

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

VERBOSE = False  # per-step progress bars off for the sweep; short status per run.

n_trials = 5     # independent trials per M. Different init each trial, but the
                 # SAME init for BOTH algorithms within a trial. Table below
                 # reports mean ± std over these trials.
WARMUP = True    # one throwaway run of each method to absorb CUDA context /
                 # kernel-compilation cost before any timing is recorded.


def _sync():
    """Block until queued GPU work finishes, so wall-clock timing is accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    """Run fn() and return (result, wall_seconds), synchronizing on CUDA."""
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t0


def mean_std(xs):
    """Sample mean and ddof=1 std of a list of floats (std=0 for a single point)."""
    n = len(xs)
    m = sum(xs) / n
    s = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s

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
# Timed benchmark: particle-count sweep x independent trials
# ══════════════════════════════════════════════════════════════════════════════


def run_svgd(init_particles, M, n_steps, verbose=VERBOSE):
    return svgd(
        post_log_dens,
        M,
        n_steps,
        dim=2,
        lr=lr_svgd,
        init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        keep_all=True,
        compile_step=False,
        verbose=verbose,
    )


def run_msip(init_particles, M, n_steps, verbose=VERBOSE):
    msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
    return msip(
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
        verbose=verbose,
    )


# Warmup (untimed): absorbs one-off CUDA/JIT startup so the first measured run
# isn't inflated. Does not affect results — every trial re-seeds before its init.
if WARMUP:
    print("Warmup ...")
    torch.manual_seed(SEED)
    _w_init = init_mean + init_std * torch.randn((16, 2))
    run_svgd(_w_init, 16, 50, verbose=False)
    run_msip(_w_init, 16, 50, verbose=False)
    _sync()
    del _w_init

# One record per (M, trial). Within a trial both methods share the SAME init;
# timing is full-run wall-clock (CUDA-synchronized). keep_all=True stores every
# step, so the timing includes that storage overhead — symmetric across methods.
records = []
for M in particle_counts:
    print(f"\n===== M = {M} particles =====")
    for trial in range(n_trials):
        seed = SEED + trial
        torch.manual_seed(seed)
        init_particles = init_mean + init_std * torch.randn((M, 2))  # shared init

        traj_svgd, t_svgd = timed(lambda: run_svgd(init_particles, M, n_steps))
        (traj_msip, wts_msip), t_msip = timed(
            lambda: run_msip(init_particles, M, n_steps)
        )

        pts_svgd = traj_svgd[-1]
        pts_msip = traj_msip[-1]
        w_msip = wts_msip[-1]

        rec = dict(
            M=M,
            trial=trial,
            seed=seed,
            all_pos=bool((w_msip > 0).all().item()),
            min_w=w_msip.min().item(),
            mmd_msip_wt=compute_mmd(pts_msip, mmd_bandwidth, weights=w_msip),
            mmd_msip_unwt=compute_mmd(pts_msip, mmd_bandwidth),
            mmd_svgd=compute_mmd(pts_svgd, mmd_bandwidth),
            t_msip=t_msip,
            t_svgd=t_svgd,
        )
        records.append(rec)
        print(
            f"  trial {trial:>2} | all>0={str(rec['all_pos']):>5} "
            f"min_w={rec['min_w']:>9.2e} | "
            f"MMD wt={rec['mmd_msip_wt']:.3e} unwt={rec['mmd_msip_unwt']:.3e} "
            f"SVGD={rec['mmd_svgd']:.3e} | "
            f"t_MSIP={t_msip:7.3f}s t_SVGD={t_svgd:7.3f}s"
        )

        del traj_svgd, traj_msip, wts_msip  # free before the next run


# --- Aggregate over trials, per M --------------------------------------------
agg = []
for M in particle_counts:
    rows = [r for r in records if r["M"] == M]
    wt_m, wt_s = mean_std([r["mmd_msip_wt"] for r in rows])
    unwt_m, unwt_s = mean_std([r["mmd_msip_unwt"] for r in rows])
    svgd_m, svgd_s = mean_std([r["mmd_svgd"] for r in rows])
    tmsip_m, tmsip_s = mean_std([r["t_msip"] for r in rows])
    tsvgd_m, tsvgd_s = mean_std([r["t_svgd"] for r in rows])
    agg.append(
        dict(
            M=M,
            n=len(rows),
            n_allpos=sum(r["all_pos"] for r in rows),
            worst_min_w=min(r["min_w"] for r in rows),
            wt_m=wt_m, wt_s=wt_s,
            unwt_m=unwt_m, unwt_s=unwt_s,
            svgd_m=svgd_m, svgd_s=svgd_s,
            tmsip_m=tmsip_m, tmsip_s=tmsip_s,
            tsvgd_m=tsvgd_m, tsvgd_s=tsvgd_s,
        )
    )


# --- Aggregate table (mean +/- std over trials) ------------------------------
def pm(m, s, w, mf=".2e", sf=".1e"):
    """Format 'mean±std' right-justified to width w."""
    return f"{format(m, mf)}±{format(s, sf)}".rjust(w)


title = (
    f"Final-iteration results, mean ± std over {n_trials} trials  "
    f"(RBF metric σ = {mmd_bandwidth}, n_steps = {n_steps})"
)
hdr = (
    f"{'M':>4} | {'all>0':>5} | {'worst min w':>11} | "
    f"{'MMD MSIP (wt)':>16} | {'MMD MSIP (unwt)':>16} | {'MMD SVGD':>16} | "
    f"{'t MSIP (s)':>13} | {'t SVGD (s)':>13}"
)
rule = "=" * len(hdr)

print("\n" + rule)
print(title)
print(rule)
print(hdr)
print("-" * len(hdr))
for a in agg:
    frac = f"{a['n_allpos']}/{a['n']}"
    print(
        f"{a['M']:>4} | {frac:>5} | {a['worst_min_w']:>11.2e} | "
        f"{pm(a['wt_m'], a['wt_s'], 16)} | {pm(a['unwt_m'], a['unwt_s'], 16)} | "
        f"{pm(a['svgd_m'], a['svgd_s'], 16)} | "
        f"{pm(a['tmsip_m'], a['tmsip_s'], 13, mf='.3f', sf='.3f')} | "
        f"{pm(a['tsvgd_m'], a['tsvgd_s'], 13, mf='.3f', sf='.3f')}"
    )
print(rule)

# --- CSV: raw per-trial data (long format) -----------------------------------
per_trial_csv = "per_trial_MSIP-Fredholm-vs-SVGD.csv"
with open(per_trial_csv, "w") as f:
    f.write(
        "n_particles,trial,seed,msip_all_weights_positive,msip_min_weight,"
        "mmd_msip_weighted,mmd_msip_unweighted,mmd_svgd,time_msip_s,time_svgd_s\n"
    )
    for r in records:
        f.write(
            f"{r['M']},{r['trial']},{r['seed']},{int(r['all_pos'])},"
            f"{r['min_w']:.8e},{r['mmd_msip_wt']:.8e},{r['mmd_msip_unwt']:.8e},"
            f"{r['mmd_svgd']:.8e},{r['t_msip']:.6f},{r['t_svgd']:.6f}\n"
        )
print(f"Saved {per_trial_csv}")

# --- CSV: aggregated summary (mirrors the table) -----------------------------
summary_csv = "summary_MSIP-Fredholm-vs-SVGD.csv"
with open(summary_csv, "w") as f:
    f.write(
        "n_particles,n_trials,msip_all_positive_count,msip_worst_min_weight,"
        "mmd_msip_weighted_mean,mmd_msip_weighted_std,"
        "mmd_msip_unweighted_mean,mmd_msip_unweighted_std,"
        "mmd_svgd_mean,mmd_svgd_std,"
        "time_msip_s_mean,time_msip_s_std,time_svgd_s_mean,time_svgd_s_std\n"
    )
    for a in agg:
        f.write(
            f"{a['M']},{a['n']},{a['n_allpos']},{a['worst_min_w']:.8e},"
            f"{a['wt_m']:.8e},{a['wt_s']:.8e},"
            f"{a['unwt_m']:.8e},{a['unwt_s']:.8e},"
            f"{a['svgd_m']:.8e},{a['svgd_s']:.8e},"
            f"{a['tmsip_m']:.6f},{a['tmsip_s']:.6f},"
            f"{a['tsvgd_m']:.6f},{a['tsvgd_s']:.6f}\n"
        )
print(f"Saved {summary_csv}")

# --- Plot: MMD vs M (mean +/- std) -------------------------------------------
Ms = [a["M"] for a in agg]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(Ms, [a["wt_m"] for a in agg], yerr=[a["wt_s"] for a in agg],
            fmt="s-", capsize=3, label="MSIP-Fredholm (weighted)")
ax.errorbar(Ms, [a["unwt_m"] for a in agg], yerr=[a["unwt_s"] for a in agg],
            fmt="^--", capsize=3, label="MSIP-Fredholm (unweighted)")
ax.errorbar(Ms, [a["svgd_m"] for a in agg], yerr=[a["svgd_s"] for a in agg],
            fmt="o-", capsize=3, label="SVGD")
ax.set_yscale("log")
# y-axis is linear; add ax.set_yscale("log") here to switch it back to log.
ax.set_xticks(Ms)
ax.set_xticklabels([str(m) for m in Ms])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("number of particles  M")
ax.set_ylabel(f"MMD  (RBF, σ={mmd_bandwidth})")
ax.set_title(f"MMD vs particle count over {n_trials} trials")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("MMD_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")
plt.close()
print("Saved MMD_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")

# --- Plot: runtime vs M (mean +/- std, log-log) ------------------------------
def _lower_err(means, stds):
    # keep the lower whisker strictly positive on a log axis
    return [min(s, m * (1 - 1e-9)) for m, s in zip(means, stds)]

tmsip_m = [a["tmsip_m"] for a in agg]
tmsip_s = [a["tmsip_s"] for a in agg]
tsvgd_m = [a["tsvgd_m"] for a in agg]
tsvgd_s = [a["tsvgd_s"] for a in agg]


# def loglog_slope(xs, ys):
#     """Least-squares slope in log-log space = empirical exponent alpha (t ~ M^a)."""
#     lx = [math.log(x) for x in xs]
#     ly = [math.log(y) for y in ys]
#     n = len(lx)
#     mx, my = sum(lx) / n, sum(ly) / n
#     sxx = sum((x - mx) ** 2 for x in lx)
#     sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
#     return sxy / sxx if sxx else float("nan")


def loglog_fit(xs, ys):
    """Least-squares power-law fit  y ~ C * x**alpha ; returns (alpha, C)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    alpha = sxy / sxx if sxx else float("nan")
    C = math.exp(my - alpha * mx)          # intercept, back to linear scale
    return alpha, C



# Fit each method to its own power law  t = C * M**alpha  (least squares,
# log-log). Use the tail — e.g. loglog_fit(Ms[-3:], tmsip_m[-3:]) — if the
# small-M points are dominated by fixed per-call overhead and you want the
# asymptotic slope rather than the whole-range average.
alpha_msip, C_msip = loglog_fit(Ms, tmsip_m)
alpha_svgd, C_svgd = loglog_fit(Ms, tsvgd_m)
print(
    f"Fitted runtime power law (t ~ C*M^alpha):  "
    f"MSIP-Fredholm: alpha={alpha_msip:.2f} C={C_msip:.2e}   "
    f"SVGD: alpha={alpha_svgd:.2f} C={C_svgd:.2e}"
)

# Log-spaced grid so the fit lines are smooth (straight on log-log, curved on
# linear-x — either way they land on the data, since C is fitted too).
Mgrid = [Ms[0] * (Ms[-1] / Ms[0]) ** (i / 200) for i in range(201)]

# Fit each method to its own power law  t = C * M**alpha  (least squares,
# log-log). Use the tail — e.g. loglog_fit(Ms[-3:], tmsip_m[-3:]) — if the
# small-M points are dominated by fixed per-call overhead and you want the
# asymptotic slope rather than the whole-range average.
alpha_msip, C_msip = loglog_fit(Ms, tmsip_m)
alpha_svgd, C_svgd = loglog_fit(Ms, tsvgd_m)
print(
    f"Fitted runtime power law (t ~ C*M^alpha):  "
    f"MSIP-Fredholm: alpha={alpha_msip:.2f} C={C_msip:.2e}   "
    f"SVGD: alpha={alpha_svgd:.2f} C={C_svgd:.2e}"
)

# Log-spaced grid so the fit lines are smooth (straight on log-log, curved on
# linear-x — either way they land on the data, since C is fitted too).
Mgrid = [Ms[0] * (Ms[-1] / Ms[0]) ** (i / 200) for i in range(201)]

# Reference power laws M^alpha to overlay (edit to taste). They share one anchor
# at the smallest M on the MSIP curve, so each is a straight line of slope alpha
# on log-log axes; whichever a measured curve runs parallel to is its exponent.
ref_alphas = [1, 2, 3]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(Ms, tmsip_m, yerr=[_lower_err(tmsip_m, tmsip_s), tmsip_s],
            fmt="s-", capsize=3, label="MSIP-Fredholm")
ax.errorbar(Ms, tsvgd_m, yerr=[_lower_err(tsvgd_m, tsvgd_s), tsvgd_s],
            fmt="o-", capsize=3, label="SVGD")


# Pin the y-range to the data (with headroom) so steep reference guides don't
# blow up the axis; each guide still shows its slope within the data window.
ylo = min(min(tmsip_m), min(tsvgd_m))
yhi = max(max(tmsip_m), max(tsvgd_m))
ax.set_ylim(ylo / 1.4, yhi * 1.4)

M0, t0 = Ms[0], tmsip_m[0]  # anchor: smallest M, MSIP runtime
# Best-fit power law per method, drawn dashed in the matching data colour
# (C0 = MSIP-Fredholm, C1 = SVGD, since those errorbars are plotted first/second).
#ax.plot(Mgrid, [C_msip * M ** alpha_msip for M in Mgrid], "--", color="C0",
#        lw=1.2, alpha=0.8, zorder=0,
#        label=rf"MSIP fit $\propto M^{{{alpha_msip:.2f}}}$")
#ax.plot(Mgrid, [C_svgd * M ** alpha_svgd for M in Mgrid], "--", color="C1",
#        lw=1.2, alpha=0.8, zorder=0,
#        label=rf"SVGD fit $\propto M^{{{alpha_svgd:.2f}}}$")


#ax.set_xscale("log")  # power laws are straight lines only on log-log axes
ax.set_yscale("log")

# for alpha in ref_alphas:
#     guide = [t0 * (M / M0) ** alpha for M in Ms]
#     ax.plot(Ms, guide, "--", lw=1.0, color="0.45", alpha=0.7, zorder=0)
#     vis = [(M, g) for M, g in zip(Ms, guide) if ylo / 1.4 <= g <= yhi * 1.4]
#     if vis:
#         Ma, ga = vis[-1]
#         ax.annotate(
#             rf"$\propto M^{{{alpha}}}$",
#             xy=(Ma, ga), xytext=(3, 3), textcoords="offset points",
#             fontsize=8, color="0.35",
#         )
ax.set_xticks(Ms)
ax.set_xticklabels([str(m) for m in Ms])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("number of particles  M")
ax.set_ylabel("wall-clock time per run (s)")
ax.set_title(
    f"Runtime vs particle count over {n_trials} trials "
    f"(n_steps={n_steps})"
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("time_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")
plt.close()
print("Saved time_vs_nparticles_MSIP-Fredholm-vs-SVGD.pdf")
# %%