import torch
import numpy as np
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, msip_gs, kfrflow, deepensembles, svgd
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from torch import nn
import random
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from datetime import datetime
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict
import torch._dynamo
torch._dynamo.config.suppress_errors = True


# ══════════════════════════════════════════════════════════════════════════════
# KSD computation
# ══════════════════════════════════════════════════════════════════════════════

def rbf_stein_kernel(x_i, x_j, score_i, score_j, h=1.0):
    """
    Computes the Stein kernel matrix element k_p(x_i, x_j) for RBF base kernel.
    """

    # Do we need to use heuristics for h?
    # Also, h here is a square of something
    # I need to check the expression one last time

    diff = x_i - x_j
    d = diff.numel()
    sq = (diff ** 2).sum()

    k_val = torch.exp(-sq / (2 * h))

    grad_k_xj = (diff / h) * k_val
    grad_k_xi = -(diff / h) * k_val
    trace = (d / h - sq / h ** 2) * k_val

    term1 = (score_i * score_j).sum() * k_val
    term2 = (score_i * grad_k_xj).sum()
    term3 = (grad_k_xi * score_j).sum()
    term4 = trace

    return term1 + term2 + term3 + term4


def imq_stein_kernel(x_i, x_j, score_i, score_j, c=1.0, beta=-0.5):
    """
    Computes the Stein kernel element k_p(x_i, x_j) for IMQ base kernel.
    """

    # Do we need to use heuristics for h?
    # Is this the canonical implementation of IMQ?

    # I need to check the expression one last time
    diff = x_i - x_j
    sq = (diff ** 2).sum()
    d = diff.numel()

    s = c**2 + sq
    k_val = s ** beta

    grad_k_xi = 2.0 * beta * (s ** (beta - 1.0)) * diff
    grad_k_xj = -grad_k_xi

    trace = (
        -2.0 * beta * d * (s ** (beta - 1.0))
        -4.0 * beta * (beta - 1.0) * sq * (s ** (beta - 2.0))
    )

    term1 = (score_i * score_j).sum() * k_val
    term2 = (score_i * grad_k_xj).sum()
    term3 = (grad_k_xi * score_j).sum()
    term4 = trace

    return term1 + term2 + term3 + term4


def polynomial_stein_kernel(x_i, x_j, score_i, score_j, c=1.0, m=2):
    """
    Computes the Stein kernel element k_p(x_i, x_j) for polynomial base kernel

        k(x_i, x_j) = (c + x_i^T x_j)^m

    Parameters
    ----------
    x_i, x_j   : (D,) tensors
    score_i    : (D,) tensor   ∇ log p(x_i)
    score_j    : (D,) tensor   ∇ log p(x_j)
    c          : float         offset parameter
    m          : int           polynomial degree

    Returns
    -------
    scalar tensor
    """
    if m < 1:
        raise ValueError(f"Polynomial degree m must be >= 1, got {m}")

    d = x_i.numel()
    dot = (x_i * x_j).sum()
    s = c + dot

    k_val = s ** m

    grad_k_xi = m * (s ** (m - 1)) * x_j
    grad_k_xj = m * (s ** (m - 1)) * x_i

    if m == 1:
        trace = c.new_tensor(0.0) if torch.is_tensor(c) else torch.tensor(0.0, dtype=x_i.dtype, device=x_i.device)
        trace = m * d * (s ** (m - 1))
    else:
        trace = m * (m - 1) * (s ** (m - 2)) * dot + m * d * (s ** (m - 1))

    term1 = (score_i * score_j).sum() * k_val
    term2 = (score_i * grad_k_xj).sum()
    term3 = (grad_k_xi * score_j).sum()
    term4 = trace

    return term1 + term2 + term3 + term4


def kernel_builder(kernel_params):

    if kernel_params['name'] == 'stein_rbf':
        h = kernel_params['h']

        def internal_stein_rbf_kernel(x_1, x_2, s_1, s_2):
            return rbf_stein_kernel(x_1, x_2, s_1, s_2, h)
        return internal_stein_rbf_kernel

    if kernel_params['name'] == 'stein_imq':
        c = kernel_params['c']
        beta = kernel_params['beta']

        def internal_stein_imq_kernel(x_1, x_2, s_1, s_2):
            return imq_stein_kernel(x_1, x_2, s_1, s_2, c, beta)
        return internal_stein_imq_kernel
    
    if kernel_params['name'] == 'stein_polynomial':
        c = kernel_params['c']
        m = kernel_params['m']
    
        def internal_stein_poly_kernel(x_1, x_2, s_1, s_2):
            return polynomial_stein_kernel(x_1, x_2, s_1, s_2, c, m)
        return internal_stein_poly_kernel

    raise ValueError(f"Unknown kernel name: {kernel_params['name']}")


def compute_ksd_arbitrary_kernel(particles, log_density_grad_val, kernel_params):
    """
    Computes the U-statistics estimate of the KSD^2 for an arbitrary kernel
    given a set of particles.

    Parameters
    ----------
    particles : tensor of shape (N, D)
    log_density_grad_val : callable
        Function returning (grad log p(x), log p(x)).
    kernel_params : dict
        Parameters used to build the Stein kernel.

    Returns
    -------
    float
        U-statistic estimate of KSD^2.
    """

    particles = particles.detach()
    N, D = particles.shape

    stein_kernel = kernel_builder(kernel_params)

    scores = []
    for i in range(N):
        g, _ = log_density_grad_val(particles[i])
        scores.append(g.detach())

    scores = torch.stack(scores)

    ksd2 = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ksd2 += stein_kernel(
                particles[i], particles[j],
                scores[i], scores[j]
            ).item()

    return ksd2 / (N * (N - 1))


def compute_ksd_along_trajectory(trajectory, log_density_grad_val, kernel_params, subsample=1):
    """
    Evaluates KSD^2 along a trajectory.
    """
    T = trajectory.shape[0]
    steps, ksds = [], []

    for t in range(0, T, subsample):
        ksd_val = compute_ksd_arbitrary_kernel(
            trajectory[t], log_density_grad_val, kernel_params
        )
        steps.append(t)
        ksds.append(ksd_val)
        print(f"  [{kernel_params['name']}] KSD² step {t:4d}: {ksd_val:.6f}")

    return steps, ksds


def plot_ksd_curves(ksd_results_dict):
    """
    Plots KSD over iterations for multiple algorithms.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    title = "Kernelized Stein Discrepancy"
    filename = "ksd_curves.pdf"

    for name, (steps, ksds, p1, p2, kernel_name) in ksd_results_dict.items():
        ax.semilogy(steps, ksds, lw=2, label=name)
        if kernel_name == "RBF":
            title = f"Kernelized Stein Discrepancy, RBF, h={p1}"
            filename = f"ksd_rbf_curves_h_{p1}.pdf"
        elif kernel_name == "IMQ":
            title = f"Kernelized Stein Discrepancy, IMQ, c={p1}, beta={p2}"
            filename = f"ksd_imq_curves_c_{p1}_beta_{p2}.pdf"
        elif kernel_name == "POLYNOMIAL":
            title = f"Kernelized Stein Discrepancy, POLYNOMIAL, c={p1}, m={p2}"
            filename = f"ksd_polynomial_curves_c_{p1}_m_{p2}.pdf"

    ax.set_xlabel("Iteration")
    ax.set_ylabel("KSD²")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ESS computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_ess_from_unnormalized_weights(unnormalized_w):
    """
    ESS uniform targets from unnormalized weights.

    Parameters
    ----------
    unnormalized_w : tensor of shape (..., N)

    Returns
    -------
    ess : tensor of shape unnormalized_w.shape[:-1]
    """
    w = unnormalized_w.detach().double()
    return (w.sum(dim=-1) ** 2) / w.pow(2).sum(dim=-1).clamp_min(1e-300)


def compute_ess_from_log_unnormalized_weights(log_unnormalized_w):
    """
    ESS for uniform targets from log unnormalized weights.

    Parameters
    ----------
    log_unnormalized_w : tensor of shape (..., N)

    Returns
    -------
    ess : tensor of shape log_unnormalized_w.shape[:-1]
    """
    lw = log_unnormalized_w.detach().double()
    m = lw.max(dim=-1, keepdim=True).values
    u = torch.exp(lw - m)
    return (u.sum(dim=-1) ** 2) / u.pow(2).sum(dim=-1).clamp_min(1e-300)


def compute_ess_particles(particles, obj_fn, q_weights=None):
    """
    Computes ESS for general target distributions:
    Effective unnormalized weights are
            \tilde w_i = a_i * gamma(x_i)
    where
        gamma(x_i) = exp(obj_fn(x_i)) are the unnormalized target density evaluations
        a_i are the quadrature weights

    Parameters
    ----------
    particles : tensor of shape (N, d)
    obj_fn : callable
        Returns log gamma(x) for one particle or a batch.
    q_weights : tensor of shape (N,), optional
        Quadrature weights a_i.
        If None, uses uniform weights 1/N.

    Returns
    -------
    ess : float
    """
    N = particles.shape[0]
    log_gamma = obj_fn(particles).detach().double()

    if q_weights is None:
        log_a = torch.full(
            (N,), -math.log(N), dtype=log_gamma.dtype, device=log_gamma.device
        )
    else:
        a = q_weights.detach().double()
        log_a = torch.log(a.clamp_min(1e-300))

    log_unnorm_w = log_a + log_gamma
    return compute_ess_from_log_unnormalized_weights(log_unnorm_w).item()


def compute_ess_along_trajectory(trajectory, obj_fn, q_weights_trajectory=None, subsample=1):
    """
    Computes ESS along a trajectory using both quadrature weights and density.

    Effective unnormalized weights at time t:
        \tilde w_i(t) = a_i(t) * exp(obj_fn(x_i(t)))

    Parameters
    ----------
    trajectory : tensor of shape (T, N, d)
    obj_fn : callable
        Returns log gamma(x) for a batch of particles.
    q_weights_trajectory : tensor of shape (T, N), optional
        If provided, uses these as quadrature weights a_i(t).
        If None, uses uniform weights at each iteration.
    subsample : int

    Returns
    -------
    steps : list[int]
    ess_vals : list[float]
    """
    T = trajectory.shape[0]
    steps, ess_vals = [], []

    for t in range(0, T, subsample):
        particles_t = trajectory[t]
        weights_t = None if q_weights_trajectory is None else q_weights_trajectory[t]
        ess_t = compute_ess_particles(particles_t, obj_fn, q_weights=weights_t)
        steps.append(t)
        ess_vals.append(ess_t)
        print(f"  ESS step {t:4d}: {ess_t:.6f}")

    return steps, ess_vals


def plot_ess_curves(ess_results_dict, filename="ess_curves.pdf"):
    """
    Plots ESS over iterations for multiple algorithms.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, (steps, ess_vals) in ess_results_dict.items():
        ax.plot(steps, ess_vals, lw=2, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# BNN model
# ══════════════════════════════════════════════════════════════════════════════

class bnn(nn.Module):
    """
    Standard MLP for binary classification.
    Output is a raw logit; apply sigmoid to get probabilities.

    Parameters
    ----------
    d_in       : int   input dimension
    hidden_dim : int   width of each hidden layer
    n_layers   : int   number of hidden layers
    """
    def __init__(self, d_in: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.net(xb).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading with train-test splitting
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name, train_ratio=0.8, seed=0):
    """
    Loads datasets/{dataset_name}.npz (keys X: (d,N), Y: (1,N))
    and splits into train / test.

    Returns X_train, Y_train, X_test, Y_test as torch.double tensors
    with shapes (N, d) and (N,)
    """
    data = np.load(f"datasets/{dataset_name}.npz")
    X = torch.from_numpy(data["X"].T).double()
    Y = torch.from_numpy(data["Y"].T).double().squeeze()

    N_total = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N_total)
    n_train = int(N_total * train_ratio)

    i_train, i_test = idx[:n_train], idx[n_train:]

    return X[i_train], Y[i_train], X[i_test], Y[i_test]


# ══════════════════════════════════════════════════════════════════════════════
# Objective functions
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(X, Y, model_class='bnn',
                   hidden_dim=10, n_layers=1,
                   beta=1.0, lambda2=0.01):
    """
    Returns a log-posterior callable theta -> scalar
    """
    d = X.shape[1]

    if model_class == 'bnn':
        model = bnn(d_in=d, hidden_dim=hidden_dim, n_layers=n_layers).double()
    else:
        raise ValueError(f"Unknown model_class: {model_class!r}")

    param_info = [(name, p.shape, p.numel())
                  for name, p in model.named_parameters()]
    buffer_dict = OrderedDict(model.named_buffers())
    total_numel = sum(nu for _, _, nu in param_info)

    print(f"  Model   : {model_class}  |  "
          f"hidden_dim = {hidden_dim}  n_layers = {n_layers}  |  "
          f"dim(theta) = {total_numel}")

    def theta_to_param_dict(theta_1d):
        out, i = OrderedDict(), 0
        for name, shape, numel in param_info:
            out[name] = theta_1d[i:i + numel].view(shape)
            i += numel
        return out

    def loss_for_single_theta(theta_1d):
        param_dict = theta_to_param_dict(theta_1d)
        pred = functional_call(model, (param_dict, buffer_dict), (X,))
        pred = pred.squeeze()
        data_loss = F.soft_margin_loss(pred, Y)
        reg = lambda2 * (theta_1d ** 2).sum()
        return -(data_loss + reg) / beta

    def objective_function(theta):
        if theta.ndim == 1:
            return loss_for_single_theta(theta)
        elif theta.ndim == 2:
            return torch.stack([loss_for_single_theta(theta[i])
                                for i in range(theta.shape[0])])
        else:
            raise ValueError(f"theta must be 1D or 2D, got {theta.shape}")

    objective_function.total_numel = total_numel
    objective_function.model_class = model_class
    objective_function.model = model
    objective_function.param_info = param_info
    objective_function.buffer_dict = buffer_dict

    return objective_function


# ══════════════════════════════════════════════════════════════════════════════
# Tools for handling grids
# ══════════════════════════════════════════════════════════════════════════════

def _make_grid(bounds, M_res):
    a, b = bounds
    xs, ys = np.linspace(a, b, M_res), np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1),
                        dtype=torch.double)
    return Xg, Yg, grid


def _get_ensemble_probs(traj, obj_fn, grid):
    particles = traj[-1]
    model = obj_fn.model
    probs = []
    for i in range(len(particles)):
        vector_to_parameters(particles[i].double(), model.parameters())
        with torch.no_grad():
            probs.append(torch.sigmoid(model(grid).squeeze()).numpy())
    return np.stack(probs, axis=0)


def _scatter(ax, X, Y):
    x = X.numpy()
    y = Y.numpy()
    ax.scatter(x[y > 0, 0], x[y > 0, 1], c='gold', s=25, zorder=5)
    ax.scatter(x[y < 0, 0], x[y < 0, 1], c='tomato', s=25, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Visualization tools
# ══════════════════════════════════════════════════════════════════════════════

def plot_boundaries(trajectories_dict, objective_fns_dict,
                    X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 5))
    if M == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        for pi in probs:
            ax.contour(Xg, Yg, np.sign(pi - 0.5).reshape(M_res, M_res),
                       levels=[0], colors=['steelblue'], linewidths=0.6, alpha=0.3)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['black'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.suptitle("Decision boundary (final iteration)", fontsize=14)
    plt.savefig("de_vs_msip_boundaries.pdf")
    plt.tight_layout()
    plt.show()


def plot_mean_prediction(trajectories_dict, objective_fns_dict,
                         X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 5))
    if M == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        im = ax.imshow(p_bar, extent=[*bounds, *bounds], origin='lower',
                       cmap='RdBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['white'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.suptitle("Mean predictive probability", fontsize=14)
    plt.savefig("de_vs_msip_mean_predictive_p.pdf")
    plt.tight_layout()
    plt.show()


def plot_diversity_curve(trajectories_dict, subsample=10):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps, divs = [], []
        for t in range(0, T, subsample):
            P = traj[t]
            sq = ((P.unsqueeze(0) - P.unsqueeze(1)) ** 2).sum(-1)
            idx = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().min().item())
            steps.append(t)
        ax.plot(steps, divs, lw=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Smallest pairwise distance")
    ax.set_title("Particle diversity over training")
    plt.savefig("de_vs_msip_div.pdf")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluations on the dataset
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(trajectories_dict, objective_fns_dict, X, Y, split_name="test"):
    y_true = (Y.numpy() > 0)
    grid = X.double()

    print(f"\n{'─' * 75}")
    print(f"  Evaluation on {split_name} set  ({len(X)} points)")
    print(f"{'─' * 75}")
    print(f"  {'Method':<12} {'Accuracy':>6} {'DAMV':>10}")
    print(f"{'─' * 75}")

    results = {}
    for name, traj in trajectories_dict.items():
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar = probs.mean(0)
        acc = ((p_bar > 0.5) == y_true).mean()
        damv = traj[-1].var(dim=0).mean().item()

        print(f"  {name:<12} {acc:>6.3f} {damv:>10.3f}")
        results[name] = dict(p_bar=p_bar, accuracy=acc, damv=damv)

    print(f"{'─' * 75}\n")
    return results


def plot_diversity_curve(trajectories_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps, divs = [], []
        for t in range(0, T, 1):
            P = traj[t]
            sq = ((P.unsqueeze(0) - P.unsqueeze(1)) ** 2).sum(-1)
            idx = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().min().item())
            steps.append(t)
        ax.plot(steps, divs, lw=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Smallest pairwise distance")
    ax.set_title("Particle diversity over training")
    plt.savefig("de_vs_msip_div.pdf")
    ax.legend()
    plt.tight_layout()
    plt.show()


def eval_best_function_trajectories(obj_fn, trajectories, algo_name):
    T, M, d = trajectories.shape
    eval_tensor = torch.zeros(T, M)
    for t in range(T):
        for m in range(M):
            eval_tensor[t, m] = -obj_fn(trajectories[t, m, :])

    plt.plot(eval_tensor.detach().numpy().min(1), label=algo_name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title("Evaluation of the best particle for " + algo_name)
    plt.savefig("best_particle_" + algo_name + ".pdf")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
def eval_function_trajectories(obj_fn, trajectories, algo_name):
    T, M, d = trajectories.shape
    eval_tensor = torch.zeros(T, M)
    for t in range(T):
        for m in range(M):
            eval_tensor[t, m] = -obj_fn(trajectories[t, m, :])

    plt.figure()
    for m in range(M):
        plt.plot(eval_tensor[:,m].detach().numpy())
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title("Evaluation of all the particles for " + algo_name)
    plt.savefig("all_particles_" + algo_name + ".pdf")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Config
    DATASET = 'two_bananas'
    MODEL_CLASS = 'bnn'
    HIDDEN_DIM = 50
    N_LAYERS = 1
    N_TRAIN = 0.8
    N_PARTICLES = 20
    N_STEPS = 1000
    BETA = 1.0
    LAMBDA2 = 0.00005
    LR_SVGD = 50e-2
    LR_MSIP = 50e-2
    SIGMA = 1.0
    GRADIENT_DECAY = 1.0

    # Data loading
    X_train, Y_train, X_test, Y_test = load_dataset(
        DATASET, train_ratio=N_TRAIN, seed=0)

    # Objective loading
    obj_msip = make_objective(
        X_train, Y_train, model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        beta=BETA, lambda2=LAMBDA2
    )

    obj_svgd = make_objective(
        X_train, Y_train, model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        beta=BETA, lambda2=LAMBDA2
    )

    obj_de = make_objective(
        X_train, Y_train, model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        beta=BETA, lambda2=LAMBDA2
    )

    dimension = obj_msip.total_numel

    # Shared initialization
    init_particles = torch.randn(N_PARTICLES, dimension).double()

    # Run MSIP
    post_log_dens_grad_val = torch.func.vmap(torch.func.grad_and_value(obj_msip))

    def mc_quad_rule(batch_size: int, N_quad: int = 50, dim: int = dimension):
        pts = torch.randn((batch_size, N_quad, dim)).double()
        wts = torch.ones((batch_size, N_quad)).div_(N_quad).double()
        return pts, wts

    msip_quadgrad = MSIPQuadGradientInformed(
        post_log_dens_grad_val, mc_quad_rule, GRADIENT_DECAY
    )
    #msip_gs
    trajectories_msip_qg, wts_msip_qg = msip(
        msip_quadgrad, N_PARTICLES, N_STEPS, dim=dimension,
        lr=LR_MSIP, init_particles=init_particles,
        kernel_length_scale=SIGMA, is_log_density_batched=True,
        kernel_diag_infl=1e-6, bounds=(-100., 100.),
        gradient_decay=GRADIENT_DECAY, keep_all=True,
        compile_step=False, verbose=True,
    )

    # Run SVGD
    trajectories_svgd = svgd(
        obj_svgd, N_PARTICLES, N_STEPS, dimension,
        LR_SVGD, seed=None, device=None, init_particles=init_particles,
        kernel_length_scale=SIGMA, keep_all=True,
        is_log_density_batched=True, verbose=True
    )

    # Run Deep Ensembles
    trajectories_de = deepensembles(
        obj_de, N_PARTICLES, N_STEPS, dimension,
        LR_SVGD, seed=None, device=None, init_particles=init_particles,
        kernel_length_scale=SIGMA, keep_all=True,
        is_log_density_batched=True, verbose=True
    )

    trajectories_dict = {
        "MSIP": trajectories_msip_qg,
        "SVGD": trajectories_svgd,
        "DE": trajectories_de
    }
    objective_fns_dict = {
        "MSIP": obj_msip,
        "SVGD": obj_svgd,
        "DE": obj_de
    }

    # Optimization diagnostics
    eval_best_function_trajectories(obj_msip, trajectories_msip_qg, "MSIP")
    eval_best_function_trajectories(obj_svgd, trajectories_svgd, "SVGD")
    eval_best_function_trajectories(obj_de, trajectories_de, "DE")


    eval_function_trajectories(obj_msip, trajectories_msip_qg, "MSIP")
    eval_function_trajectories(obj_svgd, trajectories_svgd, "SVGD")
    eval_function_trajectories(obj_de, trajectories_de, "DE")

    # Visualization
    plot_boundaries(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_mean_prediction(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_diversity_curve(trajectories_dict)

    # Evaluation on a dataset
    evaluate(trajectories_dict, objective_fns_dict, X_train, Y_train, "train")
    evaluate(trajectories_dict, objective_fns_dict, X_test, Y_test, "test")

    # ESS diagnostics ───────────────────────────────────────────────────────
    print("\n===== ESS diagnostics =====")

    print("\nComputing ESS along MSIP trajectory...")
    steps_msip_ess, ess_msip = compute_ess_along_trajectory(
        trajectories_msip_qg,
        obj_msip,
        q_weights_trajectory=wts_msip_qg,
        subsample=1
    )

    print("\nComputing ESS along SVGD trajectory...")
    steps_svgd_ess, ess_svgd = compute_ess_along_trajectory(
        trajectories_svgd,
        obj_svgd,
        q_weights_trajectory=None,
        subsample=1
    )

    print("\nComputing ESS along DE trajectory...")
    steps_de_ess, ess_de = compute_ess_along_trajectory(
        trajectories_de,
        obj_de,
        q_weights_trajectory=None,
        subsample=1
    )

    ess_results = {
        "MSIP": (steps_msip_ess, ess_msip),
        "SVGD": (steps_svgd_ess, ess_svgd),
        "DE": (steps_de_ess, ess_de),
    }
    plot_ess_curves(ess_results, filename="ess_curves.pdf")

    # KSD diagnostics ───────────────────────────────────────────────────────
    score_fn_msip = torch.func.grad_and_value(obj_msip)
    score_fn_svgd = torch.func.grad_and_value(obj_svgd)
    score_fn_de = torch.func.grad_and_value(obj_de)

    # ===== RBF KSD =====
    print("\n===== RBF KSD =====")

    for h in [2.0, 5.0, 10.0, 20.0, 50.0]:

        kernel_params = {
            "name": "stein_rbf",
            "h": h,
        }

        print(f"\n--- RBF bandwidth h = {h} ---")

        print("\nComputing KSD along MSIP trajectory...")
        steps_msip, ksds_msip = compute_ksd_along_trajectory(
            trajectories_msip_qg, score_fn_msip, kernel_params, subsample=1
        )

        print("\nComputing KSD along SVGD trajectory...")
        steps_svgd, ksds_svgd = compute_ksd_along_trajectory(
            trajectories_svgd, score_fn_svgd, kernel_params, subsample=1
        )

        print("\nComputing KSD along DE trajectory...")
        steps_de, ksds_de = compute_ksd_along_trajectory(
            trajectories_de, score_fn_de, kernel_params, subsample=1
        )

        ksd_results = {
            "MSIP": (steps_msip, ksds_msip, h, None, "RBF"),
            "SVGD": (steps_svgd, ksds_svgd, h, None, "RBF"),
            "DE": (steps_de, ksds_de, h, None, "RBF"),
        }
        plot_ksd_curves(ksd_results)

    # ===== IMQ KSD =====
    print("\n===== IMQ KSD =====")

    for ksd_c in [10.0, 5.0, 2.0, 1.0]:
        ksd_beta = -0.5

        kernel_params = {
            "name": "stein_imq",
            "c": ksd_c,
            "beta": ksd_beta,
        }

        print(f"\n--- IMQ parameters c = {ksd_c}, beta = {ksd_beta} ---")

        print("\nComputing KSD along MSIP trajectory...")
        steps_msip, ksds_msip = compute_ksd_along_trajectory(
            trajectories_msip_qg, score_fn_msip, kernel_params, subsample=1
        )

        print("\nComputing KSD along SVGD trajectory...")
        steps_svgd, ksds_svgd = compute_ksd_along_trajectory(
            trajectories_svgd, score_fn_svgd, kernel_params, subsample=1
        )

        print("\nComputing KSD along DE trajectory...")
        steps_de, ksds_de = compute_ksd_along_trajectory(
            trajectories_de, score_fn_de, kernel_params, subsample=1
        )

        ksd_results = {
            "MSIP": (steps_msip, ksds_msip, ksd_c, ksd_beta, "IMQ"),
            "SVGD": (steps_svgd, ksds_svgd, ksd_c, ksd_beta, "IMQ"),
            "DE": (steps_de, ksds_de, ksd_c, ksd_beta, "IMQ"),
        }
        plot_ksd_curves(ksd_results)


        # ksd_beta = -0.1

        # kernel_params = {
        #     "name": "stein_imq",
        #     "c": ksd_c,
        #     "beta": ksd_beta,
        # }

        # print(f"\n--- IMQ parameters c = {ksd_c}, beta = {ksd_beta} ---")

        # print("\nComputing KSD along MSIP trajectory...")
        # steps_msip, ksds_msip = compute_ksd_along_trajectory(
        #     trajectories_msip_qg, score_fn_msip, kernel_params, subsample=1
        # )

        # print("\nComputing KSD along SVGD trajectory...")
        # steps_svgd, ksds_svgd = compute_ksd_along_trajectory(
        #     trajectories_svgd, score_fn_svgd, kernel_params, subsample=1
        # )

        # print("\nComputing KSD along DE trajectory...")
        # steps_de, ksds_de = compute_ksd_along_trajectory(
        #     trajectories_de, score_fn_de, kernel_params, subsample=1
        # )

        # ksd_results = {
        #     "MSIP": (steps_msip, ksds_msip, ksd_c, ksd_beta, "IMQ"),
        #     "SVGD": (steps_svgd, ksds_svgd, ksd_c, ksd_beta, "IMQ"),
        #     "DE": (steps_de, ksds_de, ksd_c, ksd_beta, "IMQ"),
        # }
        # plot_ksd_curves(ksd_results)
        
    # ===== Polynomial KSD =====
    print("\n===== Polynomial KSD =====")

    for poly_c, poly_m in [(1.0, 1), (1.0, 2), (1.0, 2), (5.0, 1), (10.0, 1)]:

        kernel_params = {
            "name": "stein_polynomial",
            "c": poly_c,
            "m": poly_m,
        }

        print(f"\n--- Polynomial parameters c = {poly_c}, m = {poly_m} ---")

        print("\nComputing KSD along MSIP trajectory...")
        steps_msip, ksds_msip = compute_ksd_along_trajectory(
            trajectories_msip_qg, score_fn_msip, kernel_params, subsample=1
        )

        print("\nComputing KSD along SVGD trajectory...")
        steps_svgd, ksds_svgd = compute_ksd_along_trajectory(
            trajectories_svgd, score_fn_svgd, kernel_params, subsample=1
        )

        print("\nComputing KSD along DE trajectory...")
        steps_de, ksds_de = compute_ksd_along_trajectory(
            trajectories_de, score_fn_de, kernel_params, subsample=1
        )

        ksd_results = {
            "MSIP": (steps_msip, ksds_msip, poly_c, poly_m, "POLYNOMIAL"),
            "SVGD": (steps_svgd, ksds_svgd, poly_c, poly_m, "POLYNOMIAL"),
            "DE": (steps_de, ksds_de, poly_c, poly_m, "POLYNOMIAL"),
        }
        plot_ksd_curves(ksd_results)
