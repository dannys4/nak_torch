from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sized

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import nn
import torch.utils.data as torch_data
from torch.nn.utils import vector_to_parameters
from torch.func import functional_call
from collections import OrderedDict

from tqdm import tqdm

from nak_torch.tools.types import DeviceLike

# ══════════════════════════════════════════════════════════════════════════════
# BNN model
# ══════════════════════════════════════════════════════════════════════════════


class bnn(nn.Module):
    """
    Standard MLP for binary classification.
    Output is a raw logit; apply sigmoid to get probabilities.

    Parameters
    ----------
    in_dim     : int   input dimension
    hidden_dim : int   width of each hidden layer
    n_layers   : int   number of hidden layers
    """

    hidden_out_dims: list[int]
    n_layers: int

    def __init__(
        self, in_dim: int, hidden_dim: int | list[int], out_dim: int, n_layers: int = 1
    ):
        super().__init__()
        layers = []
        hidden_out_dims: list[int]
        if isinstance(hidden_dim, int):
            hidden_out_dims = [hidden_dim for _ in range(n_layers)]
        elif len(hidden_dim) != n_layers:
            raise ValueError(
                f"Expected len(hidden_dim) = {n_layers}, got {len(hidden_dim)}"
            )
        else:
            hidden_out_dims = hidden_dim
        self.hidden_out_dims, self.n_layers = hidden_out_dims, n_layers
        for out_dim in hidden_out_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_out_dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.net(xb).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading with train-test splitting
# ══════════════════════════════════════════════════════════════════════════════


def load_dataset(
    dataset_name: str,
    train_ratio: float,
    rng: torch.Generator,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLike] = None,
):
    """
    Loads  datasets/{dataset_name}.npz  (keys X: (d,N), Y: (1,N))
    and splits into train / test.

    Returns X_train, Y_train, X_test, Y_test as torch.double tensors
    with shapes (N, d) and (N,)
    """
    data = np.load(f"data/datasets/{dataset_name}.npz")
    X = torch.from_numpy(data["X"].T).to(dtype=dtype, device=device)
    Y = torch.from_numpy(data["Y"].T).to(dtype=dtype, device=device).squeeze()

    N_total = X.shape[0]
    n_train = int(N_total * train_ratio)
    idx = torch.randperm(N_total, generator=rng)

    i_train, i_test = idx[:n_train], idx[n_train:]

    return X[i_train], Y[i_train], X[i_test], Y[i_test]


def theta_to_param_dict(theta_1d, param_info):
    out, i = OrderedDict(), 0
    for name, shape, numel in param_info:
        out[name] = theta_1d[i : i + numel].view(shape)
        i += numel
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Objective functions
# ══════════════════════════════════════════════════════════════════════════════
def bnn_evaluator(theta_1d, model, buffer_dict, param_info, data):
    param_dict = theta_to_param_dict(theta_1d, param_info)
    pred = functional_call(model, (param_dict, buffer_dict), (data,))
    return pred


bnn_evaluator_v = torch.vmap(bnn_evaluator, in_dims=(0, None, None, None, None))


def soft_margin_loss(x, y):
    return torch.mean(F.softplus(-x * y))


soft_margin_v = torch.vmap(soft_margin_loss, in_dims=(0, None))


def cross_entropy(x, y):
    return F.cross_entropy(x, y)


cross_entropy_v = torch.vmap(cross_entropy, in_dims=(0, None))


class BNNPosterior(ABC):
    """
    A log-posterior for Bayesian neural networks

    Parameters
    ----------
    data, labels : torch.double tensors  (N, d) and (N,)
    model_class  : 'bnn' for now, but we can think about something else
    hidden_dim   : int   hidden width
    n_layers     : int   depth
    beta         : float temperature
    weight_decay : float prior weight
    """

    model: bnn
    param_info: list[tuple[str, torch.Size, int]]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | list[int],
        n_layers: int,
        beta: float,
        weight_decay: float,
    ):
        self.beta = beta
        self.lambda2 = weight_decay
        self.model = bnn(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers
        ).double()
        self.param_info = [
            (name, p.shape, p.numel()) for name, p in self.model.named_parameters()
        ]
        self.buffer_dict = OrderedDict(self.model.named_buffers())
        self.dimension = sum(nu for _, _, nu in self.param_info)

    def __repr__(self):
        return (
            f"  Model   : BNNClassifierPosterior  |  "
            f"hidden_dim = {self.model.hidden_dim}  n_layers = {self.model.n_layers}  |  "
            f"dim(theta) = {self.dimension}"
        )

    @abstractmethod
    def __call__(self, theta: Tensor, dataset: Optional[tuple[Tensor, ...]]): ...

    @abstractmethod
    def get_data_loader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        *data_loader_args,
        **data_loader_kwargs,
    ) -> torch.utils.data.DataLoader: ...


class BNNClassifierPosterior(BNNPosterior):
    """
    A log-posterior for BNN for classification

    Parameters
    ----------
    data, labels : torch.double tensors  (N, d) and (N,)
    model_class  : 'bnn' for now, but we can think about something else
    hidden_dim   : int   hidden width
    n_layers     : int   depth
    beta         : float temperature
    weight_decay : float prior weight
    """

    model: bnn
    param_info: list[tuple[str, torch.Size, int]]

    def __init__(
        self,
        data: Float[Tensor, "N_samples dim"],
        labels: Float[Tensor, " N_samples"],
        hidden_dim: int = 10,
        n_layers: int = 1,
        beta: float = 1.0,
        weight_decay: float = 0.01,
    ):
        self.data = data
        self.labels = labels
        super().__init__(data.shape[1], 1, hidden_dim, n_layers, beta, weight_decay)

    def __repr__(self):
        return (
            f"  Model   : BNNClassifierPosterior  |  "
            f"hidden_dim = {self.model.hidden_dim}  n_layers = {self.model.n_layers}  |  "
            f"dim(theta) = {self.dimension}"
        )

    def __call__(self, theta: Tensor, dataset: Optional[tuple[Tensor, ...]] = None):
        single_theta = theta.ndim == 1
        if single_theta:
            theta = theta.unsqueeze(0)
        elif theta.ndim > 2:
            raise ValueError(f"theta.ndim must be 1 or 2. Got {theta.ndim}")

        if dataset is None:
            data, labels = self.data, self.labels
        else:
            if len(dataset) != 2:
                raise ValueError(
                    f"Expected dataset=(data,labels), got len(dataset)={len(dataset)}"
                )
            data, labels = dataset

        pred: Tensor = bnn_evaluator_v(
            theta, self.model, self.buffer_dict, self.param_info, data
        )
        data_loss = soft_margin_v(pred, labels) * self.data.shape[0]
        reg = self.lambda2 * (theta**2).sum(-1)
        # Post = -(soft_margin(theta) + lambda_2 ||theta||^2)
        return data_loss.add_(reg).div_(-self.beta).squeeze_()

    def get_data_loader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        *data_loader_args,
        **data_loader_kwargs,
    ):
        import torch.utils.data as torch_data

        data: torch_data.TensorDataset
        data = torch_data.TensorDataset(self.data, self.labels)
        return torch_data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *data_loader_args,
            **data_loader_kwargs,
        )


class BNNMultiClassifierPosterior(BNNPosterior):
    """
    A log-posterior for BNN for classification

    Parameters
    ----------
    data, labels : torch.double tensors  (N, d) and (N,)
    model_class  : 'bnn' for now, but we can think about something else
    hidden_dim   : int   hidden width
    n_layers     : int   depth
    beta         : float temperature
    weight_decay : float prior weight
    """

    dataset: torch_data.Dataset
    N_data_total: int

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_labels: int,
        hidden_dims: int | list[int] = [128, 64],
        n_layers: int = 1,
        beta: float = 1.0,
        weight_decay: float = 0.01,
        N_data_total: int = 0,
        data_dim: int = 0,
    ):
        self.dataset = dataset
        if N_data_total <= 0:
            if isinstance(dataset, Sized):
                self.N_data_total = len(dataset)
            else:
                raise NotImplementedError
        else:
            self.N_data_total = N_data_total

        if data_dim <= 0:
            data_dim = dataset.data.shape[0]  # type: ignore

        super().__init__(
            data_dim,
            n_labels,
            hidden_dim=hidden_dims,
            n_layers=n_layers,
            beta=beta,
            weight_decay=weight_decay,
        )

    def __repr__(self):
        return (
            f"  Model   : BNNMultiClassifierPosterior  |  "
            f"hidden_dim = {self.model.hidden_dim}  n_layers = {self.model.n_layers}  |  "
            f"dim(theta) = {self.dimension}  n_labels = {self.n_labels}"
        )

    def __call__(self, theta: Tensor, dataset: Optional[tuple[Tensor, ...]] = None):
        single_theta = theta.ndim == 1
        if single_theta:
            theta = theta.unsqueeze(0)
        elif theta.ndim > 2:
            raise ValueError(f"theta.ndim must be 1 or 2. Got {theta.ndim}")

        if dataset is None:
            data_loader_all = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=len(self.dataset),  # type: ignore
            )
            data, labels = next(iter(data_loader_all))
            if data is None or labels is None:
                raise ValueError(
                    "Expected dataset is not None or given (test_data, test_labels)"
                )
        else:
            if len(dataset) != 2:
                raise ValueError(
                    f"Expected dataset = (data, labels). Got len(dataset)={len(dataset)}"
                )
            data, labels = dataset

        pred: Tensor = bnn_evaluator_v(
            theta, self.model, self.buffer_dict, self.param_info, data
        )
        data_loss = cross_entropy_v(pred, labels) * self.N_data_total
        reg = self.lambda2 * (theta**2).sum(-1)
        return data_loss.add_(reg).div_(-self.beta).squeeze_()

    def get_data_loader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        *data_loader_args,
        **data_loader_kwargs,
    ):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *data_loader_args,
            **data_loader_kwargs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tools for handling grids
# ══════════════════════════════════════════════════════════════════════════════


def _make_grid(bounds, M_res):
    a, b = bounds
    xs, ys = np.linspace(a, b, M_res), np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1), dtype=torch.double)
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
    ax.scatter(x[y > 0, 0], x[y > 0, 1], c="gold", s=25, zorder=5)
    ax.scatter(x[y < 0, 0], x[y < 0, 1], c="tomato", s=25, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# Visualization tools
# ══════════════════════════════════════════════════════════════════════════════


def plot_boundaries(
    trajectories_dict, objective_fns_dict, X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60
):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 5))
    if M == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        for pi in probs:
            ax.contour(
                Xg,
                Yg,
                np.sign(pi - 0.5).reshape(M_res, M_res),
                levels=[0],
                colors=["steelblue"],
                linewidths=0.6,
                alpha=0.3,
            )
        p_bar = probs.mean(0).reshape(M_res, M_res)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=["black"], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.suptitle("Decision boundary (final iteration)", fontsize=14)
    plt.savefig("de_vs_msip_boundaries.pdf")
    plt.tight_layout()
    plt.show()


def plot_mean_prediction(
    trajectories_dict, objective_fns_dict, X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60
):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 5))
    if M == 1:
        axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        im = ax.imshow(
            p_bar,
            extent=[*bounds, *bounds],
            origin="lower",
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=ax)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=["white"], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.suptitle("Mean predictive probability", fontsize=14)
    plt.savefig("de_vs_msip_mean_predictive_p.pdf")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluations on the dataset
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(trajectories_dict, objective_fns_dict, X, Y, split_name="test"):
    y_true = Y.numpy() > 0
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


def eval_function_trajectories(obj_fn, trajectories, algo_name):
    T, M, d = trajectories.shape
    eval_tensor = torch.zeros(T, M, dtype=trajectories.dtype)
    prog = tqdm(total=T * M)
    for t in range(T):
        for m in range(M):
            eval_tensor[t, m] = -obj_fn(trajectories[t, m, :], None)
            prog.update(1)
    prog.close()
    plt.plot(eval_tensor.detach().numpy().min(1), label=algo_name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title("Evaluation of the best particle for " + algo_name)
    plt.savefig("best_particle_" + algo_name + ".pdf")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
