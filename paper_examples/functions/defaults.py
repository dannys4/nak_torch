from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
import os
import pickle
from typing import Any, Callable, Iterable, Iterator, Optional
import warnings

import torch
from torch import Tensor
from tqdm import tqdm

import nak_torch
from nak_torch.algorithms import (
    CBS,
    DeepEnsembles,
    EKS,
    MSIP,
    SVGD,
    GradALDI,
    GradFreeALDI,
)
from nak_torch.algorithms.msip.estimators import (
    MSIPFredholm,
    MSIPQuadGradientFree,
    MSIPQuadGradientInformed,
)
from nak_torch.tools.metrics import CrossEntropy, Metric
from nak_torch.tools.quadrature import gauss_MC
from dataclasses import dataclass, field

from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchLogDensityEvaluator,
    DeviceLike,
    GaussianModel,
)

DEFAULT_MSIP_KERNEL_LENGTHSCALE = 1e-1


def output_stack(t):
    if isinstance(t[0], tuple):
        points, weights = [torch.stack([pw[j] for pw in t]) for j in range(2)]
        return points.numpy(), weights.numpy()
    return torch.stack(t).numpy()


@dataclass
class Defaults:
    gradient_decay: float = 1.0
    quadrule: str = "monte carlo"
    msip_gi_quadrule_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(N_quad=1)
    )
    msip_gf_quadrule_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(N_quad=10)
    )
    metric: str = "cross entropy"
    metric_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    msip_kwargs = dict(
        kernel_lengthscale=DEFAULT_MSIP_KERNEL_LENGTHSCALE, kernel_diag_infl=1e-6
    )
    svgd_kwargs: dict[str, float] = field(
        default_factory=lambda: dict(kernel_lengthscale_quantile=0.5)
    )
    cbs_kwargs: dict[str, float] = field(
        default_factory=lambda: dict(default_inverse_temp=0.9)
    )
    bounds: tuple[float, float] = (-1000.0, 1000.0)
    n_particles: int = 25
    n_steps: int = 1000


def get_quadrule(name: str, **kwargs):
    if name == "monte carlo":
        return partial(gauss_MC, **kwargs)
    raise ValueError(f"Not implemented yet: {name}")


def get_metric(metric, log_dens: BatchLogDensityEvaluator, grad_log_dens: BatchGradLogDensityEvaluator) -> Metric:
    if metric == "cross entropy":
        return CrossEntropy(log_dens.log_density, is_log_dens_vectorized=True, is_log_dens_grad_val=False)
    raise NotImplementedError(f"Metric {metric} not implemented")

@dataclass
class Example(ABC):
    dim: int
    rng: torch.Generator
    log_dens_evaluator: BatchLogDensityEvaluator
    grad_log_dens_evaluator: BatchGradLogDensityEvaluator
    target_msip_f: MSIPFredholm
    target_msip_gi: MSIPQuadGradientInformed
    target_msip_gf: MSIPQuadGradientFree
    svgd: SVGD
    msip: MSIP
    deepens: DeepEnsembles
    grad_aldi: GradALDI
    cbs: CBS
    get_target_args_factory: Callable[[], Optional[Iterator]]

    def __init__(self, D: Defaults, device: Optional[DeviceLike] = None, dtype: Optional[torch.dtype] = None):
        grad_val = self.target_msip_f.log_dens_grad_val
        self.target_msip_gi = MSIPQuadGradientInformed(
            grad_val,
            get_quadrule(
                D.quadrule,
                d=self.dim,
                rng=self.rng,
                **D.msip_gi_quadrule_kwargs,
                device=device,
                dtype=dtype,
            ),
            D.gradient_decay,
        )
        self.target_msip_gf = MSIPQuadGradientFree(
            self.log_dens_evaluator.log_density,
            get_quadrule(
                D.quadrule,
                d=self.dim,
                rng=self.rng,
                **D.msip_gf_quadrule_kwargs,
                device=device,
                dtype=dtype,
            ),
        )
        all_kwargs = dict(
            dim=self.dim, n_particles=D.n_particles, device=device, dtype=dtype
        )
        self.svgd = SVGD(**all_kwargs, **D.svgd_kwargs)
        self.msip = MSIP(**all_kwargs, **D.msip_kwargs)
        self.cbs = CBS(**all_kwargs, rng=self.rng, **D.cbs_kwargs)
        self.deepens = DeepEnsembles(**all_kwargs)
        self.eks = EKS(**all_kwargs, rng_or_seed=self.rng)
        self.grad_aldi = GradALDI(**all_kwargs, rng=self.rng)
        self.gradfree_aldi = GradFreeALDI(**all_kwargs, rng_or_seed=self.rng)
        if not hasattr(self, "get_target_args_factory"):
            self.get_target_args_factory = lambda: None

    def is_gauss_model(self) -> bool:
        return False

    def get_algorithms(self) -> list[str]:
        ret = ["svgd", "msip_f", "msip_gi", "msip_gf", "cbs", "grad_aldi", "deepens"]
        if self.is_gauss_model():
            ret.extend(["eks", "gradfree_aldi"])
        return ret

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def init_particles(self, D: Defaults, device=None, dtype=None) -> Tensor: ...


@dataclass
class GaussExample(Example):
    gauss_model: GaussianModel
    eks: EKS
    gradfree_aldi: GradFreeALDI
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_gauss_model(self) -> bool:
        return False

    def init_particles(self, D, device=None, dtype=None):
        inv_std = torch.as_tensor(
            self.gauss_model.prior_precision, device=device, dtype=dtype
        ).sqrt()
        return (
            torch.randn(
                (D.n_particles, self.dim),
                device=device,
                dtype=dtype,
                generator=self.rng,
            )
            / inv_std
            + self.gauss_model.prior_mean
        )


def run_example(
    n_experiments: int,
    D: Defaults,
    rng: torch.Generator,
    ex: Example,
    lr: float,
    filepath: str,
    dtype: Optional[torch.dtype] = None,
):
    metric = get_metric(D.metric, ex.log_dens_evaluator, ex.grad_log_dens_evaluator)
    metric_evals = {alg: [] for alg in ex.get_algorithms()}
    outputs = {alg: [] for alg in ex.get_algorithms()}

    def run_nak(target, alg, init_particles, alg_name, metric_evals, outputs):
        res: Any
        try:
            res = nak_torch.nak(
                target, alg, D.n_steps, lr, rng, init_particles, D.bounds,
                get_target_args=ex.get_target_args_factory()
            )
            outputs[alg_name].append(res)
        except:  # noqa: E722
            warnings.warn(f"Problem running algorithm {alg_name}")
            return metric_evals, outputs
        try:
            iterator = zip(*res) if isinstance(res, tuple) else zip(res)
            metric_evals[alg_name].append(
                [
                    metric(*pw)
                    for pw in tqdm(iterator, desc=alg_name, total=D.n_steps + 1)
                ]
            )
        except:  # noqa: E722
            warnings.warn(f"Problem assessing algorithm {alg_name}")
        return metric_evals, outputs

    now_str = str(datetime.now()).replace(" ", "_").replace("-", "_")
    parent_dir = os.path.join(os.path.dirname(filepath), "data")
    fname = ex.name() + "_" + now_str + ".pkl"
    output_path = os.path.join(parent_dir, "result", fname)

    all_init_particles = []
    for r in range(n_experiments):
        print("\n\nStarting experiment {}\n\n".format(r))
        init_particles = ex.init_particles(D, dtype=dtype)
        all_init_particles.append(init_particles.numpy())

        # msip
        metric_evals, outputs = run_nak(
            ex.target_msip_f, ex.msip, init_particles, "msip_f", metric_evals, outputs
        )
        metric_evals, outputs = run_nak(
            ex.target_msip_gi, ex.msip, init_particles, "msip_gi", metric_evals, outputs
        )
        metric_evals, outputs = run_nak(
            ex.target_msip_gf, ex.msip, init_particles, "msip_gf", metric_evals, outputs
        )

        # svgd
        metric_evals, outputs = run_nak(
            ex.grad_log_dens_evaluator,
            ex.svgd,
            init_particles,
            "svgd",
            metric_evals,
            outputs,
        )

        # DeepEns
        metric_evals, outputs = run_nak(
            ex.grad_log_dens_evaluator,
            ex.deepens,
            init_particles,
            "deepens",
            metric_evals,
            outputs,
        )

        # aldi
        metric_evals, outputs = run_nak(
            ex.grad_log_dens_evaluator,
            ex.grad_aldi,
            init_particles,
            "grad_aldi",
            metric_evals,
            outputs,
        )

        # CBS
        metric_evals, outputs = run_nak(
            ex.log_dens_evaluator, ex.cbs, init_particles, "cbs", metric_evals, outputs
        )

        # Gauss Model Algorithms
        if isinstance(ex, GaussExample):
            # Gradfree ALDI
            metric_evals, outputs = run_nak(
                ex.gauss_model,
                ex.gradfree_aldi,
                init_particles,
                "grad_aldi",
                metric_evals,
                outputs,
            )

            # EKS
            metric_evals, outputs = run_nak(
                ex.gauss_model, ex.eks, init_particles, "eks", metric_evals, outputs
            )

        # Dump after each experiment in case things crash
        outputs_save = {k: output_stack(v) for (k, v) in outputs.items() if len(v) > 0}

        all_experiment_data = dict(
            lr=lr,
            init_particles=all_init_particles,
            metrics=metric_evals,
            hyperparams=D.__dict__,
        )

        with open(output_path, "wb") as f:
            pickle.dump(outputs_save, f)

        experiment_path = os.path.join(parent_dir, "experiment", fname)

        with open(experiment_path, "wb") as f:
            pickle.dump(all_experiment_data, f)
