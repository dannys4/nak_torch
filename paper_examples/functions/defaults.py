from abc import ABC
from functools import partial
from typing import Any, Callable, LiteralString, Optional

import torch

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
    BatchPtType,
    BatchQuadratureRule,
    GaussianModel,
)

DEFAULT_MSIP_KERNEL_LENGTHSCALE = 1e-1


@dataclass
class Defaults:
    gradient_decay: float = 1.0
    quadrule: str = "monte carlo"
    msip_gi_quadrule_kwargs: dict[str,Any] = field(default_factory=lambda:dict(N_quad=1))
    msip_gf_quadrule_kwargs: dict[str,Any] = field(default_factory=lambda:dict(N_quad=10))
    metric: Callable[[Any], Metric] = CrossEntropy
    msip_kwargs = dict(
        kernel_lengthscale=DEFAULT_MSIP_KERNEL_LENGTHSCALE, kernel_diag_infl=1e-6
    )
    svgd_kwargs: dict[str,float] = field(default_factory=lambda:dict(kernel_lengthscale_quantile=0.5))
    cbs_kwargs: dict[str,float] = field(default_factory=lambda:dict(default_inverse_temp=0.9))
    bounds: tuple[float, float] = (-1000.0, 1000.0)
    n_particles: int = 25
    n_steps: int = 1000

def get_quadrule(name: str, **kwargs):
    if name == "monte carlo":
        return partial(gauss_MC, **kwargs)
    raise ValueError(f"Not implemented yet: {name}")

@dataclass
class Example:
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

    def is_gauss_model(self) -> bool:
        return False

    def get_algorithms(self) -> list[str]:
        ret = ["svgd", "msip_f", "msip_gi", "msip_gf", "cbs", "grad_aldi", "deepens"]
        if self.is_gauss_model():
            ret.extend(["eks", "gradfree_aldi"])
        return ret

@dataclass
class GaussExample(Example):
    gauss_model: GaussianModel
    eks: EKS
    gradfree_aldi: GradFreeALDI

    def is_gauss_model(self) -> bool:
        return False

    def init_particles(self, D: Defaults, device=None, dtype = None):
        inv_std = torch.as_tensor(self.gauss_model.prior_precision, device=device, dtype=dtype).sqrt()
        return (
            torch.randn((D.n_particles, self.dim), device=device, dtype=dtype, generator=self.rng) / inv_std
            + self.gauss_model.prior_mean
        )