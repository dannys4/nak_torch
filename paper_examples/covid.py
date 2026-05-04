from typing import Optional

import posteriordb
import stan
import torch
from functions import build_aristoff_bangerth
from nak_torch.algorithms import (
    MSIP,
    SVGD,
    CBS,
    EKS,
    GradALDI,
    GradFreeALDI,
    DeepEnsembles,
)

from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)

from nak_torch.tools import stan_tools
from nak_torch.tools.pyro_tools import DeviceLike
from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchLogDensityEvaluator,
)
from functions.defaults import (
    Defaults,
    Example,
    get_quadrule,
    run_example,
)


# def covid_prior_sample(
#     N_samples: int,
#     M_y: int = 14,
#     dtype: Optional[torch.dtype] = None,
#     device: Optional[DeviceLike] = None,
#     rng: Optional[torch.Generator] = None,
# ):
#     if rng is None:
#         rng = torch.default_generator
#     if dtype is None:
#         dtype = torch.get_default_dtype()
#     if device is None:
#         device = torch.get_default_device()
#     tau = (
#         torch.empty((N_samples, 1), dtype=dtype, device=device)
#         .exponential_(generator=rng)
#         .div_(0.03)
#     )
#     y = (
#         torch.empty((N_samples, M_y), dtype=dtype, device=device)
#         .exponential_(generator=rng)
#         .div_(tau)
#     )
#     phi = torch.randn((N_samples, 1), generator=rng).mul_(5.0)
#     kappa = torch.randn((N_samples, 1), generator=rng).mul_(0.5)
#     mu = torch.randn((N_samples, M_y), generator=rng).mul_(kappa).add_(3.28)
#     alpha_hier = torch._standard_gamma(
#         torch.as_tensor(0.1667, dtype=dtype, device=device).expand(N_samples, 6),
#         generator=rng,
#     )
#     ifr_noise = torch.randn((N_samples, M_y), generator=rng).mul_(0.1).add_(1.0)
#     log_tau = tau.log_()
#     log_alpha_hier = alpha_hier.log_()
#     log_y = y.log_()
#     return torch.column_stack((mu, log_alpha_hier, kappa, log_y, phi, log_tau, ifr_noise))


DIMENSION = 51

def create_covid_stan_model() -> stan_tools.StanModel:
    pdb = posteriordb.PosteriorDatabase()
    which_posterior = "ecdc0501-covid19imperial_v3"
    posterior = pdb.posterior(which_posterior)
    post_model = stan.build(posterior.model.stan_code(), data=posterior.data.values())
    dim = sum(posterior.information["dimensions"].values())
    return stan_tools.StanModel(post_model, dim)


class COVID(Example):
    lr: float = 1e-3

    def name(self):
        return "covid"

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        device=None,
        dtype=None,
    ):
        stan_model = create_covid_stan_model()
        assert stan_model.dim == DIMENSION
        self.dim = DIMENSION
        self.rng = rng
        self.log_dens_evaluator = BatchLogDensityEvaluator(stan_model.log_dens_batch, True)
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(stan_model.grad_log_dens_batch, True, True)
        grad_val = stan_model.grad_val_log_dens_batch
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, grad_val, is_batched=True, is_grad_val=True
        )
        self.target_msip_gi = MSIPQuadGradientInformed(
            grad_val,
            get_quadrule(
                D.quadrule,
                d=DIMENSION,
                **D.msip_gi_quadrule_kwargs,
                device=device,
                dtype=dtype,
            ),
            D.gradient_decay,
        )
        self.target_msip_gf = MSIPQuadGradientFree(
            stan_model.log_dens_batch,
            get_quadrule(
                D.quadrule,
                d=DIMENSION,
                **D.msip_gf_quadrule_kwargs,
                device=device,
                dtype=dtype,
            ),
        )
        all_kwargs = dict(
            dim=DIMENSION, n_particles=D.n_particles, device=device, dtype=dtype
        )
        self.svgd = SVGD(**all_kwargs, **D.svgd_kwargs)
        self.msip = MSIP(**all_kwargs, **D.msip_kwargs)
        self.cbs = CBS(**all_kwargs, rng=rng, **D.cbs_kwargs)
        self.deepens = DeepEnsembles(**all_kwargs)
        self.eks = EKS(**all_kwargs, rng_or_seed=rng)
        self.grad_aldi = GradALDI(**all_kwargs, rng=rng)
        self.gradfree_aldi = GradFreeALDI(**all_kwargs, rng_or_seed=rng)
    def init_particles(self, D, device=None, dtype=None) -> torch.Tensor:
        return torch.randn((D.n_particles, DIMENSION), generator=self.rng, device=device, dtype=dtype)

if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 1
    SEED = 23802123
    N_STEPS = 10
    DTYPE = torch.float64

    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ex = COVID(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
