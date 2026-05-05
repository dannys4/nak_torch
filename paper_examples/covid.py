import posteriordb
import stan
import torch
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
        super().__init__(D, device, dtype)
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
