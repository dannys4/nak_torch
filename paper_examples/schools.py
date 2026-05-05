import posteriordb
import stan
import torch

from nak_torch.algorithms.msip import MSIPFredholm

from nak_torch.tools import stan_tools
from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchLogDensityEvaluator,
)
from functions.defaults import (
    Defaults,
    Example,
    run_example,
)


DIMENSION = 10


def create_schools_stan_model() -> stan_tools.StanModel:
    pdb = posteriordb.PosteriorDatabase()
    which_posterior = "eight_schools-eight_schools_centered"
    posterior = pdb.posterior(which_posterior)
    post_model = stan.build(posterior.model.stan_code(), data=posterior.data.values())
    dim = sum(posterior.information["dimensions"].values())
    return stan_tools.StanModel(post_model, dim)


class Schools(Example):
    lr: float = 1e-3

    def name(self):
        return "schools"

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        device=None,
        dtype=None,
    ):
        stan_model = create_schools_stan_model()
        assert stan_model.dim == DIMENSION
        self.dim = DIMENSION
        self.rng = rng
        self.log_dens_evaluator = BatchLogDensityEvaluator(
            stan_model.log_dens_batch, True
        )
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(
            stan_model.grad_log_dens_batch, True, True
        )
        grad_val = stan_model.grad_val_log_dens_batch
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, grad_val, is_batched=True, is_grad_val=True
        )
        super().__init__(D, device, dtype)

    def init_particles(self, D, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(
            (D.n_particles, self.dim), generator=self.rng, device=device, dtype=dtype
        )


if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 20
    SEED = 231239842
    N_STEPS = 1000
    DTYPE = torch.float64

    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ex = Schools(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
