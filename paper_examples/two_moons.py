# %%
from typing import Optional

import torch
import nak_torch
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm
from functions import bnn
from nak_torch.tools.types import BatchGradLogDensityEvaluator, BatchLogDensityEvaluator, DeviceLike
from nak_torch.tools.util import infinite_iter
from functions.defaults import Defaults, Example, run_example


def create_two_moons_bnn_model(
    rng: torch.Generator,
    regularization: float,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLike] = None,
    prop_train: float = 0.8,
    n_layers: int = 1,
    hidden_dim: int = 50,
    beta: float = 1.0,
    use_test: bool = False,
) -> bnn.BNNClassifierPosterior:
    DATASET = "two_bananas"
    data_train, labels_train, data_test, labels_test = bnn.load_dataset(
        DATASET, prop_train, rng, dtype, device
    )
    if use_test:
        data, labels = data_test, labels_test
    else:
        data, labels = data_train, labels_train

    return bnn.BNNClassifierPosterior(
        data=data,
        labels=labels,
        beta=beta,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        weight_decay=regularization**2,
    )

def initialize_particles(n_particles: int, dim: int, regularization: float):
    return torch.randn(n_particles, dim).double() / regularization

class TwoMoons(Example):
    lr: float = 1e-3
    regularization: float = 0.2
    batch_size: int = 64

    def name(self):
        return "two_moons"

    def __init__(
        self, rng: torch.Generator, D: Defaults, device=None, dtype=None, **kwargs
    ):
        posterior = create_two_moons_bnn_model(rng, self.regularization, dtype=dtype, device=device, **kwargs)
        data_loader = posterior.get_data_loader(self.batch_size, shuffle=True, generator=rng)
        self.rng = rng
        self.dim = posterior.dimension
        self.log_dens_evaluator = BatchLogDensityEvaluator(
            posterior, True
        )
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(
            posterior, False, True
        )
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, posterior, is_batched=True, is_grad_val=False
        )
        self.get_target_args_factory = lambda: infinite_iter(data_loader)
        super().__init__(D, device, dtype)


    def init_particles(self, D, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(
            (D.n_particles, self.dim), generator=self.rng, device=device, dtype=dtype
        ) / self.regularization


if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 20
    SEED = 3120312
    N_STEPS = 1000
    DTYPE = torch.float64

    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ex = TwoMoons(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
