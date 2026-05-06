import os
from typing import Optional
from urllib.request import urlretrieve

import torch

from nak_torch.algorithms.msip import (
    MSIPFredholm,
)

from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchLogDensityEvaluator,
    DeviceLike,
    LogisticRegressionModel,
)
from functions.defaults import (
    Defaults,
    Example,
    run_example,
)

import scipy.io
import numpy as np

from nak_torch.tools.util import infinite_iter

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

# %%
DATA_URL = "https://raw.githubusercontent.com/DartML/Stein-Variational-Gradient-Descent/refs/heads/master/data/covertype.mat"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "covertype.npy")


def download_file(data_url: str = DATA_URL, data_path: str = DATA_PATH):
    urlretrieve(data_url, data_path)
    data_mat = scipy.io.loadmat(data_path)
    data_arr = data_mat["covtype"]
    # Flip first col to be (0,1) instead of (2,1) (where 2 is false)
    covariates = data_arr[:, 1:]
    labels = -1 * (data_arr[:, 0] - 2)
    data_arr = np.column_stack((covariates, labels))
    # Save
    np.save(data_path, data_arr)


# def accuracy(coeffs):
#     data, labels = regression_model.test_data, regression_model.test_labels
#     prob = torch.sigmoid(coeffs[:-1] @ data.T)
#     pred_labels = prob > 0.5
#     return torch.mean((pred_labels == labels).to(torch.float64))


def init_sample(
    N_samples: int,
    model: LogisticRegressionModel,
    rng: torch.Generator,
    device: Optional[DeviceLike] = None,
    dtype: Optional[torch.dtype] = None,
):
    prev_default_state = torch.default_generator.get_state()
    torch.default_generator.set_state(rng.get_state())
    alpha_init = model.hyperprior.sample((N_samples, 1))
    log_alpha_init = alpha_init.log()
    rng.set_state(torch.default_generator.get_state())
    torch.default_generator.set_state(prev_default_state)
    coeff_init = (
        torch.randn(
            (N_samples, model.dim - 1), dtype=dtype, device=device, generator=rng
        )
        / alpha_init.sqrt()
    )
    return torch.column_stack((coeff_init, log_alpha_init))


class CovtypeLogreg(Example):
    lr: float = 1e-3
    regression_model: LogisticRegressionModel

    def is_gauss_model(self) -> bool:
        return True

    def name(self):
        return "blr_covtype"

    def init_particles(self, D: Defaults, device=None, dtype=None) -> torch.Tensor:
        return init_sample(D.n_particles, self.regression_model, self.rng)

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        *_,
        batch_size: int = 50,
        use_compiled: bool = True,
        device=None,
        dtype=None,
        use_test: bool = False,
        **kwargs
    ):

        if not os.path.isfile(DATA_PATH):
            download_file()
        regression_model = LogisticRegressionModel(
            DATA_PATH, None, hyperprior_b=0.01, train_proportion=0.8, reduction="mean"
        )
        if use_test:
            regression_model.train_data, regression_model.train_labels = regression_model.test_data, regression_model.test_labels
        self.regression_model = regression_model
        self.dim = regression_model.dim
        self.rng = rng
        log_p = regression_model.to_log_dens(use_compiled=use_compiled)

        self.get_target_args_factory = lambda: infinite_iter(
            regression_model.get_data_loader(False, batch_size)
        )
        self.log_dens_evaluator = BatchLogDensityEvaluator(log_p, True)
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(log_p, False, True)
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, log_p, is_batched=True, is_grad_val=False
        )
        super().__init__(D, device, dtype)


if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 20
    SEED = 23802123
    N_STEPS = 1000
    DTYPE = torch.float64

    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ex = CovtypeLogreg(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
