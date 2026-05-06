from typing import Iterable

import torch
from torch import Tensor
import torchvision
from torchvision import transforms
import os

from functions import bnn
from nak_torch.algorithms.msip.estimators import MSIPFredholm
from nak_torch.tools.types import BatchGradLogDensityEvaluator, BatchLogDensityEvaluator
from functions.defaults import Defaults, run_example, Example

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "datasets")


def supervised_to_infinite_iter(
    iterable: Iterable[tuple[Tensor, Tensor]], device=None, dtype=None
):
    while True:
        for data, labels in iter(iterable):
            yield (
                data.to(device=device, dtype=dtype),
                labels.to(device=device, dtype=torch.int32),
            )


def create_mnist_bnn_model(
    regularization: float,
    n_layers: int = 2,
    hidden_dims: list[int] | int = [128, 64],
    beta: float = 1.0,
    use_test: bool = False,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Normalize with MNIST mean and std
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        DATA_PATH, transform=transform, train=not use_test, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        DATA_PATH, transform=transform, train=False, download=True
    )
    return bnn.BNNMultiClassifierPosterior(
        dataset,
        n_labels=10,
        hidden_dims=hidden_dims,
        n_layers=n_layers,
        beta=beta,
        weight_decay=regularization,
        data_dim=28**2,
        N_data_total=60000,
    ), test_dataset


def initialize_particles(n_particles: int, dim: int, regularization: float):
    return torch.randn(n_particles, dim).double() / regularization


class MNIST(Example):
    lr: float = 1e-3
    regularization: float = 0.2
    batch_size: int = 64

    def name(self):
        return "two_moons"

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        device=None,
        dtype=None,
        use_test: bool = False,
        **kwargs,
    ):
        posterior, test_dataset = create_mnist_bnn_model(
            self.regularization, use_test=use_test, **kwargs
        )
        data_loader = posterior.get_data_loader(
            self.batch_size, shuffle=True, generator=rng
        )
        self.rng = rng
        self.dim = posterior.dimension
        self.test_dataset = test_dataset
        self.log_dens_evaluator = BatchLogDensityEvaluator(posterior, True)
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(
            posterior, False, True
        )
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, posterior, is_batched=True, is_grad_val=False
        )
        self.get_target_args_factory = lambda: supervised_to_infinite_iter(
            data_loader, device, dtype
        )
        super().__init__(D, device, dtype)

    def init_particles(self, D, device=None, dtype=None) -> torch.Tensor:
        return (
            torch.randn(
                (D.n_particles, self.dim),
                generator=self.rng,
                device=device,
                dtype=dtype,
            )
            / self.regularization
        )

    def get_test_target_args(self, device=None, dtype=None):
        all_dataset = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset)
        )
        data, labels = next(iter(all_dataset))
        data = data.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.int32)
        return data, labels


if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 1
    SEED = 3120312
    N_STEPS = 10
    DTYPE = torch.float64

    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator()
    rng.manual_seed(SEED)
    ex = MNIST(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
