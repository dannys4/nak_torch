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

from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchLogDensityEvaluator,
)
from functions.defaults import (
    Defaults,
    GaussExample,
    get_quadrule,
    run_example,
)

DIMENSION = 64


class AristoffBangerth(GaussExample):
    lr: float = 1e-3

    def is_gauss_model(self) -> bool:
        return True

    def name(self):
        return "aristoff_bangerth"

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        use_compiled: bool = True,
        device=None,
        dtype=None,
        **_
    ):
        self.dim = DIMENSION
        self.rng = rng
        self.gauss_model = build_aristoff_bangerth(
            use_compiled=use_compiled, dtype=dtype, device=device
        )
        log_p = self.gauss_model.to_log_dens(use_compiled=use_compiled)
        self.log_dens_evaluator = BatchLogDensityEvaluator(log_p, True)
        self.grad_log_dens_evaluator = BatchGradLogDensityEvaluator(log_p, False, True)
        self.target_msip_f = MSIPFredholm(
            D.gradient_decay, log_p, is_batched=True, is_grad_val=False
        )
        super().__init__(D, device, dtype)


if __name__ == "__main__":
    # EXAMPLE SPECIFIC DEFAULTS
    N_EXPERIMENTS = 20
    SEED = 758313
    N_STEPS = 1000
    DEVICE = torch.device('cuda:0')
    DTYPE = torch.float64
    
    torch.set_default_device(DEVICE)
    D = Defaults(n_steps=N_STEPS)
    rng = torch.Generator(DEVICE)
    rng.manual_seed(SEED)
    ex = AristoffBangerth(rng, D, dtype=DTYPE)

    run_example(N_EXPERIMENTS, D, rng, ex, ex.lr, __file__, dtype=DTYPE)
