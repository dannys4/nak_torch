import datetime
import os
import pickle
from typing import Any
import warnings
import torch
from tqdm import tqdm
from functions import build_aristoff_bangerth
import nak_torch
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
from functions.defaults import Defaults, GaussExample, get_quadrule

DIMENSION = 64


class AristoffBangerth(GaussExample):
    lr: float = 1e-3

    def is_gauss_model(self) -> bool:
        return True

    def __init__(
        self,
        rng: torch.Generator,
        D: Defaults,
        use_compiled: bool = True,
        device=None,
        dtype=None,
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
        grad_val = self.target_msip_f.log_dens_grad_val
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
            log_p,
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


if __name__ == "__main__":
    N_EXPERIMENTS = 1
    SEED = 23802123
    D = Defaults()
    D.n_steps = 10
    rng = torch.Generator()
    rng.manual_seed(SEED)
    dtype = torch.float64
    ex = AristoffBangerth(rng, D, dtype=dtype)
    LR = ex.lr
    metric_evals = {alg: [] for alg in ex.get_algorithms()}
    outputs = {alg: [] for alg in ex.get_algorithms()}
    n_steps, bounds = D.n_steps, D.bounds

    metric = D.metric(ex.log_dens_evaluator)

    def run_nak(target, alg, init_particles, alg_name, metric_evals, outputs):
        res: Any
        try:
            res = nak_torch.nak(
                target, alg, n_steps, LR, rng, init_particles, bounds, dtype=dtype
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
                    for pw in tqdm(iterator, desc=alg_name, total=n_steps + 1)
                ]
            )
        except:  # noqa: E722
            warnings.warn(f"Problem assessing algorithm {alg_name}")
        return metric_evals, outputs

    all_init_particles = []
    for r in range(N_EXPERIMENTS):
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
            ex.grad_log_dens_evaluator, ex.svgd, init_particles, "svgd", metric_evals, outputs
        )

        # DeepEns
        pts_cbs = run_nak(
            ex.grad_log_dens_evaluator, ex.deepens, init_particles, "deepens", metric_evals, outputs
        )

        # aldi
        metric_evals, outputs = run_nak(
            ex.grad_log_dens_evaluator, ex.grad_aldi, init_particles, "grad_aldi", metric_evals, outputs
        )
        metric_evals, outputs = run_nak(
            ex.gauss_model, ex.gradfree_aldi, init_particles, "grad_aldi", metric_evals, outputs
        )

        # CBS
        pts_cbs = run_nak(
            ex.log_dens_evaluator, ex.cbs, init_particles, "cbs", metric_evals, outputs
        )

        # EKS
        pts_eks = run_nak(
            ex.gauss_model, ex.eks, init_particles, "eks", metric_evals, outputs
        )

    def output_stack(t):
        if isinstance(t[0], tuple):
            points, weights = [torch.stack([pw[j] for pw in t]) for j in range(2)]
            return points.numpy(), weights.numpy()
        return torch.stack(t).numpy()

    outputs = {k: output_stack(v) for (k, v) in outputs.items() if len(v) > 0}

    all_experiment_data = dict(
        lr=LR,
        init_particles=all_init_particles,
        metrics=metric_evals,
        hyperparams=D.__dict__,
    )
    fname = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_") + ".pkl"
    output_path = os.path.join(os.path.dirname(__file__), "data", "result", fname)

    experiment_path = os.path.join(
        os.path.dirname(__file__), "data", "experiment", fname
    )

    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)
    with open(experiment_path, "wb") as f:
        pickle.dump(all_experiment_data, f)
