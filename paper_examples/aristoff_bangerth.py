# %%
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
            get_quadrule(D.quadrule, d=DIMENSION, **D.msip_gi_quadrule_kwargs, device=device, dtype=dtype),
            D.gradient_decay,
        )
        self.target_msip_gf = MSIPQuadGradientFree(
            log_p,
            get_quadrule(D.quadrule, d=DIMENSION, **D.msip_gf_quadrule_kwargs, device=device, dtype=dtype),
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
    print(ex.get_algorithms())
    n_steps, bounds = D.n_steps, D.bounds

    def run_nak(target, alg, init_particles):
        return nak_torch.nak(target, alg, n_steps, LR, rng, init_particles, bounds, dtype=dtype)

    metric = D.metric(ex.log_dens_evaluator)
    all_init_particles = []
    for r in range(N_EXPERIMENTS):
        init_particles = ex.init_particles(D, dtype=dtype)
        all_init_particles.append(init_particles.numpy())

        # msip
        pts_msip_f, wts_msip_f = run_nak(ex.target_msip_f, ex.msip, init_particles)
        metric_evals["msip_f"].append([metric(p,w) for (p,w) in tqdm(zip(pts_msip_f, wts_msip_f), desc="msip_f", total=len(pts_msip_f))])
        outputs["msip_f"].append((pts_msip_f, wts_msip_f))

        pts_msip_gi, wts_msip_gi = run_nak(ex.target_msip_gi, ex.msip, init_particles)
        metric_evals["msip_gi"].append([metric(p,w) for (p,w) in tqdm(zip(pts_msip_gi, wts_msip_gi), desc="msip_gi", total=len(pts_msip_gi))])
        outputs["msip_gi"].append((pts_msip_gi, wts_msip_gi))

        pts_msip_gf, wts_msip_gf = run_nak(ex.target_msip_gf, ex.msip, init_particles)
        metric_evals["msip_gf"].append([metric(p,w) for (p,w) in tqdm(zip(pts_msip_gf, wts_msip_gf), desc="msip_gf", total=len(pts_msip_gf))])
        outputs["msip_gf"].append((pts_msip_gf, wts_msip_gf))

        # svgd
        pts_svgd = run_nak(ex.grad_log_dens_evaluator, ex.svgd, init_particles)
        metric_evals["svgd"].append([metric(p) for p in tqdm(pts_svgd, desc="SVGD")])
        outputs["svgd"].append(pts_svgd)

        # aldi
        pts_grad_aldi = run_nak(ex.grad_log_dens_evaluator, ex.grad_aldi, init_particles)
        pts_gradfree_aldi = run_nak(ex.gauss_model, ex.gradfree_aldi, init_particles)
        metric_evals["grad_aldi"].append([metric(p) for p in tqdm(pts_grad_aldi, desc="grad_aldi")])
        outputs["grad_aldi"].append(pts_grad_aldi)
        metric_evals["gradfree_aldi"].append([metric(p) for p in tqdm(pts_gradfree_aldi, desc="gradfree_aldi")])
        outputs["gradfree_aldi"].append(pts_gradfree_aldi)

        # CBS
        pts_cbs = run_nak(ex.log_dens_evaluator, ex.cbs, init_particles)
        metric_evals["cbs"].append([metric(p) for p in tqdm(pts_cbs, desc="CBS")])
        outputs["cbs"].append(pts_cbs)

        # EKS
        pts_eks = run_nak(ex.gauss_model, ex.eks, init_particles)
        metric_evals["eks"].append([metric(p) for p in tqdm(pts_eks, desc="EKS")])
        outputs["eks"].append(pts_eks)
    def output_stack(t):
        if isinstance(t[0], tuple):
            points, weights = [torch.stack([pw[j] for pw in t]) for j in range(2)]
            return points.numpy(), weights.numpy()
        return torch.stack(t).numpy()
    outputs = {k:output_stack(v) for (k,v) in outputs.items()}

    all_experiment_data = dict(
        lr= LR,
        init_particles=all_init_particles,
        metrics=metric_evals,
        hyperparams=D.__dict__
    )
