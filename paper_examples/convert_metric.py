from datetime import datetime
import os
import pickle
from typing import Any, Callable

from torch import Tensor
import torch
from tqdm import tqdm
from functions.defaults import Example, Defaults
from nak_torch import metrics, tools
import aristoff_bangerth
import covid
import covtype
import schools
import two_moons


def get_most_recent_file(path: str, prefix: str) -> str:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(prefix)]
    return max(files, key=os.path.getmtime)


def last_configs(fname: str):
    last_data = {}
    with open(fname, "rb") as f:
        data = pickle.load(f)
        for alg, dat in data.items():
            if len(dat) <= 0:
                continue
            if isinstance(dat, tuple):
                dat = tuple(torch.as_tensor(d[:,-1]) for d in dat)
            else:
                dat = torch.as_tensor(dat[:,-1])
            last_data[alg] = dat
    return last_data


def create_ksd(
    configs: dict[str, Tensor | tuple[Tensor, ...]],
    ex: Example,
    kernel_elem: tools.types.KernelFunction,
    kernel_lengthscale: float,
) -> dict[str, list[float]]:
    ksd = metrics.KernelSteinDiscrepancy(
        ex.grad_log_dens_evaluator.grad_log_density,
        kernel_lengthscale,
        kernel_elem=kernel_elem,
    )
    ksd_evals: dict[str, list[float]] = {}
    for (k,v) in tqdm(configs.items()):
        if isinstance(v, tuple):
            if len(v) != 2:
                raise ValueError("Only supports values of pts or (pts, wts)!")
            ksd_evals[k + "_wt"] = [ksd(p,w).item() for (p, w) in zip(*v)]
            ksd_evals[k + "_unwt"] = [ksd(p).item() for (p, _) in zip(*v)]
        else:
            ksd_evals[k] = [ksd(p).item() for p in v]
    return ksd_evals

def save_metric(metric_dir: str, metric: str, example: str, metric_values: dict[Any, dict[str, list[float]]]):
    save_dict = {"metric": metric, "example": example, "values": metric_values}
    fname = os.path.join(metric_dir, "_".join([metric, example])) + f"_{datetime.now()}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(save_dict, f)

ExConstructor = Callable[[torch.Generator, Defaults], Example]

EXAMPLES: list[tuple[str, ExConstructor]] = [
    # ("abpde", aristoff_bangerth.AristoffBangerth),
    # ("covid", covid.COVID),
    ("schools", schools.Schools),
    ("two_moons", two_moons.TwoMoons),
    ("blr_covtype", covtype.CovtypeLogreg)
]

LENGTHSCALES = [1e-2, 5e-2, 1e-1, 5e-1, 1e-0]

KERNELS = [
    ("sqexp", tools.kernel.sqexp_kernel_elem),
    ("imq", tools.kernel.inverse_multi_quadric_kernel_elem),
]


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    metric_dir = os.path.join(data_dir, "metrics")
    rng = torch.default_generator
    D = Defaults()
    j, max_j = 0, len(EXAMPLES)*len(KERNELS)*len(LENGTHSCALES)
    for ex_name, ex_constr in EXAMPLES:
        ex = ex_constr(rng, D)
        metric_dict: dict[Any, dict[str, list[float]]] = {}
        config_file = get_most_recent_file(os.path.join(data_dir, "result"), ex_name)
        configs = last_configs(config_file)
        for kernel_name, kernel_elem in KERNELS:
            for lengthscale in LENGTHSCALES:
                print("{}/{}\t{}  \t{}  \t{:.2e}".format(j, max_j, ex_name, kernel_name, lengthscale))
                metric_evals = create_ksd(configs, ex, kernel_elem, lengthscale)
                metric_dict[(kernel_name, lengthscale)] = metric_evals
                j += 1
        save_metric(metric_dir, "ksd", ex_name, metric_dict)