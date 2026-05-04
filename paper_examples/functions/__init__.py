# Various functions for illustrations
# twodims: 2D functions
# nns: neural networks
# bayes: posterior distributions for 'realistic' problems
# kde: kernel density estimator, useful to go from discrete to continuous

# Ayoub Belhadji
# 05/12/2025


from .mixture_of_gaussians import mixture_of_gaussians
from .himmelblau import himmelblau
from . import banana, kde
from .aristoff_bangerth import build_aristoff_bangerth

__all__ = [
    "mixture_of_gaussians",
    "banana",
    "himmelblau",
    "kde",
    "build_aristoff_bangerth"
]
