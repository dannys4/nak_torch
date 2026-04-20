#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of batch sgd svgd
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from typing import Optional, Callable
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem, kernel_grad_and_value_factory
from nak_torch.tools.types import KernelFunction, BatchGradLogDensity, BatchPtType
from nak_torch.tools.util import batched_grad_log_density_factory, initialize_particles




def create_sgd_svgd_step(
    kernel_elem: KernelFunction,
    grad_log_prior: BatchGradLogDensity,
    grad_log_likelihood: BatchGradLogDensity,
    *kernel_elem_args
) -> Callable[[BatchPtType], BatchPtType]:
    which_argnum = 1
    kernel_grad_val = kernel_grad_and_value_factory(
        kernel_elem, which_argnum, *kernel_elem_args
    )

    def sgd_svgd_step_dir(points: BatchPtType):
        # ASSUME SYMMETRY OF KERNEL
        # kg[i,j,ell] = grad(x_j[ell]) k(x_i, x_j), k[i,j] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val(points, points)
        # lpg[j,ell] = grad(x_j[ell]) log_p(x_j)
        log_prior_grad_ev = grad_log_prior(points)
        log_likelihood_grad_ev = grad_log_likelihood(points)
        
        log_p_grad_ev = log_prior_grad_ev + log_likelihood_grad_ev
        # term_1[i, ell] = sum_j k(i, j) grad(x_j[ell]) log_p(x_j)
        term_1 = k_eval @ log_p_grad_ev
        # term_2[i, ell] = sum_j grad(x_j[ell]) k(x_i, x_j)
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return sgd_svgd_step_dir



def sgd_svgd(
    dataset,
    log_prior,
    make_log_likelihood_batch,
    n_particles: int,
    n_epochs: int,
    dim: int,
    lr: float,
    batch_size: int = 64,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_prior_batched: bool = False,
    is_log_likelihood_batched: bool = False,
    grad_log_prior: Optional[BatchGradLogDensity] = None,
    grad_log_likelihood: Optional[BatchGradLogDensity] = None,
    dataloader_kwargs: Optional[dict] = None,
    verbose: bool = False,
    **unused_kwargs,
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn(f"Unused kwargs:\n{unused_kwargs}")

    if seed is not None:
        torch.manual_seed(seed)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)
    N_total = len(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    n_updates = n_epochs * len(train_loader)

    if keep_all:
        trajectories = torch.empty(
            (n_updates + 1, *particles.shape),
            device=device,
            dtype=particles.dtype,
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    # gradient of log prior
    batch_grad_log_prior = batched_grad_log_density_factory(
        log_prior,
        is_log_prior_batched,
        grad_log_prior,
    )

    t = 0
    for epoch in tqdm(range(n_epochs), disable=not verbose):
        for batch in train_loader:
            #batch = batch.to(device=device, dtype=particles.dtype)
            batch = tuple(x.to(device=device, dtype=particles.dtype) for x in batch)
            
            # build current minibatch objective
            log_likelihood_batch = make_log_likelihood_batch(batch, N_total)

            batch_grad_log_likelihood = batched_grad_log_density_factory(
                log_likelihood_batch,
                is_log_likelihood_batched,
                grad_log_likelihood,
            )

            step_fcn = create_sgd_svgd_step(
                kernel_elem,
                batch_grad_log_prior,
                batch_grad_log_likelihood,
                kernel_length_scale,
            )

            particles_diff = step_fcn(particles)

            with torch.no_grad():
                particles = particles + lr * particles_diff
                if bounds is not None:
                    particles.clamp_(bounds[0], bounds[1])

            t += 1
            if keep_all:
                trajectories[t].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze(0)