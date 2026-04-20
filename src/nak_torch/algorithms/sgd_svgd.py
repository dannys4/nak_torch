#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
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




def create_svgd_step(
    kernel_elem: KernelFunction, grad_log_p: BatchGradLogDensity, *kernel_elem_args
) -> Callable[[BatchPtType], BatchPtType]:
    which_argnum = 1
    kernel_grad_val = kernel_grad_and_value_factory(
        kernel_elem, which_argnum, *kernel_elem_args
    )

    def svgd_step_dir(points: BatchPtType):
        # ASSUME SYMMETRY OF KERNEL
        # kg[i,j,ell] = grad(x_j[ell]) k(x_i, x_j), k[i,j] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val(points, points)
        # lpg[j,ell] = grad(x_j[ell]) log_p(x_j)
        log_p_grad_ev = grad_log_p(points)
        # term_1[i, ell] = sum_j k(i, j) grad(x_j[ell]) log_p(x_j)
        term_1 = k_eval @ log_p_grad_ev
        # term_2[i, ell] = sum_j grad(x_j[ell]) k(x_i, x_j)
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return svgd_step_dir


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


def svgd(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    verbose: bool = False,
    **unused_kwargs,
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    

    for idx in tqdm(range(n_steps), disable=not verbose):
        
        grad_log_p = batched_grad_log_density_factory(
            log_density, is_log_density_batched, grad_log_density
        )
        step_fcn = create_svgd_step(kernel_elem, grad_log_p, kernel_length_scale) 
        
        particles_diff = step_fcn(particles)
        with torch.no_grad():
            particles = particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx + 1].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)


def make_log_likelihood_batch_factory(
    model_class='bnn',
    hidden_dim=10,
    n_layers=1,
    beta=1.0,
):
    def make_log_likelihood_batch(X_batch, Y_batch, N_total):
        d = X_batch.shape[1]

        if model_class == 'bnn':
            model = bnn(d_in=d, hidden_dim=hidden_dim, n_layers=n_layers).double()
        else:
            raise ValueError(f"Unknown model_class: {model_class!r}")

        param_info = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]
        buffer_dict = OrderedDict(model.named_buffers())
        batch_size = X_batch.shape[0]

        def theta_to_param_dict(theta_1d):
            out, i = OrderedDict(), 0
            for name, shape, numel in param_info:
                out[name] = theta_1d[i:i+numel].view(shape)
                i += numel
            return out

        def log_likelihood_single_theta(theta_1d):
            param_dict = theta_to_param_dict(theta_1d)
            pred = functional_call(model, (param_dict, buffer_dict), (X_batch,))
            pred = pred.squeeze()

            # unbiased full-data scaling
            return -(N_total / batch_size) * soft_margin_loss(pred, Y_batch) / beta

        def log_likelihood(theta):
            if theta.ndim == 1:
                return log_likelihood_single_theta(theta)
            elif theta.ndim == 2:
                return torch.stack([
                    log_likelihood_single_theta(theta[i]) for i in range(theta.shape[0])
                ])
            else:
                raise ValueError(f"theta must be 1D or 2D, got {theta.shape}")

        return log_likelihood

    return make_log_likelihood_batch


def sgd_svgd(
    dataset,
    log_prior,
    log_likelihood,
    n_particles: int,
    n_epochs: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    verbose: bool = False,
    **unused_kwargs,
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    if keep_all:
        trajectories = torch.empty(
            (n_epochs + 1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())


    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    #optimizer = SGD(model.parameters(), lr=lr)




    


    t = 0
    for epoch in tqdm(range(n_epochs), disable=not verbose):
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device=device, dtype=particles.dtype)
            labels = labels.to(device=device, dtype=particles.dtype)
            
            # prior gradient
            grad_log_prior = batched_grad_log_density_factory(
                log_prior,
                is_log_density_batched,
                grad_log_density = None
            )
            
            # likelihood gradient from a batch
            log_likelihood_batch = make_log_likelihood_batch(inputs, labels, N_total)
            #log_likelihood_batch = make_log_likelihood_batch(inputs, labels, N_total)
            
            grad_log_likelihood = batched_grad_log_density_factory(
                log_likelihood,
                is_log_density_batched,
                grad_log_density = None
                )
            
            step_fcn = create_sgd_svgd_step(kernel_elem,
                                            grad_log_prior,
                                            grad_log_likelihood,
                                            kernel_length_scale)

            particles_diff = step_fcn(particles)
            
            
            with torch.no_grad():
                particles = particles + lr * particles_diff
                if bounds is not None:
                    particles.clamp_(bounds[0], bounds[1])
            t+=1
            if keep_all:
                trajectories[t].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
