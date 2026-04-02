#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
from torch import Tensor
from jaxtyping import Float
from typing import Optional

# Monte Carlo on the hyperphere


def MC_on_hypersphere(batch_sizes: tuple[int, ...], N: int, d: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    r""" Create Monte Carlo samples on the unit hypersphere"""
    normal_pts = torch.randn((*batch_sizes, N, d), dtype=dtype, device=device)
    pts = normal_pts / torch.linalg.norm(normal_pts, axis=-1).unsqueeze_(-1)
    return pts, torch.ones((*batch_sizes, N), dtype=dtype, device=device) / N



def OMC_on_hypersphere(
    batch_sizes: tuple[int, ...],
    N: int,
    d: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    r"""
    Create Orthogonal Monte Carlo samples on the unit hypersphere.

    Instead of i.i.d. normal samples, we tile ceil(N/d) Haar-uniform orthogonal
    matrices (each giving d orthonormal rows), then truncate to exactly N points.

    Each orthogonal matrix is constructed via QR decomposition of a random
    Gaussian matrix, with the standard sign correction to ensure Haar measure.

    Args:
        batch_sizes : leading batch dimensions
        N           : number of sample points desired
        d           : ambient dimension (hypersphere lives in R^d)
        dtype       : torch dtype (default: torch.get_default_dtype())
        device      : torch device

    Returns:
        pts     : (*batch_sizes, N, d)  — points on the unit (d-1)-sphere
        weights : (*batch_sizes, N)     — uniform weights 1/N
    """
    num_blocks = math.ceil(N / d)          # number of d×d orthogonal matrices
    total = num_blocks * d                 # may be > N; will be sliced at the end

    # Sample num_blocks Haar-uniform orthogonal matrices 
    # Shape: (*batch_sizes, num_blocks, d, d)
    G = torch.randn((*batch_sizes, num_blocks, d, d), dtype=dtype, device=device)

    # QR decomposition — flatten batch dims so torch.linalg.qr is happy,
    # then restore shape.
    flat_G = G.reshape(-1, d, d)          # (B * num_blocks, d, d)
    Q, R = torch.linalg.qr(flat_G)       # Q: (B * num_blocks, d, d)

    # Sign correction: multiply each column of Q by sign(diag(R))
    # so that the distribution is exactly Haar (not just "almost").
    sign = R.diagonal(dim1=-2, dim2=-1).sign()   # (B * num_blocks, d)
    Q = Q * sign.unsqueeze(-2)                   # broadcast over rows

    Q = Q.reshape(*batch_sizes, num_blocks, d, d)

    # Flatten the (num_blocks, d) tile into one long list of rows 
    # (*batch_sizes, total, d)  where total = num_blocks * d
    pts_tiled = Q.reshape(*batch_sizes, total, d)

    # --- Truncate to exactly N points ---
    pts = pts_tiled[..., :N, :]           # (*batch_sizes, N, d)

    weights = torch.ones((*batch_sizes, N), dtype=dtype, device=device) / N
    return pts, weights




def gaussian_laguerre_quadrature(N: int, alpha: float, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    r""" Create generalized Gauss--Laguerre quadrature """
    # N is the number of nodes of the radial quadrature
    n_range = torch.arange(N-1, dtype=dtype, device=device)
    diag_0 = 2 * torch.arange(N, dtype=dtype, device=device) + 1 + alpha
    diag_m1 = torch.sqrt((n_range + 1) * (n_range + 1 + alpha))
    diag_p1 = torch.sqrt((n_range + 1) * (n_range + 1 + alpha))
    J_N = torch.diag(diag_0) + torch.diag(diag_m1, -1) + torch.diag(diag_p1, 1)
    e, v = torch.linalg.eigh(J_N)
    wts = v[0]**2
    pts = e
    return pts, wts


def combine_radial_spherical_quadrature(
        sph_pts: Float[Tensor, "N_s d"], sph_wts: Float[Tensor, " N_s"],
        r_pts: Float[Tensor, "N_r d"], r_wts: Float[Tensor, " N_r"]
) -> tuple[Float[Tensor, "N_s*N_r d"], Float[Tensor, " N_s*N_r"]]:
    r""" Take a spherical and radial quadrature and tensorize them appropriately """
    sph_pts_13 = sph_pts.reshape(sph_pts.shape[0], 1, sph_pts.shape[1])
    sph_wts_1 = sph_wts.reshape(-1, 1)
    r_pts_2 = torch.sqrt(2 * r_pts.reshape(1, -1, 1))
    r_wts_2 = r_wts.reshape(1, -1)
    pts_combine = sph_pts_13 * r_pts_2
    wts_combine = sph_wts_1 * r_wts_2
    return pts_combine.reshape(-1, sph_pts.shape[1]), wts_combine.reshape(-1)

vmap_combine_radial_spherical_quadrature = torch.vmap(
    combine_radial_spherical_quadrature, in_dims=(0, 0, None, None)
)

def spherical_MC_radial_Laguerre(
        batch_size: int, N_spherical: int, d: int, N_radial: int = 3,
        dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
) -> tuple[Float[Tensor, "batch N_s*N_r d"], Float[Tensor, "batch N_s*N_r"]]:
    r""" Perform MC sampling on hypersphere and appropriate Gauss--Laguerre quadrature over radius """
    alpha = d/2-1
    r_pts, r_wts = gaussian_laguerre_quadrature(N_radial, alpha, dtype, device)
    MC_pts, MC_wts = MC_on_hypersphere((batch_size,), N_spherical, d, dtype, device)

    pts, wts = vmap_combine_radial_spherical_quadrature(
        MC_pts, MC_wts, r_pts, r_wts
    )
    return pts, wts


def O_spherical_MC_radial_Laguerre(
        batch_size: int, N_spherical: int, d: int, N_radial: int = 3,
        dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
) -> tuple[Float[Tensor, "batch N_s*N_r d"], Float[Tensor, "batch N_s*N_r"]]:
    r""" Perform MC sampling on hypersphere and appropriate Gauss--Laguerre quadrature over radius """
    alpha = d/2-1
    r_pts, r_wts = gaussian_laguerre_quadrature(N_radial, alpha, dtype, device)
    MC_pts, MC_wts = OMC_on_hypersphere((batch_size,), N_spherical, d, dtype, device)

    pts, wts = vmap_combine_radial_spherical_quadrature(
        MC_pts, MC_wts, r_pts, r_wts
    )
    return pts, wts

def gauss_MC(
        batch_size: int, N_quad: int, d: int
) -> tuple[Float[Tensor, "batch N_quad d"], Float[Tensor, "batch N_quad"]]:
    pts = torch.randn((batch_size, N_quad, d))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts