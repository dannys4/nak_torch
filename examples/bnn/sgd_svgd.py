import torch
from torch.utils.data import TensorDataset

from nak_torch.algorithms import svgd
from nak_torch.algorithms import sgd_svgd   # adapt import path if needed
from bnn_sgd_impl import *
from bnn_impl import *

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────────
    DATASET      = "two_bananas"
    MODEL_CLASS  = "bnn"
    HIDDEN_DIM   = 50
    N_LAYERS     = 1
    TRAIN_RATIO  = 0.8

    N_PARTICLES  = 50
    N_STEPS      = 500          # full-batch SVGD iterations
    N_EPOCHS     = 100           # SGD-SVGD epochs
    BATCH_SIZE   = 64

    BETA         = 1.0
    LAMBDA2      = 0.00005
    SIGMA        = 1.0

    LR_SVGD      = 0.1
    LR_SGD_SVGD  = 0.1

    DEVICE = None   # or torch.device("cpu") / torch.device("cuda")
    SEED   = 0

    torch.manual_seed(SEED)

    # ─────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────
    X_train, Y_train, X_test, Y_test = load_dataset(
        DATASET,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    train_dataset = TensorDataset(X_train, Y_train)

    # ─────────────────────────────────────────────────────────────
    # Full-batch objective for SVGD
    # ─────────────────────────────────────────────────────────────
    obj_svgd = make_objective(
        X_train, Y_train,
        model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        beta=BETA,
        lambda2=LAMBDA2,
    )

    # ─────────────────────────────────────────────────────────────
    # Split objective for SGD-SVGD
    # ─────────────────────────────────────────────────────────────
    log_prior= make_sgd_logprior(
        X_train, Y_train,
        model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        beta=BETA,
        lambda2=LAMBDA2,
    )

    make_log_likelihood_batch = log_likelihood_batch_factory(
        model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        beta=BETA,
    )

    dimension = obj_svgd.total_numel

    # ─────────────────────────────────────────────────────────────
    # Shared initialization
    # ─────────────────────────────────────────────────────────────
    init_particles = 5.0 * torch.randn(N_PARTICLES, dimension).double()

    # ─────────────────────────────────────────────────────────────
    # Run full-batch SVGD
    # ─────────────────────────────────────────────────────────────
    trajectories_svgd = svgd(
        log_density=obj_svgd,
        n_particles=N_PARTICLES,
        n_steps=N_STEPS,
        dim=dimension,
        lr=LR_SVGD,
        seed=None,
        device=DEVICE,
        init_particles=init_particles.clone(),
        kernel_length_scale=SIGMA,
        keep_all=True,
        is_log_density_batched=True,
        verbose=True,
    )

    # ─────────────────────────────────────────────────────────────
    # Run SGD-SVGD
    # ─────────────────────────────────────────────────────────────
    trajectories_sgd_svgd = sgd_svgd(
        dataset=train_dataset,
        log_prior=log_prior,
        make_log_likelihood_batch=make_log_likelihood_batch,
        n_particles=N_PARTICLES,
        n_epochs=N_EPOCHS,
        dim=dimension,
        lr=LR_SGD_SVGD,
        batch_size=BATCH_SIZE,
        seed=None,
        device=DEVICE,
        init_particles=init_particles.clone(),
        kernel_length_scale=SIGMA,
        keep_all=True,
        is_log_prior_batched=True,
        is_log_likelihood_batched=True,
        verbose=True,
    )

    # ─────────────────────────────────────────────────────────────
    # Collect for plots
    # ─────────────────────────────────────────────────────────────
    trajectories_dict = {
        "SVGD": trajectories_svgd,
        "SGD-SVGD": trajectories_sgd_svgd,
    }

    # For evaluation / plotting, use the full objective for both
    # since both methods target the same posterior.
    objective_fns_dict = {
        "SVGD": obj_svgd,
        "SGD-SVGD": obj_svgd,
    }

    # ─────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────
    eval_function_trajectories(obj_svgd, trajectories_svgd, "SVGD")
    eval_function_trajectories(obj_svgd, trajectories_sgd_svgd, "SGD-SVGD")

    # ─────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────
    plot_boundaries(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_mean_prediction(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_diversity_curve(trajectories_dict)

    # ─────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────
    evaluate(trajectories_dict, objective_fns_dict, X_train, Y_train, "train")
    evaluate(trajectories_dict, objective_fns_dict, X_test,  Y_test,  "test")