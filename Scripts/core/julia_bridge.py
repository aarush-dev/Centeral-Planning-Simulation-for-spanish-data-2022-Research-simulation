import os
from pathlib import Path

# Performance Optimization: Load pre-compiled Sysimage if available
SCRIPTS_DIR = Path(__file__).parent
SYSIMAGE_PATH = SCRIPTS_DIR / "sysimage.so"
if SYSIMAGE_PATH.exists():
    os.environ["PYTHON_JULIACALL_SYSIMAGE"] = str(SYSIMAGE_PATH)

os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import numpy as np
import logging
import scipy.sparse as sp

logger = logging.getLogger(__name__)

from juliacall import Main as jl

# -- Load models ---------------------------------------------------------------
JULIA_CORE = Path(__file__).parent / "model_core.jl"


def _load_core():
    # Only include the file if the Module isn't already defined to avoid constant redefinition errors
    jl.seval(f"""
        if !isdefined(Main, :ModelCore)
            include("{str(JULIA_CORE)}")
        end
    """)
    core = jl.ModelCore
    jl.seval("using Random")
    return core


try:
    CORE = _load_core()
    logger.info("Julia Core Engine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Julia Core Engine: {e}")
    raise

# -- Helpers ------------------------------------------------------------------


def _to_dense(A):
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=np.float64, order="C")
    return np.asarray(A, dtype=np.float64, order="C")


# -- Public API ---------------------------------------------------------------


def solve_planner(
    alpha,
    A,
    B,
    l_tilde,
    dK,
    K,
    L_total,
    G_vec,
    sigma_v,
    C_prev,
    k=25,
    tol_p=1e-3,
    tol_d=1e-4,
    eta_K=0.15,
    eta_L=0.15,
    max_iter=2000,
):
    """
    Bridge to Julia's FISTA-based dual ascent planner solver.
    Optimizes target consumption (C*) and gross output (X*) under CRRA welfare.
    """
    a_jl = np.asarray(alpha, dtype=np.float64)
    A_jl = _to_dense(A)
    B_jl = _to_dense(B)
    lt_jl = np.asarray(l_tilde, dtype=np.float64)
    dk_jl = np.asarray(dK, dtype=np.float64)
    K_jl = np.asarray(K, dtype=np.float64)
    Gv_jl = np.asarray(G_vec, dtype=np.float64)
    sig_jl = np.asarray(sigma_v, dtype=np.float64)
    c0_jl = np.asarray(C_prev, dtype=np.float64)

    res = CORE.solve_planner(
        a_jl,
        A_jl,
        B_jl,
        lt_jl,
        dk_jl,
        K_jl,
        float(L_total),
        Gv_jl,
        sig_jl,
        c0_jl,
        k=int(k),
        tol_p=float(tol_p),
        tol_d=float(tol_d),
        eta_K=float(eta_K),
        eta_L=float(eta_L),
        max_iter=int(max_iter),
    )

    # Convert Julia NamedTuple to Python dict using attribute access
    return {
        "C_star": np.array(res.C_star),
        "X_star": np.array(res.X_star),
        "pi_star": np.array(res.pi_star),
        "success": bool(res.success),
        "lam_K": np.array(res.lambda_K),
        "lam_L": float(res.lambda_L),
        "iterations": int(res.iterations),
        "mvps": int(res.mvps),
    }


def fast_loop(
    P_base,
    C_plan,
    alpha_true,
    alpha_slow,
    rng,
    drift_rho,
    drift_sigma,
    noise_sigma,
    Y,
    sigma_v,
    K_v,
    A_bar,
    B,
    theta_drift=0.1,
    alpha_h=None,
    Y_h=None,
    w_h=None,
    mu_t=1.0,
    alpha_slow_h=None,
    price_tol=0.005,
    max_price_iter=25,
    k_sigma=1.0,
    neumann_k=20,
    rho_M=None,
):
    """
    Bridge to Julia's monthly market clearing loop.
    Handles preference drift, multi-household CRRA demand, and price tatonnement.
    """
    pb_jl = np.asarray(P_base, dtype=np.float64)
    cp_jl = np.asarray(C_plan, dtype=np.float64)
    at_jl = np.asarray(alpha_true, dtype=np.float64)
    as_jl = np.asarray(alpha_slow, dtype=np.float64)
    sig_jl = np.asarray(sigma_v, dtype=np.float64)
    kv_jl = np.asarray(K_v, dtype=np.float64)
    ab_jl = _to_dense(A_bar)
    b_jl = _to_dense(B)

    seed = int(rng.integers(0, 2**32))
    jl_rng = jl.Random.MersenneTwister(seed)

    # Build optional household kwargs
    hh_kwargs = {}
    if alpha_h is not None and Y_h is not None:
        hh_kwargs["alpha_h"] = np.asarray(alpha_h, dtype=np.float64)
        hh_kwargs["Y_h"] = np.asarray(Y_h, dtype=np.float64)
    if w_h is not None:
        hh_kwargs["w_h"] = np.asarray(w_h, dtype=np.float64)
    if alpha_slow_h is not None:
        hh_kwargs["alpha_slow_h"] = np.asarray(alpha_slow_h, dtype=np.float64)

    rm_jl = float(rho_M) if rho_M is not None else -1.0
    res = CORE.fast_loop(
        pb_jl,
        cp_jl,
        at_jl,
        as_jl,
        jl_rng,
        float(drift_rho),
        float(drift_sigma),
        float(noise_sigma),
        float(Y),
        sig_jl,
        kv_jl,
        ab_jl,
        b_jl,
        3,
        theta_drift=float(theta_drift),
        price_tol=float(price_tol),
        max_price_iter=int(max_price_iter),
        k_sigma=float(k_sigma),
        neumann_k=int(neumann_k),
        rho_M_in=rm_jl,
        mu_t=float(mu_t),
        **hh_kwargs,
    )
    return {
        "C_monthly": np.asarray(res.C_monthly),
        "P_monthly": np.asarray(res.P_monthly),
        "P_final": np.asarray(res.P_final),
        "price_drift": float(res.price_drift),
        "signed_drift": float(res.signed_drift),
        "monthly_drifts": np.asarray(res.monthly_drifts),
        "monthly_resid_Y": np.asarray(res.monthly_resid_Y),
        "alpha_true_final": np.asarray(res.alpha_true_final),
        "C_hat": np.asarray(res.C_hat),
        "G_hat_bare": np.asarray(res.G_hat_bare),
        "alpha_h_final": np.asarray(res.alpha_h_final),
        "alpha_macro_final": np.asarray(res.alpha_macro_final),
        "rho_M": float(res.rho_M),
    }


def compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step, k=25):
    res = CORE.compute_investment(
        np.asarray(G_hat, dtype=np.float64),
        _to_dense(A_bar),
        _to_dense(B),
        np.asarray(C_prev, dtype=np.float64),
        np.asarray(G_vec, dtype=np.float64),
        float(g_step),
        float(c_step),
        k=int(k),
    )
    return np.array(res)


def solve_firm_lp(v_MIP, B_dense, K_firms, X_star, tol=0.001):
    v_jl = np.asarray(v_MIP, dtype=np.float64)
    B_jl = _to_dense(B_dense)
    K_jl = np.asarray(K_firms, dtype=np.float64)
    X_jl = np.asarray(X_star, dtype=np.float64)

    res = CORE.solve_firm_lp(v_jl, B_jl, K_jl, X_jl, float(tol))
    return np.asarray(res)
