import numpy as np
import logging
import pickle
import time
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.calibration import ModelState
from core.torch_core import (
    compute_investment,
    solve_planner,
    fast_loop,
    solve_firm_lp,
    neumann_apply,
)

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Raised when the simulation hits a terminal numerical state (e.g. NaNs)."""

    pass


def run_quarter(state: ModelState) -> ModelState:
    """
    Execute a single quarterly step of the macroeconomic simulation.

    This incorporates agent habit formation, central planner target optimization (CRRA),
    firm-level parallel production via JuMP, Laspeyres factor chaining for zero-inflation
    anchoring, and a multi-household market clearing loop for price discovery.
    """
    t = state.t
    slim = getattr(state, "slim_history", False)
    logger.info(f"\n-- Q{t + 1} {'-' * 50}")

    # --- Per-household Habit formation -------------------------------------------
    gamma_habit = getattr(state, "habit_persistence", 0.7)
    n_h = state.n_households

    if t == 0:
        state.alpha_habit_h = state.alpha_h.copy()
        state.alpha_habit = state.alpha_bar.copy()
    else:
        # We use alpha_true_h (the每月 evolved values returned from fast_loop)
        # as the realized preferences for each household.
        for h in range(n_h):
            alpha_hh = state.alpha_true_h[h, :]
            state.alpha_habit_h[h, :] = (
                gamma_habit * state.alpha_habit_h[h, :] + (1 - gamma_habit) * alpha_hh
            )
            state.alpha_habit_h[h, :] /= state.alpha_habit_h[h, :].sum()

        # Update aggregate habit as a mean for the planner/diagnostics
        state.alpha_habit = state.alpha_habit_h.mean(axis=0)
        state.alpha_habit /= state.alpha_habit.sum()

    state.alpha_slow_h = state.alpha_habit_h.copy()
    state.alpha_slow = state.alpha_habit.copy()

    # --- Growth signal smoothing -------------------------------------------------
    if t == 0:
        C_hat = state.C.copy()
        G_hat = state.G_hat_init.copy()
        G_hat_bare = state.G_hat_init.copy()
        state.G_hat_lagged = G_hat_bare.copy()
    else:
        G_hat_bare = getattr(state, "G_hat_bare_prev", state.G_hat_init)
        G_hat_prev = getattr(state, "G_hat_lagged", G_hat_bare)
        G_hat = (G_hat_bare + G_hat_prev) / 2.0
        state.G_hat_lagged = G_hat_bare.copy()

    logger.info(
        f"   G_hat  mean={G_hat.mean() * 100:.3f}%  max={G_hat.max() * 100:.3f}%"
    )

    # --- Government expenditure --------------------------------------------------
    state.G = state.G * (1.0 + state.g_step)
    G_vec = state.G
    logger.info(
        f"   G_gov = {G_vec.sum() / 1e9:.2f} B EUR  (g_step={state.g_step * 100:.2f}%)"
    )

    # --- Investment demand -------------------------------------------------------
    dK = compute_investment(
        G_hat,
        state.A_bar,
        state.B,
        state.C,
        G_vec,
        state.g_step,
        state.c_step,
        k=state.neumann_k,
    )
    logger.info(f"   dK = {dK.sum() / 1e9:.2f} B EUR")

    Nc = np.asarray(
        neumann_apply(
            state.A_bar.toarray()
            if hasattr(state.A_bar, "toarray")
            else np.asarray(state.A_bar),
            np.asarray(G_vec, dtype=np.float64),
            state.neumann_k,
        )
    )
    K_eff = state.K - state.B @ Nc
    L_eff = float(state.L_total - state.l_vec @ Nc)

    M_G = state.B @ Nc
    n_negative = int((K_eff <= 0).sum())
    frac_K_rem = float(K_eff.sum()) / max(float(state.K.sum()), 1e-30)
    logger.info(f"   [DIAG] G_raw total      = {G_vec.sum() / 1e9:.2f} B EUR")
    logger.info(
        f"   [DIAG] M·G (cap req)    = {M_G.sum() / 1e9:.2f} B EUR  "
        f"(K used by govt: {(1 - frac_K_rem) * 100:.1f}%)"
    )
    logger.info(
        f"   [DIAG] K_eff total      = {K_eff.sum() / 1e9:.2f} B EUR  "
        f"({frac_K_rem * 100:.1f}% of K remaining)"
    )
    logger.info(f"   [DIAG] Sectors K_eff<=0 = {n_negative}")
    logger.info(
        f"   [DIAG] L_eff            = {L_eff / 1e9:.2f} B hrs  "
        f"({L_eff / max(state.L_total, 1e-30) * 100:.1f}% of L remaining)"
    )

    # --- Planner optimisation ----------------------------------------------------
    opt = solve_planner(
        state.alpha,
        state.A_bar,
        state.B,
        state.l_tilde,
        dK,
        K_eff,
        L_eff,
        G_vec,
        state.sigma_vec,
        state.C,
        k=state.neumann_k,
        tol_p=state.primal_tol,
        tol_d=state.dual_tol,
        eta_K=state.eta_K,
        eta_L=state.eta_L,
        max_iter=state.max_iter,
    )

    for k in ["lam_K", "lam_L", "C_star"]:
        if opt.get(k) is None:
            logger.error(f"   [CRITICAL] opt['{k}'] is None!")

    C_star = opt["C_star"]
    X_star = opt["X_star"]
    pi = opt["pi_star"]

    # --- Marginal Utility Chaining (Enforce Zero Inflation per Eq. 45) ------------
    # mu_t / mu_{t-1} = (pi_t @ C_{t-1}) / (pi_{t-1} @ C_{t-1})
    pi_prev = state.pi_prev
    # Use previous quarter's consumption for the Laspeyres anchor
    C_prev = state.C

    # Calculate shadow inflation from previous optimal state
    shadow_inflation = float(np.dot(pi, C_prev) / max(np.dot(pi_prev, C_prev), 1e-30))
    state.mu_planner *= shadow_inflation
    state.pi_prev = pi.copy()

    # Update nominal income Y to maintain budget clearing at chained prices:
    # Y_t = (pi_t @ C_t*) / mu_t
    state.Y = float(np.dot(pi, C_star) / max(state.mu_planner, 1e-30))

    logger.info(f"   [DIAG] mu_planner       = {state.mu_planner:.4e}")
    logger.info(f"   [DIAG] nominal_income Y = {state.Y / 1e9:.2f} B EUR")
    logger.info(
        f"   [DIAG] C_star total     = {C_star.sum() / 1e9:.2f} B EUR  "
        f"({'converged' if opt['success'] else 'NOT CONVERGED'}, "
        f"{opt.get('iterations', -1)} iters)"
    )
    logger.info(f"   [DIAG] Sectors C_star=0 = {int((C_star == 0).sum())}")

    if not opt["success"]:
        logger.warning(
            f"[Q{t + 1}] solve_planner did not converge -- "
            f"using best iterate. Resetting warm-start for Q{t + 2}."
        )
        opt["lambda_K"] = None
        opt["lambda_L"] = None

    if np.isnan(C_star).any() or np.isnan(X_star).any() or np.isnan(pi).any():
        raise SimulationError(f"Q{t + 1}: Solver produced NaNs")

    # --- Income ------------------------------------------------------------------
    if t == 0:
        state.pi_0_fixed = pi.copy()
        state.dual_weight_0 = pi - state.A.T @ pi
        state.C_0_init = C_star.copy()
        lK = opt.get("lam_K")
        state.lambda_K_0 = lK.copy() if lK is not None else np.zeros_like(C_star)
        lL = opt.get("lam_L")
        state.lambda_L_0 = float(lL) if lL is not None else 0.0
        state.real_scale_factor = state.Y_0_init / max(float(pi @ C_star), 1e-30)
        state.VAL_0_Laspeyres = float((pi * state.C_0_init).sum())

    BT_lamK = (state.B.T @ state.lambda_K_0).astype(np.float64)
    A_dense_T = state.A_bar.T.toarray().astype(np.float64)
    M_T_lamK = np.asarray(neumann_apply(A_dense_T, BT_lamK, int(state.neumann_k)))
    W1 = state.lambda_L_0 * state.l_vec + M_T_lamK
    sector_demand = C_star + G_vec + dK
    shadow_net_product = float(pi @ sector_demand)
    v_eff = W1 / max(shadow_net_product, 1e-30)
    v_MIP = v_eff

    # --- Firm production LP ------------------------------------------------------
    t_pre_lp = time.time()
    B_dense = state.B.toarray() if hasattr(state.B, "toarray") else np.asarray(state.B)

    # Offload the massive generic sparse block-angular constraint LP directly
    # to the compiled Julia-side Highs optimizer across all sectors simultaneously
    X_f = solve_firm_lp(v_MIP, B_dense, state.K_firms, X_star, tol=0.001)

    t_post_lp = time.time()
    logger.info(f"   [TIMING] solve_firm_lp took {t_post_lp - t_pre_lp:.3f}s")

    total_cap_all_sectors = np.zeros(state.n)
    for j in range(state.n):
        col_B = B_dense[:, j]
        nz_idx = col_B > 1e-12
        if nz_idx.any():
            cap_caps = np.min(state.K_firms[:, nz_idx] / col_B[nz_idx], axis=1)
            total_cap_all_sectors[j] = cap_caps.sum()
        else:
            total_cap_all_sectors[j] = 1e30

    X_actual = X_f.sum(axis=1)
    # Relative RMSE: ||X_star - X_actual|| / ||X_star||
    rrmse = float(
        np.linalg.norm(X_actual - X_star) / max(np.linalg.norm(X_star), 1e-30)
    )

    X_star_cap_violation = np.maximum(X_star - total_cap_all_sectors, 0.0)
    max_violation = np.max(
        X_star_cap_violation / np.maximum(total_cap_all_sectors, 1e-12)
    )
    # if max_violation > 0.0025:
    #     logger.warning(f"   [WARNING] Planner target exceeds sectoral capacity by {max_violation*100:.2f}%")
    # if rrmse > 1e-4:
    #     logger.warning(f"   [WARNING] Firm allocation RRMSE = {rrmse:.2e}")

    t_post_cap = time.time()
    logger.info(
        f"   [TIMING] post_lp capacity checking took {t_post_cap - t_post_lp:.3f}s"
    )

    # --- Capital and Investment update (Post-Production) -------------------------
    # Now that we have X_actual, we compute total investment actually realized
    # subtracting consumption target and govt from actual gross output.
    I_gross = np.maximum(X_actual - (state.A @ X_actual + C_star + G_vec), 0.0)
    I_vec = I_gross

    # Firm-level investment: each firm receives a proportional share of gross
    # investment, preserving the identity dK_f,i = (K_f,i / K_i) * dK_i.
    K_total_i = state.K_firms.sum(axis=0)
    safe_denom = np.where(K_total_i > 1e-12, K_total_i, 1.0)
    firm_capital_share = state.K_firms / safe_denom[None, :]
    firm_capital_share = np.where(
        K_total_i[None, :] > 1e-12, firm_capital_share, 1.0 / state.n_firms
    )
    I_firms = firm_capital_share * I_vec[None, :]
    K_firms_new = (1 - state.delta) * state.K_firms + I_firms

    # Aggregate capital is derived from firm layer (accounting identity: K ≡ Σ K_firms)
    K_new = K_firms_new.sum(axis=0)

    if np.isnan(K_new).any() or np.isnan(I_gross).any():
        raise SimulationError(f"Q{t + 1}: Capital/Investment update produced NaNs")

    t_post_inv = time.time()
    logger.info(f"   [TIMING] Investment update took {t_post_inv - t_post_cap:.3f}s")

    # --- Firm income and aggregate Y ---------------------------------------------
    Y_f_mat = v_eff[:, None] * X_f  # (N, n_firms)
    Y_f = Y_f_mat.sum(axis=0)  # (n_firms,) per-firm total income

    # income_scale is calibrated once at Q1 and held fixed thereafter
    if t == 0:
        state.income_scale = state.Y_0_init / max(float(Y_f.sum()), 1e-30)

    Y = float(Y_f.sum()) * state.income_scale
    Y_planner = float(v_eff @ X_star) * state.income_scale

    # --- Prices and inflation ----------------------------------------------------
    # mu_planner is (pi @ C) / Y => P = pi / mu_planner
    P_new = pi / max(state.mu_planner, 1e-30)

    denom_inf = np.dot(state.P, state.C)
    inflation_link = np.dot(P_new, state.C) / max(denom_inf, 1e-30)
    inflation_rate = inflation_link - 1.0

    # Geometric mean inflation
    price_relatives = P_new / np.where(state.P > 1e-30, state.P, 1e-30)
    valid_mask = (state.P > 1e-6) & (P_new > 1e-6)
    if valid_mask.any():
        inflation_rate_geom = np.exp(np.mean(np.log(price_relatives[valid_mask]))) - 1.0
    else:
        inflation_rate_geom = 0.0

    if t > 0:
        state.CPI_chained *= inflation_link
    else:
        state.CPI_chained = 1.0

    opt_iter = opt.get("iterations", -1)
    opt_mvps = opt.get("mvps", -1)
    logger.info(
        f"   C* = {(P_new * C_star).sum() / 1e9:.1f} B EUR  "
        f"X_actual = {(P_new * X_actual).sum() / 1e9:.1f} B EUR  "
        f"(X* target: {(P_new * X_star).sum() / 1e9:.1f} B EUR)  "
        f"{'(OK)' if opt['success'] else '(WARNING)'}  ({opt_iter} iters, {opt_mvps} MVPs)"
    )
    logger.info(
        f"   Y_firm = {Y / 1e9:.2f} B EUR  Y_planner = {Y_planner / 1e9:.2f} B EUR  "
        f"(gap = {abs(Y - Y_planner) / max(Y, 1e-30) * 100:.3f}%)"
    )

    # --- Multi-household demand aggregation --------------------------------------
    # Distribute firm income to households via ownership matrix W
    n_h = state.n_households
    Y_f_scaled = Y_f * state.income_scale  # (n_firms,) scaled firm incomes
    Y_h = state.W_ownership @ Y_f_scaled  # (n_h,) household incomes
    state.w_h = Y_h / max(Y_h.sum(), 1e-30)  # Update welfare weights per Eq. 115

    # Physical consumption achievable from ACTUAL production (Net of Depreciation!)
    # We use A_bar here to reserve goods for replacement investment.
    # We add a tiny epsilon (1e-6) to ensure no sector is 'zero-supply', which prevents price spikes.
    C_actual = X_actual - state.A_bar @ X_actual - dK - G_vec
    C_actual = np.maximum(C_actual, 1e-6)

    t_pre_fast_loop = time.time()
    logger.info(
        f"   [TIMING] pre-fast_loop intermediate calcs took {t_pre_fast_loop - t_post_inv:.3f}s"
    )

    # Intra-quarter tatonnement: run with multi-household demand aggregation
    fast_res = fast_loop(
        P_new,
        C_actual,
        state.alpha_macro,
        state.alpha_slow,
        state.rng,
        state.pref_drift_rho,
        state.pref_drift_sigma,
        state.pref_noise_sigma,
        Y,
        state.sigma_vec,
        K_eff,
        state.A_bar,
        state.B,
        theta_drift=state.theta_drift,
        alpha_h=state.alpha_true_h,
        Y_h=Y_h,
        w_h=state.w_h,
        mu_t=state.mu_planner
        * 3.0,  # Convert quarterly MU to monthly (3 months in quarter)
        alpha_slow_h=state.alpha_slow_h,
        price_tol=state.price_tol,
        max_price_iter=state.max_price_iter,
        k_sigma=state.cybernetic_k_sigma,
        neumann_k=state.neumann_k,
        rho_M=getattr(state, "rho_M", None),
    )

    t_post_fast_loop = time.time()
    logger.info(f"   [TIMING] fast_loop took {t_post_fast_loop - t_pre_fast_loop:.3f}s")

    # Use the cleared prices from fast_loop for household demand
    P_final = fast_res["P_final"]

    # Per-household preferences are now evolved monthly inside fast_loop via
    # independent LN-AR processes; pick up the end-of-quarter state.
    alpha_h_evolved = fast_res.get("alpha_h_final")
    if alpha_h_evolved is not None and alpha_h_evolved.size > 0:
        state.alpha_true_h = alpha_h_evolved

    # Retrieve the pure, revealed preference estimates aggregated across the
    # market clearing loop internally by Julia.
    state.alpha_true = np.array(fast_res["alpha_true_final"]).copy()

    alpha_gap = float(np.linalg.norm(state.alpha_true - state.alpha))
    alpha_gap_linf = float(np.abs(state.alpha_true - state.alpha).max())
    logger.info(
        f"   alpha_true gap (multi-HH): norm={alpha_gap:.4f}  inf={alpha_gap_linf:.4f}"
    )

    # --- Preference update -------------------------------------------------------
    state.P = fast_res["P_final"]
    state.C = fast_res["C_monthly"].mean(axis=0) * 3
    state.C_monthly = fast_res["C_monthly"]
    state.P_monthly = fast_res["P_monthly"]
    state.price_drift = fast_res["price_drift"]
    state.alpha_macro = fast_res["alpha_macro_final"]
    state.rho_M = fast_res["rho_M"]

    # Planner alpha update: Smoothed move toward revealed preferences + epsilon floor
    alpha_new = np.array(fast_res["alpha_true_final"]).copy()
    alpha_new += 1e-5
    alpha_new /= alpha_new.sum()

    # --- Real metrics (constant Q1 prices) ---------------------------------------
    Real_GDP = state.real_scale_factor * float((state.dual_weight_0 * X_actual).sum())
    resource_value = W1 @ X_actual
    Y_ratio = Y / max(Real_GDP, 1e-30)
    Gross_Output_Real = state.real_scale_factor * float(
        (X_actual * state.pi_0_fixed).sum()
    )
    C_real = state.real_scale_factor * float((C_star * state.pi_0_fixed).sum())
    C_realized = state.real_scale_factor * float(
        (fast_res["C_monthly"].sum(axis=0) * state.pi_0_fixed).sum()
    )
    I_gross_real = state.real_scale_factor * float((I_vec * state.pi_0_fixed).sum())
    I_net_real = state.real_scale_factor * float((dK * state.pi_0_fixed).sum())
    G_real = state.real_scale_factor * float((G_vec * state.pi_0_fixed).sum())
    Real_AD = C_real + I_gross_real + G_real
    Real_AD_realized = C_realized + I_gross_real + G_real

    cap_used = state.B @ X_actual
    cap_slack = state.K - cap_used
    P_0_prices = state.P_0 if len(state.P_0) == state.n else np.zeros(state.n)
    K_val_Q1 = float((state.K * P_0_prices).sum())
    slack_val_Q1 = float((cap_slack * P_0_prices).sum())
    used_val_Q1 = K_val_Q1 - slack_val_Q1

    labor_used = float((state.l_vec * X_actual).sum())
    labor_slack = (state.L_total - labor_used) / max(state.L_total, 1e-12)

    logger.info(
        f"   Real GDP (ann.) = {Real_GDP * 4 / 1e9:.1f} B EUR  "
        f"Real AD (ann.) = {Real_AD * 4 / 1e9:.1f} B EUR"
    )
    logger.info(
        f"   (C = {C_real * 4 / 1e9:.1f}B, I = {I_gross_real * 4 / 1e9:.1f}B, "
        f"G = {G_real * 4 / 1e9:.1f}B  — annualised)"
    )
    logger.info(
        f"   Capital (Q1 prices): K_total = {K_val_Q1 / 1e9:.1f} B EUR  "
        f"slack = {slack_val_Q1 / 1e9:.1f} B EUR  "
        f"({slack_val_Q1 / max(K_val_Q1, 1e-30) * 100:.1f}% idle)"
    )
    logger.info(f"   Labor utilization: {(1 - labor_slack) * 100:.2f}%")

    assert abs(alpha_new.sum() - 1.0) < 1e-8, "alpha no longer sums to 1"
    assert (K_new >= 0).all(), "Negative capital stock"
    if (C_star < 0).any():
        logger.warning("C_star has strictly negative elements (constraint violation)")

    I_pct_GDP = (I_gross_real / max(Real_GDP, 1e-30)) * 100.0

    ed_nom_val = 0.0
    exp_nom_val = 0.0
    for tau in range(3):
        p_tau = fast_res["P_monthly"][tau]
        c_rev_tau = fast_res["C_monthly"][tau]
        ed_nom_val += float(((c_rev_tau - C_star / 3.0) * p_tau).sum())
        exp_nom_val += float((c_rev_tau * p_tau).sum())

    logger.info(
        f"   [DIAG] Price drift from planner baseline: "
        f"mean={fast_res['price_drift'] * 100:.2f}%  signed={fast_res['signed_drift'] * 100:.2f}%"
    )

    common = dict(
        t=t + 1,
        GDP=Real_GDP,
        Real_AD=Real_AD,
        Y=Y,
        Y_planner=Y_planner,
        Y_f=Y_f.copy(),
        Inflation=inflation_rate if t > 0 else 0.0,
        Inflation_Geom=inflation_rate_geom if t > 0 else 0.0,
        Gross_Output_Real=Gross_Output_Real,
        C_real=C_real,
        C_realized=C_realized,
        I_gross_real=I_gross_real,
        I_net_real=I_net_real,
        G_real=G_real,
        Real_AD_realized=Real_AD_realized,
        K_val_Q1=K_val_Q1,
        slack_val_Q1=slack_val_Q1,
        used_val_Q1=used_val_Q1,
        labor_slack=labor_slack,
        I_pct_GDP=I_pct_GDP,
        iterations=opt.get("iterations", 0),
        alpha_gap=alpha_gap,
        alpha_gap_linf=alpha_gap_linf,
        price_drift=float(fast_res["price_drift"]),
        monthly_resid_Y=fast_res["monthly_resid_Y"],
        resource_value=resource_value,
        shadow_net_product=shadow_net_product,
        Y_ratio=Y_ratio,
        lambda_K_mean=float(opt["lam_K"].mean()) if opt["lam_K"] is not None else 0.0,
        lambda_K_max=float(opt["lam_K"].max()) if opt["lam_K"] is not None else 0.0,
        lambda_L=float(opt["lam_L"]) if opt["lam_L"] is not None else 0.0,
        C_rev=fast_res["C_monthly"].mean(axis=0),
        ED_nom=ed_nom_val,
        Exp_nom=exp_nom_val,
        sector_short=state.sector_short.copy(),
        v_MIP=v_MIP.copy(),
        X_f=X_f.copy(),
        RRMSE=rrmse,
    )

    # --- Planner optimal household allocation ------------------------------------
    # Under CRRA, optimal C_{h,i}^* = (w_h * alpha_{h,i} / pi_i)^{1 / sigma_i}
    C_h_star_raw = (
        state.w_h[:, None] * state.alpha_true_h / np.maximum(pi[None, :], 1e-15)
    ) ** (1.0 / state.sigma_vec[None, :])
    # Scale to exactly match aggregate C_star (in case of slight numerical drift)
    C_h_sum = C_h_star_raw.sum(axis=0)
    scale_factor = np.where(C_h_sum > 1e-30, C_star / C_h_sum, 1.0)
    C_h_star = C_h_star_raw * scale_factor[None, :]

    if slim:
        state.history.append(
            {
                **common,
                **dict(
                    C_total=float((state.P * C_star).sum()),
                    X_actual_total=float((state.P * X_actual).sum()),
                    X_star_total=float((state.P * X_star).sum()),
                    I_total=float((state.P * I_vec).sum()),
                    G_total=float((state.P * state.G).sum()),
                    K_total=float((state.P * state.K).sum()),
                    G_mean=float(G_hat.mean()),
                    G_max=float(G_hat.max()),
                    # Keep household metadata and per-household optimal consumption
                    alpha_true_h=state.alpha_true_h.copy(),
                    w_h=state.w_h.copy(),
                    C_h_star=C_h_star.copy(),
                ),
            }
        )
    else:
        state.history.append(
            {
                **common,
                **dict(
                    C_star=C_star.copy(),
                    X_actual=X_actual.copy(),
                    X_star=X_star.copy(),
                    I_vec=I_vec.copy(),
                    G_vec=state.G.copy(),
                    K=state.K.copy(),
                    K_firms=state.K_firms.copy(),
                    pi_star=pi.copy(),
                    alpha=state.alpha.copy(),
                    alpha_true=state.alpha_true.copy(),
                    alpha_true_h=state.alpha_true_h.copy(),
                    w_h=state.w_h.copy(),
                    C_h_star=C_h_star.copy(),
                    G_hat_mean=float(G_hat.mean()),
                    CPI=state.CPI_chained,
                ),
            }
        )

    # --- State update ------------------------------------------------------------
    state.t = t + 1
    state.K_firms = K_firms_new
    state.K = K_new  # accounting identity: K = K_firms.sum(axis=0)

    assert np.allclose(state.K, state.K_firms.sum(axis=0), rtol=1e-5), (
        "K ≠ Σ K_firms — accounting identity violated"
    )

    state.alpha = alpha_new
    state.P = P_new
    state.C = C_star
    state.X = X_actual
    state.X_star = X_star
    state.X_f = X_f.copy()
    state.Y = Y
    state.C_monthly = fast_res["C_monthly"]
    state.P_monthly = fast_res["P_monthly"]
    state.G_hat_bare_prev = fast_res["G_hat_bare"]
    state.lambda_K = opt["lam_K"]
    state.lambda_L = opt["lam_L"]

    t_end = time.time()
    logger.info(f"   [TIMING] final state updates took {t_end - t_post_fast_loop:.3f}s")
    return state


def _print_investment_gdp_summary(history: list, start_year: int = 2022) -> None:
    """Print annual I/GDP ratios with 5-year rolling averages."""
    n_quarters = len(history)
    n_years = n_quarters // 4
    leftover = n_quarters % 4

    annual_ratios, annual_years = [], []
    for y in range(n_years):
        bucket = history[y * 4 : y * 4 + 4]
        annual_ratios.append(np.mean([h["I_pct_GDP"] for h in bucket]))
        annual_years.append(start_year + y)

    partial_label = partial_avg = None
    if leftover > 0:
        bucket = history[n_years * 4 :]
        partial_avg = np.mean([h["I_pct_GDP"] for h in bucket])
        partial_label = f"{start_year + n_years} (partial, {leftover}Q)"

    logger.info("")
    logger.info("  Investment / GDP  —  Annual Averages & 5-Year Rolling Mean")
    logger.info("  " + "-" * 54)
    logger.info(f"  {'Year':<20}  {'I / GDP (%)':<12}  {'5-yr Avg (%)':<14}")
    logger.info("  " + "-" * 54)

    for i, (yr, ratio) in enumerate(zip(annual_years, annual_ratios)):
        window = annual_ratios[max(0, i - 4) : i + 1]
        rolling5 = np.mean(window)
        win_lbl = f"({len(window)}-yr avg)" if len(window) < 5 else "(5-yr avg)"
        logger.info(f"  {str(yr):<20}  {ratio:>12.2f}  {rolling5:>10.2f}  {win_lbl}")

    if partial_label and partial_avg is not None:
        logger.info(f"  {partial_label:<20}  {partial_avg:>12.2f}")

    logger.info("  " + "-" * 54)
    all_ratios = annual_ratios + ([partial_avg] if partial_avg is not None else [])
    if len(all_ratios) >= 5:
        logger.info(
            f"  5-Year Average (last 5 obs):        {np.mean(all_ratios[-5:]):>8.2f} %"
        )
    elif len(all_ratios) > 0:
        logger.info(
            f"  Overall Average ({len(all_ratios)} years):       {np.mean(all_ratios):>8.2f} %"
        )
    logger.info("")


def run_simulation(
    state: ModelState,
    n_quarters: int = 20,
    checkpoint_dir: Path = None,
    checkpoint_every: int = 5,
) -> ModelState:
    slim = getattr(state, "slim_history", False)
    logger.info(f"\n{'#' * 60}")
    logger.info(f"  Cybernetic In-Kind Planning Simulation")
    logger.info(
        f"  n = {state.n:,}   T = {n_quarters}   delta = {state.delta}"
        f"   {'slim history' if slim else 'full history'}"
    )
    logger.info(f"{'#' * 60}")

    step_times = []
    for i in range(n_quarters):
        t0 = time.time()
        try:
            run_quarter(state)
        except SimulationError as e:
            logger.error(f"   [Simulation] TERMINATED at Q{state.t + 1}: {e}")
            break
        t1 = time.time()
        step_times.append(t1 - t0)
        logger.info(f"   [Step completed in {t1 - t0:.2f}s]")

        if checkpoint_dir and (i + 1) % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            chk_path = checkpoint_dir / f"checkpoint_q{state.t}.pkl"
            with open(chk_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"  Saved checkpoint to {chk_path}")

    g0 = state.history[0]["GDP"]
    gT = state.history[-1]["GDP"]

    # Calculate geometric mean of the time-series inflation
    inflations = [h.get("Inflation", 0.0) for h in state.history[1:]]
    if inflations:
        overall_geom_inf = np.exp(np.mean(np.log(1 + np.array(inflations)))) - 1.0
    else:
        overall_geom_inf = 0.0

    logger.info(f"\n{'#' * 60}")
    logger.info(
        f"  Done.  Annualised GDP: {g0 * 4 / 1e9:.1f}B → {gT * 4 / 1e9:.1f}B EUR  "
        f"({(gT / g0 - 1) * 100:+.1f}% cumulative)"
    )
    logger.info(
        f"  Geometric Mean Inflation (Time-Series Avg): {overall_geom_inf * 100:.3f}% per quarter"
    )

    if len(step_times) > 1:
        avg_time = sum(step_times[1:]) / (len(step_times) - 1)
        logger.info(
            f"  Avg time per quarter (excluding Q1 compilation): {avg_time:.2f}s"
        )
    elif len(step_times) == 1:
        logger.info(f"  Time for Q1 (includes compilation): {step_times[0]:.2f}s")
    logger.info(f"{'#' * 60}")

    _print_investment_gdp_summary(state.history)
    return state
