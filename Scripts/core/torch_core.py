import torch
import numpy as np
from scipy.optimize import linprog
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device_str):
    global device
    device = torch.device(device_str)


def _to_tensor(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype)
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def neumann_apply(A, v, k=20):
    A_t = _to_tensor(A)
    v_t = _to_tensor(v)
    res = v_t.clone()
    term = v_t.clone()
    for _ in range(k):
        term = torch.matmul(A_t, term)
        res += term
    return _to_numpy(res)


def compute_investment(G_hat, A_bar, B, C_prev, G_vec, g_step, c_step, k=25):
    G_hat_t = _to_tensor(G_hat)
    A_m = _to_tensor(A_bar)
    B_m = _to_tensor(B)
    C_prev_v = _to_tensor(C_prev)
    G_vec_v = _to_tensor(G_vec)
    g = float(g_step)

    def B_apply(v):
        return torch.matmul(B_m, v)

    gC = torch.clamp(G_hat_t, min=0.0)

    Gv = gC * C_prev_v
    term1_C = B_apply(_to_tensor(neumann_apply(A_m, Gv, k)))
    term2_C = B_apply(_to_tensor(neumann_apply(A_m, gC * term1_C, k)))
    term3_C = B_apply(_to_tensor(neumann_apply(A_m, gC * term2_C, k)))

    G_v_g = g * G_vec_v
    term1_G = B_apply(_to_tensor(neumann_apply(A_m, G_v_g, k)))
    term2_G = B_apply(_to_tensor(neumann_apply(A_m, g * term1_G, k)))
    term3_G = B_apply(_to_tensor(neumann_apply(A_m, g * term2_G, k)))

    res = term1_C + term2_C + term3_C + term1_G + term2_G + term3_G
    return _to_numpy(res)


def solve_planner(
    alpha,
    A_bar,
    B,
    l_tilde,
    dK,
    K,
    L_total,
    G_vec,
    sigma_v,
    C_prev,
    k=20,
    tol_p=1e-4,
    tol_d=1e-4,
    eta_K=0.25,
    eta_L=0.35,
    max_iter=2000,
    L0=1.0,
    L_scale_up=2.0,
    L_scale_dn=1.05,
):
    eps_val = 1e-15
    alpha_t = _to_tensor(alpha)
    A_m = _to_tensor(A_bar)
    B_m = _to_tensor(B)
    l_tilde_t = _to_tensor(l_tilde)
    dK_t = _to_tensor(dK)
    K_t = _to_tensor(K)
    L_total_t = float(L_total)
    G_t = _to_tensor(G_vec)
    sigma_t = _to_tensor(sigma_v)

    n = len(alpha_t)
    A_m_T = A_m.T
    B_m_T = B_m.T

    def Bt(v):
        return torch.matmul(B_m, _to_tensor(neumann_apply(A_m, v, k)))

    def BtT(w):
        return _to_tensor(neumann_apply(A_m_T, torch.matmul(B_m_T, w), k))

    K_eff = K_t - Bt(dK_t)
    L_eff = L_total_t - torch.dot(l_tilde_t, dK_t).item()

    log_lam_K = torch.zeros(n, dtype=torch.float64, device=device)
    log_lam_L = torch.tensor(0.0, dtype=torch.float64, device=device)

    log_y_K = log_lam_K.clone()
    log_y_L = log_lam_L.clone()

    log_x_K = log_lam_K.clone()
    log_x_L = log_lam_L.clone()

    tk_curr = 1.0
    L_curr = L0
    total_mvps = 2

    converged = False
    opt_iter = max_iter

    for i_iter in range(1, max_iter + 1):
        opt_iter = i_iter

        y_K = torch.exp(log_y_K)
        y_L = torch.exp(log_y_L)
        pi_vec = BtT(y_K) + y_L * l_tilde_t
        total_mvps += 1

        C_res = (
            torch.clamp(alpha_t, min=1e-15) / torch.clamp(pi_vec, min=eps_val)
        ) ** (1.0 / sigma_t)

        s_K = K_eff - Bt(C_res)
        s_L = L_eff - torch.dot(l_tilde_t, C_res)
        total_mvps += 1

        grad_K = -s_K / torch.clamp(K_eff, min=1e-12)
        grad_L = -s_L / max(abs(L_total_t), 1e-12)

        if (
            torch.all(torch.abs(y_K * s_K) <= tol_d)
            and abs((y_L * s_L).item()) <= tol_d
            and torch.all(
                torch.abs(torch.clamp(s_K, max=0.0)) / torch.clamp(K_eff, min=eps_val)
                <= tol_p
            )
            and (abs(torch.clamp(s_L, max=0.0).item()) / (abs(L_eff) + eps_val))
            <= tol_p
        ):
            converged = True
            break

        if torch.isnan(grad_K).any() or torch.isnan(grad_L):
            print(f"   [CRITICAL] FISTA: NaNs at iteration {i_iter}")
            break

        obj_y = torch.dot(y_K, K_eff) + y_L * L_eff
        sig_mask_1 = torch.abs(sigma_t - 1.0) < 1e-6
        p_clamped = torch.clamp(pi_vec, min=eps_val)
        al_clamped = torch.clamp(alpha_t, min=eps_val)

        obj_y += torch.sum(
            torch.where(
                sig_mask_1,
                alpha_t * torch.log(al_clamped / p_clamped) - alpha_t,
                (sigma_t / (1.0 - sigma_t))
                * (alpha_t ** (1.0 / sigma_t))
                * (p_clamped ** (1.0 - 1.0 / sigma_t)),
            )
        )

        L_curr /= L_scale_dn

        while True:
            step_K = eta_K / L_curr
            step_L = eta_L / L_curr

            log_x_K_new = log_y_K + step_K * grad_K
            log_x_L_new = log_y_L + step_L * grad_L

            x_K_try = torch.exp(log_x_K_new)
            x_L_try = torch.exp(log_x_L_new)
            pi_try = BtT(x_K_try) + x_L_try * l_tilde_t
            total_mvps += 1

            obj_try = torch.dot(x_K_try, K_eff) + x_L_try * L_eff
            p_try_clamped = torch.clamp(pi_try, min=eps_val)

            obj_try += torch.sum(
                torch.where(
                    sig_mask_1,
                    alpha_t * torch.log(al_clamped / p_try_clamped) - alpha_t,
                    (sigma_t / (1.0 - sigma_t))
                    * (alpha_t ** (1.0 / sigma_t))
                    * (p_try_clamped ** (1.0 - 1.0 / sigma_t)),
                )
            )

            if obj_try <= obj_y or L_curr > 1e10:
                break
            L_curr *= L_scale_up

        if torch.dot(log_y_K - log_x_K, grad_K) + (log_y_L - log_x_L) * grad_L < 0:
            tk_curr = 1.0

        tk_next = (1.0 + np.sqrt(1.0 + 4.0 * tk_curr**2)) / 2.0
        beta_t = (tk_curr - 1.0) / tk_next

        log_y_K = log_x_K_new + beta_t * (log_x_K_new - log_x_K)
        log_y_L = log_x_L_new + beta_t * (log_x_L_new - log_x_L)

        log_x_K = log_x_K_new.clone()
        log_x_L = log_x_L_new.clone()
        tk_curr = tk_next

    lam_K = torch.exp(log_x_K)
    lam_L = torch.exp(log_x_L)
    pi_vec_star = BtT(lam_K) + lam_L * l_tilde_t
    C_star = (
        torch.clamp(alpha_t, min=1e-15) / torch.clamp(pi_vec_star, min=eps_val)
    ) ** (1.0 / sigma_t)
    X_star = _to_tensor(neumann_apply(A_m, C_star + dK_t + G_t, k))

    return {
        "C_star": _to_numpy(C_star),
        "X_star": _to_numpy(X_star),
        "pi_star": _to_numpy(pi_vec_star),
        "success": converged,
        "lam_K": _to_numpy(lam_K),
        "lam_L": lam_L.item(),
        "iterations": opt_iter,
        "mvps": total_mvps + 2,
    }


def soft_clamp(s, s_max):
    return s / (1.0 + torch.abs(s) / s_max)


def fast_loop(
    P_base,
    C_plan,
    alpha_true_start,
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
    n_months=3,
    theta_drift=0.1,
    max_price_iter=50,
    price_tol=0.005,
    price_step_cap=0.5,
    alpha_h=None,
    Y_h=None,
    w_h=None,
    alpha_slow_h=None,
    k_sigma=1.0,
    neumann_k=20,
    rho_M=None,
    mu_t=1.0,
):
    rho_M_in = rho_M if rho_M is not None else -1.0
    n = len(P_base)
    multi_hh = alpha_h is not None
    n_h_count = alpha_h.shape[0] if multi_hh else 0

    if multi_hh:
        alpha_h_ev = _to_tensor(alpha_h).clone()

    P_base_t = _to_tensor(P_base)
    C_plan_t = _to_tensor(C_plan)
    alpha_true_start_t = _to_tensor(alpha_true_start)
    alpha_s_t = _to_tensor(alpha_slow)
    sigma_t = _to_tensor(sigma_v)
    A_bar_t = _to_tensor(A_bar)
    B_t = _to_tensor(B)

    C_m = C_plan_t / n_months
    Y_m = Y / n_months

    if multi_hh:
        Y_h_m = None if Y_h is None else _to_tensor(Y_h) / n_months
        w_h_t = (
            _to_tensor(w_h)
            if w_h is not None
            else torch.full(
                (n_h_count,), 1.0 / n_h_count, device=device, dtype=torch.float64
            )
        )

    C_monthly = torch.zeros((n_months, n), dtype=torch.float64, device=device)
    P_monthly = torch.zeros((n_months, n), dtype=torch.float64, device=device)

    C_hat_sum = torch.zeros(n, dtype=torch.float64, device=device)
    a_reveal_sum = torch.zeros(n, dtype=torch.float64, device=device)
    monthly_drifts = torch.zeros(n_months, dtype=torch.float64, device=device)
    monthly_resid_Y = torch.zeros(n_months, dtype=torch.float64, device=device)

    a_f = alpha_true_start_t.clone()
    active = a_f > 0

    # Handle random shocks locally to PyTorch
    agg_shocks_drift = (
        torch.randn(n_months, n, device=device, dtype=torch.float64) * drift_sigma
    )
    agg_shocks_noise = (
        torch.randn(n_months, n, device=device, dtype=torch.float64) * noise_sigma
    )

    noise_persistent = torch.zeros(n, dtype=torch.float64, device=device)

    if multi_hh:
        hh_shocks_drift = (
            torch.randn(n_months, n_h_count, n, device=device, dtype=torch.float64)
            * drift_sigma
        )
        hh_shocks_noise = (
            torch.randn(n_months, n_h_count, n, device=device, dtype=torch.float64)
            * noise_sigma
        )

    rho_M = rho_M_in
    if rho_M < 0.0:
        x_pi = torch.rand(n, device=device, dtype=torch.float64)
        x_pi_norm = torch.norm(x_pi)
        if x_pi_norm > 1e-12:
            x_pi /= x_pi_norm
            for _ in range(100):
                y_pi = torch.matmul(
                    B_t, _to_tensor(neumann_apply(A_bar_t, x_pi, neumann_k))
                )
                g_vec = y_pi / torch.clamp(x_pi, min=1e-16)
                g_mean = torch.mean(g_vec)
                g_mad = torch.mean(torch.abs(g_vec - g_mean))
                rmad = (g_mad / g_mean).item() if g_mean.item() > 1e-16 else 1.0
                rho_M = g_mean.item()
                if rmad <= 0.01:
                    break
                nrm = torch.norm(y_pi)
                if nrm > 1e-12:
                    x_pi = y_pi / nrm
                else:
                    break

    for tau in range(n_months):
        log_f = torch.where(
            a_f > 1e-25,
            torch.log(torch.clamp(a_f, min=1e-30)),
            torch.tensor(-25.0, device=device, dtype=torch.float64),
        )
        log_s = torch.where(
            alpha_s_t > 1e-25,
            torch.log(torch.clamp(alpha_s_t, min=1e-30)),
            torch.tensor(-25.0, device=device, dtype=torch.float64),
        )

        drift = theta_drift * (log_s - log_f)
        noise_persistent = drift_rho * noise_persistent + agg_shocks_drift[tau]
        log_f = torch.where(
            active, log_f + drift + noise_persistent + agg_shocks_noise[tau], log_f
        )

        exp_f = torch.exp(log_f)
        a_f = exp_f / max(torch.sum(exp_f).item(), 1e-30)
        log_f_norm = torch.where(
            a_f > 1e-25,
            torch.log(torch.clamp(a_f, min=1e-30)),
            torch.tensor(-25.0, device=device, dtype=torch.float64),
        )

        if multi_hh:
            log_ah_ev = torch.log(torch.clamp(alpha_h_ev, min=1e-30))
            alpha_h_ev = torch.exp(
                log_ah_ev
                + theta_drift * (log_f_norm.unsqueeze(0) - log_ah_ev)
                + hh_shocks_drift[tau]
                + hh_shocks_noise[tau]
            )
            alpha_h_ev /= torch.clamp(
                torch.sum(alpha_h_ev, dim=1, keepdim=True), min=1e-30
            )

        P_iter = P_base_t.clone()

        for _ in range(max_price_iter):
            if multi_hh:
                C_d_iter = torch.sum(
                    (w_h_t.unsqueeze(1) * alpha_h_ev / (mu_t * P_iter.unsqueeze(0)))
                    ** (1.0 / sigma_t.unsqueeze(0)),
                    dim=0,
                )
            else:
                C_d_iter = (a_f / (mu_t * P_iter)) ** (1.0 / sigma_t)

            Z_iter = C_d_iter - C_m
            Z_iter_rel = sigma_t * Z_iter / torch.clamp(C_m, min=1e-12)
            P_iter = P_iter * (1.0 + soft_clamp(Z_iter_rel, price_step_cap))
            P_iter = torch.clamp(P_iter, min=1e-12)

            if (
                torch.max(torch.abs(Z_iter) / torch.clamp(C_m, min=1e-12)).item()
                < 0.005
            ):
                break

        P_clear = P_iter

        if multi_hh:
            C_d = torch.sum(
                (w_h_t.unsqueeze(1) * alpha_h_ev / (mu_t * P_clear.unsqueeze(0)))
                ** (1.0 / sigma_t.unsqueeze(0)),
                dim=0,
            )
            resid_Y_star = torch.sum(Y_h_m).item() if Y_h_m is not None else 1.0
        else:
            C_d = (a_f / (mu_t * P_clear)) ** (1.0 / sigma_t)
            resid_Y_star = Y_m

        reveal_nums = P_clear * (torch.clamp(C_d, min=1e-12) ** sigma_t)
        a_reveal_sum += reveal_nums / max(torch.sum(reveal_nums).item(), 1e-30)

        eps_i = 1.0 / sigma_t
        delta_p = (P_clear - P_base_t) / torch.clamp(P_base_t, min=1e-30)
        val = eps_i * delta_p

        s_max = 1.0 / (rho_M + 1.0)
        signal = -(s_max * torch.tanh(torch.clamp(val, min=0.0) / s_max))
        denom_sig = 1.0 + signal
        C_hat_sum += C_m / denom_sig

        C_monthly[tau] = C_d
        P_monthly[tau] = P_clear
        monthly_resid_Y[tau] = resid_Y_star
        monthly_drifts[tau] = torch.mean(torch.abs(P_clear / P_base_t - 1.0))

    G_hat_bare = (C_hat_sum - C_plan_t) / torch.clamp(C_plan_t, min=1e-30)
    P_final = P_monthly[-1]

    alpha_s_mask = alpha_s_t > 1e-12
    signed_drift = torch.mean((P_final / P_base_t - 1.0)[alpha_s_mask]).item()

    abs_drifts_v = torch.abs(P_final / P_base_t - 1.0)[alpha_s_mask]
    if len(abs_drifts_v) == 0:
        abs_drift = 0.0
    else:
        abs_drift = (len(abs_drifts_v) / torch.sum(1.0 / (abs_drifts_v + 1e-12))).item()

    return {
        "C_monthly": _to_numpy(C_monthly),
        "C_hat": _to_numpy(C_hat_sum),
        "G_hat_bare": _to_numpy(G_hat_bare),
        "P_monthly": _to_numpy(P_monthly),
        "P_final": _to_numpy(P_final),
        "price_drift": abs_drift,
        "signed_drift": signed_drift,
        "monthly_drifts": _to_numpy(monthly_drifts),
        "monthly_resid_Y": _to_numpy(monthly_resid_Y),
        "alpha_true_final": _to_numpy(a_reveal_sum / n_months),
        "alpha_macro_final": _to_numpy(a_f),
        "alpha_h_final": _to_numpy(alpha_h_ev) if multi_hh else np.zeros((0, 0)),
        "rho_M": rho_M,
    }


def solve_one_firm_lp(f, v_plan, B_dense, K_firms, X_star, eps_reg=0.001):
    n = len(X_star)
    K_total_sector = np.sum(K_firms, axis=0)

    c = np.zeros(2 * n)
    c[:n] = -v_plan  # Maximize sum(v*x) -> Minimize -sum(v*x)
    c[n:] = eps_reg  # - eps_reg * sum(u) -> + eps_reg * sum(u)

    A_ub = []
    b_ub = []

    share = np.where(
        K_total_sector > 1e-12, K_firms[f, :] / K_total_sector, 1.0 / K_firms.shape[0]
    )
    target = share * X_star

    # x_i - u_i <= target_i
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -1
        A_ub.append(row)
        b_ub.append(target[i])

    # -x_i - u_i <= -target_i
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -1
        row[n + i] = -1
        A_ub.append(row)
        b_ub.append(-target[i])

    # Capacity constraints: sum(B[i,j]*x_j) <= K_firms[f, i]
    for i in range(n):
        if np.any(B_dense[i, :] > 1e-12):
            row = np.zeros(2 * n)
            row[:n] = B_dense[i, :]
            A_ub.append(row)
            b_ub.append(max(K_firms[f, i], 0.0))

    bounds = [(0, None)] * (2 * n)

    # scipy.optimize.linprog with Highs
    res = linprog(
        c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs"
    )
    if res.success:
        return f, res.x[:n]
    return f, np.zeros(n)


def solve_firm_lp(v_MIP_py, B_dense_py, K_firms_py, X_star_py, tol=0.001):
    v_plan = np.maximum(np.asarray(v_MIP_py), 1e-8)
    B_dense = np.asarray(B_dense_py)
    K_firms = np.asarray(K_firms_py)
    X_star = np.asarray(X_star_py)

    n_firms = K_firms.shape[0]
    n = len(X_star)

    X_f_total = np.zeros((n, n_firms))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(solve_one_firm_lp, f, v_plan, B_dense, K_firms, X_star): f
            for f in range(n_firms)
        }
        for future in concurrent.futures.as_completed(futures):
            f, x_val = future.result()
            X_f_total[:, f] = x_val

    return X_f_total
