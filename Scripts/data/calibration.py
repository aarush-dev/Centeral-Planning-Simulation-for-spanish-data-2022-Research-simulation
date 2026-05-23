import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs as sp_eigs
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

from core.torch_core import neumann_apply

_B_MATRIX_CACHE = None


def _get_structural_b_matrix(
    n: int, sector_names: List[str], data: dict
) -> sp.csr_matrix:
    """Retrieve or generate the global structural B-matrix (stochastic ensemble consistency)."""
    global _B_MATRIX_CACHE
    if _B_MATRIX_CACHE is not None:
        return _B_MATRIX_CACHE.copy()

    diag_kappa = (
        np.asarray(data["kappa"], dtype=float)
        if "kappa" in data
        else _kappa(sector_names)
    )

    B_diag = sp.diags(diag_kappa, format="csr")

    # Generate structural noise (seeds consistent matrix across Monte Carlo ensemble)
    rng = np.random.default_rng(42)
    noise_density = 0.05
    n_noise = int(n * n * noise_density)
    rows = rng.integers(0, n, size=n_noise)
    cols = rng.integers(0, n, size=n_noise)
    vals = rng.uniform(0.15, 0.2, size=n_noise)

    B_noise = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    B = (B_diag + B_noise).tolil()
    B.setdiag(diag_kappa)  # Ensure diagonal strictly follows the kappa lookup

    B = B.tocsr()
    B.eliminate_zeros()

    _B_MATRIX_CACHE = B.copy()
    return _B_MATRIX_CACHE.copy()


def _kappa(sector_names: List[str]) -> np.ndarray:
    """Return sector capital-intensity coefficients (diagonal of B, annual units)."""
    k = np.full(len(sector_names), 0.50)

    # Mapping of sector substrings to structural kappa values
    mappings = [
        (["real estate", "imputed rent"], 3.10),
        (["electricity", "gas", "steam"], 2.17),
        (["mining", "quarrying"], 1.86),
        (["petroleum", "coke"], 1.55),
        (["water supply", "natural water"], 1.24),
        (["sewerage", "waste", "agricultur", "forestr", "fish"], 0.93),
        (["chemical", "pharmaceutical", "motor vehicle"], 0.74),
        (
            [
                "food",
                "textile",
                "wood",
                "paper",
                "rubber",
                "plastic",
                "fabricated metal",
                "electrical",
                "machinery",
                "basic metal",
            ],
            0.62,
        ),
        (["construct", "financial", "insurance"], 0.31),
        (["telecommunication", "computer programming"], 0.37),
    ]

    for i, name in enumerate(sector_names):
        nm = name.lower()
        for keywords, value in mappings:
            if any(key in nm for key in keywords):
                k[i] = value
                break
    return k


def _v_per_unit(sector_names: List[str]) -> np.ndarray:
    """Return value-added per physical unit of output by sector.

    I use these structural parameters (analogous to kappa) to recover physical
    output X_real from observed value-added V as X_real = V / v_per_unit.
    """
    v = np.ones(len(sector_names)) * 0.60
    for i, name in enumerate(sector_names):
        nm = name.lower()
        if any(x in nm for x in ["agricultur", "forestr", "fish"]):
            v[i] = 0.50
        elif any(x in nm for x in ["mining", "quarrying"]):
            v[i] = 0.45
        elif "petroleum" in nm or "coke" in nm:
            v[i] = 0.25
        elif any(x in nm for x in ["electricity", "gas", "steam"]):
            v[i] = 0.40
        elif any(x in nm for x in ["water supply", "natural water"]):
            v[i] = 0.65
        elif any(x in nm for x in ["sewerage", "waste"]):
            v[i] = 0.70
        elif any(x in nm for x in ["food", "beverage", "tobacco"]):
            v[i] = 0.30
        elif any(x in nm for x in ["textile", "leather", "apparel"]):
            v[i] = 0.50
        elif any(x in nm for x in ["wood", "paper", "printing"]):
            v[i] = 0.50
        elif any(x in nm for x in ["chemical", "pharmaceutical"]):
            v[i] = 0.60
        elif any(x in nm for x in ["rubber", "plastic"]):
            v[i] = 0.50
        elif any(x in nm for x in ["basic metal", "fabricated metal"]):
            v[i] = 0.45
        elif any(x in nm for x in ["computer, elec", "electrical equip"]):
            v[i] = 0.50
        elif any(x in nm for x in ["machinery", "motor vehicle", "transport equip"]):
            v[i] = 0.45
        elif any(x in nm for x in ["furniture", "repair and install", "mineral"]):
            v[i] = 0.60
        elif "construct" in nm:
            v[i] = 0.65
        elif any(x in nm for x in ["wholesale", "retail", "trade"]):
            v[i] = 0.70
        elif any(
            x in nm
            for x in [
                "land transport",
                "water transport",
                "air transport",
                "warehousing",
                "postal",
            ]
        ):
            v[i] = 0.65
        elif "accommodation" in nm:
            v[i] = 0.75
        elif any(x in nm for x in ["financial", "insurance", "auxiliary to financial"]):
            v[i] = 0.85
        elif any(x in nm for x in ["real estate", "imputed rent"]):
            v[i] = 0.90
        elif any(x in nm for x in ["telecommunication"]):
            v[i] = 0.75
        elif any(
            x in nm for x in ["computer programming", "publishing", "motion picture"]
        ):
            v[i] = 0.80
        elif any(
            x in nm
            for x in [
                "legal",
                "architectural",
                "scientific research",
                "advertising",
                "professional",
            ]
        ):
            v[i] = 0.85
        elif any(
            x in nm
            for x in [
                "rental and leas",
                "employment",
                "travel agency",
                "security",
                "services auxiliary",
            ]
        ):
            v[i] = 0.75
        elif any(x in nm for x in ["public admin", "defence", "compulsory social"]):
            v[i] = 0.90
        elif any(x in nm for x in ["education"]):
            v[i] = 0.90
        elif any(x in nm for x in ["health", "social work"]):
            v[i] = 0.85
        elif any(x in nm for x in ["arts", "entertainment", "recreation", "sport"]):
            v[i] = 0.75
        elif any(
            x in nm
            for x in ["membership organisation", "household employ", "extraterritorial"]
        ):
            v[i] = 0.90
    return v


@dataclass
class ModelState:
    """Complete simulation state passed between quarters.

    Capital and all dynamic variables live at the firm level (K_firms).
    Aggregate K is an accounting identity: K = K_firms.sum(axis=0).
    """

    # Static structure
    n: int
    A: object  # scipy.sparse.csr_matrix
    A_bar: object  # A + delta*B
    B: object  # capital-coefficient matrix
    l_tilde: np.ndarray  # (I - A_bar)^{-T} @ l
    v_per_unit: np.ndarray  # value-added per physical unit (EUR/unit)
    l_vec: np.ndarray
    L_total: float
    delta: float
    max_iter: int
    inflation_target: float
    pref_drift_rho: float
    pref_drift_sigma: float
    pref_noise_sigma: float
    theta_drift: float
    price_tol: float
    max_price_iter: int
    epsilon: float
    wage_rate: float
    sigma_vec: np.ndarray
    neumann_k: int
    primal_tol: float
    dual_tol: float
    eta_K: float
    eta_L: float
    habit_persistence: float
    sector_names: List[str]
    sector_short: List[str]
    cybernetic_k_sigma: float
    # Dynamic state
    t: int
    K: np.ndarray  # (N,) accounting aggregate = K_firms.sum(axis=0)
    K_firms: np.ndarray  # (5, N) firm-level capital stocks
    G: np.ndarray = field(default_factory=lambda: np.array([]))
    g_step: float = 0.0
    c_step: float = 0.015
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_true: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_macro: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_slow: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_habit: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_bar: np.ndarray = field(default_factory=lambda: np.array([]))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_prev: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_planner: float = 1.0
    C: np.ndarray = field(default_factory=lambda: np.array([]))
    X: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: float = 0.0
    income_scale: float = 1.0
    C_monthly: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    P_monthly: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    rng: object = field(default_factory=lambda: np.random.default_rng())
    G_hat_init: np.ndarray = field(default_factory=lambda: np.array([]))
    G_hat_raw_prev: np.ndarray = field(default_factory=lambda: np.array([]))
    dK_0: np.ndarray = field(default_factory=lambda: np.array([]))
    history: list = field(default_factory=list)
    slim_history: bool = False
    P_0: np.ndarray = field(default_factory=lambda: np.array([]))
    VAL_0: float = 1.0
    Y_0_init: float = 0.0
    VAL_0_Laspeyres: float = 1.0
    pi_0_fixed: np.ndarray = field(default_factory=lambda: np.array([]))
    dual_weight_0: np.ndarray = field(default_factory=lambda: np.array([]))
    C_0_init: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_K_0: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_L_0: float = 0.0
    CPI_chained: float = 1.0
    # Multi-household demand side
    n_households: int = 4
    w_h: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_households,)
    W_ownership: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (n_households, n_firms)
    alpha_h: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (n_households, n_sectors)
    alpha_true_h: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (n_households, n_sectors)
    alpha_slow_h: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (n_households, n_sectors)
    alpha_habit_h: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # (n_households, n_sectors)
    n_firms: int = 5

    @property
    def v(self) -> np.ndarray:
        return self.v_per_unit


def calibrate(
    data: dict,
    delta: float = 0.015,
    pref_drift_rho: float = 0.95,
    pref_drift_sigma: float = 0.04,
    pref_noise_sigma: float = 0.01,
    theta_drift: float = 0.1,
    epsilon: float = 0.5,
    neumann_k: int = 25,
    kappa_factor: float = 1.0,
    L_total: float = 9.75e9,
    wage_rate: float = 21.0,
    labor_mult: float = 1.0,
    primal_tol: float = 1e-3,
    dual_tol: float = 1e-4,
    cybernetic_k_sigma: float = 1.0,
    eta_K: float = 0.40,
    eta_L: float = 0.50,
    max_iter: int = 2000,
    g_step: float = 0.0,
    c_step: float = 0.01,
    nominal_consumption_annual: float = 807e9,
    inflation_target: float = 0.0,
    habit_persistence: float = 0.7,
    slim_history: bool = None,
    n_households: int = 4,
    hh_dispersion: float = 0.05,
    price_tol: float = 0.005,
    max_price_iter: int = 25,
    n_firms: int = 5,
    firm_shares: np.ndarray = None,
    phi_matrix: np.ndarray = None,
    sigma_val: float = 1.0,
) -> ModelState:
    """
    Build a fully calibrated ModelState from the IO data dict.

    This includes:
    1. Recovering physical units using sector-specific capital intensity (kappa).
    2. Generating firm-level heterogeneity and multi-household preference distributions.
    3. Initializing aggregate preferences via the CES transition formula.
    4. Anchoring market prices and marginal utility of income (mu_0).
    """
    A_in = data["A"]
    V = np.asarray(data["V_total"], dtype=float).copy()
    C_hh = np.asarray(data["C"], dtype=float).copy()
    X_data = np.asarray(data["X"], dtype=float).copy()
    sector_names = list(data["sector_names"])
    sector_short = list(data["sector_short"])
    n = len(sector_names)

    if sp.issparse(A_in):
        A = A_in.tocsr().astype(float)
    else:
        A = sp.csr_matrix(np.asarray(A_in, dtype=float))
        sparsity = 1.0 - A.nnz / n**2
        if n > 1000:
            logger.info(
                f"[calibration] Dense A converted to CSR  "
                f"(sparsity={sparsity * 100:.1f}%, nnz={A.nnz:,})"
            )

    _dead_threshold = 1e6
    dead = np.where((X_data < _dead_threshold) & (V < _dead_threshold))[0]
    if len(dead):
        logger.info(f"[calibration] Zeroing {len(dead)} dead sector(s)")
        A = A.tolil()
        for i in dead:
            A[i, :] = 0
            A[:, i] = 0
            X_data[i] = 0
            V[i] = 0
            C_hh[i] = 0
        A = A.tocsr()
        A.eliminate_zeros()

    v_per_unit = (
        np.asarray(data["v_per_unit"], dtype=float)
        if "v_per_unit" in data
        else _v_per_unit(sector_names)
    )

    X_real = V / np.maximum(v_per_unit, 1e-12)

    # Get structural B-matrix (cached for Monte-Carlo consistency)
    B = _get_structural_b_matrix(n, sector_names, data)
    B = B * kappa_factor

    A_bar = A + delta * B
    A_bar.eliminate_zeros()

    try:
        eigvals_top = sp_eigs(
            A_bar, k=1, which="LM", return_eigenvectors=False, maxiter=10 * n, tol=1e-4
        )
        rho = float(np.abs(eigvals_top).max())
        if rho < 1:
            logger.info(f"[calibration] rho(A_bar) = {rho:.4f}  (OK)")
        else:
            logger.warning(
                f"[calibration] rho(A_bar) = {rho:.4f}  (WARNING) -- Neumann will diverge!"
            )
    except Exception:
        row_sums = np.asarray(np.abs(A_bar).sum(axis=1)).ravel()
        rho = row_sums.max()
        level = logger.info if rho < 1 else logger.warning
        level(f"[calibration] Gershgorin rho(A_bar) <= {rho:.4f}")

    # Cost-push prices from value-added via Leontief inverse
    P_real = np.asarray(
        neumann_apply(
            A.T.toarray() if hasattr(A.T, "toarray") else np.asarray(A.T),
            np.asarray(v_per_unit, dtype=np.float64),
            neumann_k,
        )
    )

    Y_0 = nominal_consumption_annual / 4.0

    C_0 = C_hh / np.where(P_real > 1e-30, P_real, 1e-30)
    C_0[dead] = 0

    G_raw_cal = np.asarray(data.get("G_raw", np.zeros(n)), dtype=float)
    G_0 = G_raw_cal / np.where(P_real > 1e-30, P_real, 1e-30)

    g_init = 0.0075
    v1 = g_init * C_0 + g_step * G_0
    v2 = (g_init**2) * C_0 + (g_step**2) * G_0
    term1 = B @ np.asarray(
        neumann_apply(
            A_bar.toarray() if hasattr(A_bar, "toarray") else np.asarray(A_bar),
            np.asarray(v1, dtype=np.float64),
            neumann_k,
        )
    )
    inner_v2 = np.asarray(
        neumann_apply(
            A_bar.toarray() if hasattr(A_bar, "toarray") else np.asarray(A_bar),
            np.asarray(v2, dtype=np.float64),
            neumann_k,
        )
    )
    term2 = B @ np.asarray(
        neumann_apply(
            A_bar.toarray() if hasattr(A_bar, "toarray") else np.asarray(A_bar),
            np.asarray(B @ inner_v2, dtype=np.float64),
            neumann_k,
        )
    )
    dK_0 = term1 + term2

    # K_0 is the calibration anchor; firm capitals are derived from it with a 0.5% slack buffer
    K_0 = B @ np.asarray(
        neumann_apply(
            A_bar.toarray() if hasattr(A_bar, "toarray") else np.asarray(A_bar),
            np.asarray(C_0 + G_0 + dK_0, dtype=np.float64),
            neumann_k,
        )
    )

    exp = np.where(P_real * C_0 > 0, P_real * C_0, 0.0)
    alpha_0 = exp / max(exp.sum(), 1e-10)

    P_0 = np.divide(Y_0 * alpha_0, C_0, out=np.zeros_like(C_0), where=C_0 > 1e-12)

    logger.info(
        f"[calibration] Initial Prices anchored "
        f"(implied price level: {(P_0.mean() / max(P_real.mean(), 1e-30)):.4f})"
    )

    l_vec = (v_per_unit.copy() / wage_rate) * labor_mult
    l_tilde = np.asarray(
        neumann_apply(
            A_bar.T.toarray() if hasattr(A_bar.T, "toarray") else np.asarray(A_bar.T),
            np.asarray(l_vec, dtype=np.float64),
            neumann_k,
        )
    )

    G_hat_init = np.where(C_0 > 0, g_init, 0.0)

    logger.info(f"[calibration] n={n:,}  nnz(B)={B.nnz:,}  nnz(A_bar)={A_bar.nnz:,}")
    logger.info(
        f"[calibration] Y0={Y_0 / 1e9:.1f}B EUR  "
        f"K0={K_0.sum() / 1e9:.1f}B units  "
        f"C0={C_0.sum() / 1e9:.1f}B units/quarter"
    )

    # Split K_0 across firms with near-equal stochastic shares
    B_dense = B.toarray()
    if firm_shares is not None:
        shares = firm_shares
    else:
        shares = np.abs(np.random.normal(1.0 / n_firms, 0.02, n_firms))
        shares /= shares.sum()
    # Split K_0 across firms using partitioned final demand (Vectorized)
    demand_total = C_0 + G_0 + dK_0

    # Construct a Target Matrix (n, n_firms)
    F_mat = demand_total[:, None] * shares[None, :]

    # One batch-call to Julia handles all firms
    X_firm = np.asarray(
        neumann_apply(
            A_bar.toarray() if hasattr(A_bar, "toarray") else np.asarray(A_bar),
            F_mat,
            neumann_k,
        )
    )

    # Derive K_firms accordingly
    K_firms = (B_dense @ X_firm).T

    state = ModelState(
        n=n,
        A=A,
        A_bar=A_bar,
        B=B,
        l_tilde=l_tilde,
        v_per_unit=v_per_unit,
        l_vec=l_vec,
        L_total=L_total,
        delta=delta,
        pref_drift_rho=pref_drift_rho,
        pref_drift_sigma=pref_drift_sigma,
        pref_noise_sigma=pref_noise_sigma,
        theta_drift=theta_drift,
        price_tol=price_tol,
        max_price_iter=max_price_iter,
        epsilon=epsilon,
        wage_rate=wage_rate,
        neumann_k=neumann_k,
        primal_tol=primal_tol,
        dual_tol=dual_tol,
        cybernetic_k_sigma=cybernetic_k_sigma,
        eta_K=eta_K,
        eta_L=eta_L,
        max_iter=max_iter,
        inflation_target=inflation_target,
        habit_persistence=habit_persistence,
        n_firms=n_firms,
        sector_names=sector_names,
        sector_short=sector_short,
        t=0,
        K=K_0.copy(),
        K_firms=K_firms,
        G=G_0.copy(),
        g_step=float(g_step),
        c_step=float(c_step),
        alpha=alpha_0.copy(),
        alpha_true=alpha_0.copy(),
        alpha_macro=alpha_0.copy(),
        alpha_slow=alpha_0.copy(),
        alpha_habit=alpha_0.copy(),
        alpha_bar=alpha_0.copy(),
        P=P_0.copy(),
        C=C_0.copy(),
        X=X_real.copy(),
        Y=Y_0,
        C_monthly=np.tile(C_0 / 3, (3, 1)),
        P_monthly=np.tile(P_0, (3, 1)),
        G_hat_init=G_hat_init,
        G_hat_raw_prev=G_hat_init.copy(),
        dK_0=dK_0.copy(),
        sigma_vec=np.full(n, sigma_val),
    )

    if slim_history is None:
        slim_history = n > 5000
        if slim_history:
            logger.info("[calibration] Auto-enabled slim_history for n > 5000")
    state.slim_history = slim_history

    pi_0 = alpha_0 / (np.maximum(C_0, 1e-12) ** state.sigma_vec)

    state.P_0 = P_0.copy()
    state.pi_prev = pi_0.copy()
    state.mu_planner = float(np.dot(pi_0, C_0) / max(Y_0, 1e-30))
    state.C_0_init = C_0.copy()
    state.VAL_0_Laspeyres = 1.0
    state.pi_0_fixed = np.array([])
    state.dual_weight_0 = np.array([])
    state.Y_0_init = float(Y_0)
    state.B = B.copy()

    # --- Multi-household initialization ------------------------------------------
    state.n_households = n_households
    state.n_firms = n_firms

    # Ownership matrix W: (n_households, n_firms)
    # Columns sum to 1: each column is the distribution of one firm's income
    rng_hh = np.random.default_rng(123)
    if phi_matrix is not None:
        state.W_ownership = phi_matrix.copy()
    else:
        # dirichlet(size=n_firms) gives (n_firms, n_households) where rows sum to 1
        # Transposing gives (n_households, n_firms) where COLUMNS sum to 1
        state.W_ownership = rng_hh.dirichlet(np.ones(n_households), size=n_firms).T

    # Per-household subsistence: removed

    # Per-household preferences: initialise with log-normal dispersion around
    # aggregate alpha_0.  σ = 0.15 gives meaningful heterogeneity for the
    # LN-AR drift to amplify over time.
    alpha_h = np.zeros((n_households, n))
    for h in range(n_households):
        noise = rng_hh.normal(0.0, hh_dispersion, n)
        a_h = alpha_0 * np.exp(noise)
        a_h = np.maximum(a_h, 1e-30)
        a_h /= a_h.sum()
        alpha_h[h, :] = a_h
    # Initialize households with Dirichlet ownership and preferences
    state.alpha_h = alpha_h

    # Synchronize welfare weights (w_h) with the initial income distribution
    if phi_matrix is not None and firm_shares is not None:
        hh_income_raw = phi_matrix @ firm_shares
        state.w_h = hh_income_raw / max(hh_income_raw.sum(), 1e-30)
    else:
        state.w_h = np.ones(n_households) / n_households  # Default: uniform
    state.alpha_true_h = alpha_h.copy()
    state.alpha_slow_h = alpha_h.copy()
    state.alpha_habit_h = alpha_h.copy()

    # Initial Aggregate Alpha: CES aggregate per Eq. 57
    # alpha_agg_i = [sum_h (w_h * alpha_hi)^(1/sigma_i)]^sigma_i
    def get_alpha_agg(weights):
        a_agg = np.zeros(n)
        for i in range(n):
            inner_sum = np.sum((weights * alpha_h[:, i]) ** (1.0 / state.sigma_vec[i]))
            a_agg[i] = inner_sum ** state.sigma_vec[i]
        return a_agg

    # Scale weights so that alpha_agg sums to 1
    current_agg = get_alpha_agg(state.w_h)
    S = current_agg.sum()
    # Note: alpha_agg is proportional to weights.
    # alpha_agg_i(k*w) = [sum(k*w*a)^(1/sig)]^sig = k * alpha_agg_i(w)
    state.w_h /= S

    state.alpha = get_alpha_agg(state.w_h)  # Now sums to 1

    state.price_tol = price_tol
    state.max_price_iter = max_price_iter

    # Sync planner's starting belief with the actual mean of the perturbed households
    # to eliminate the initialization bias (the 'Expectation Gap').
    state.alpha_bar = alpha_h.mean(axis=0)
    state.alpha_bar /= state.alpha_bar.sum()
    state.alpha_slow = state.alpha_bar.copy()
    state.alpha_habit = state.alpha_bar.copy()

    return state
