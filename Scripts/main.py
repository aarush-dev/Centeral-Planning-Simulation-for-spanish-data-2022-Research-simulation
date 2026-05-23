import os
import sys

# Fix Kaggle matplotlib error
os.environ.pop("MPLBACKEND", None)

import matplotlib

matplotlib.use("Agg")

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from data.data_loader import load_data, sector_groups, annualise
from data.calibration import calibrate
from engine.simulation import run_simulation
from analysis.plots import (
    GROUP_COLORS,
    qlabels,
    group_agg,
    savefig,
    plot_gdp,
    plot_aggregate_demand_breakdown,
    plot_output_consumption,
    plot_investment,
    plot_shadow_prices,
    plot_capital,
    plot_alpha,
    plot_alpha_gap,
    plot_capital_output_ratio,
    plot_capital_slack,
    plot_labor_utilization,
    plot_shadow_price_index,
    plot_cybernetic_signals,
    plot_real_income_index,
    plot_labor_productivity,
    plot_growth_targets,
    plot_excess_demand,
    plot_inflation,
    plot_investment_gdp_ratio,
    plot_mvps,
    plot_firm_income_distribution,
)

# --- Config ------------------------------------------------------------------
config = {
    "n_quarters": 20,
    "delta": 0.0125,
    "neumann_k": 20,
    "rng_seed": 42,
    "kappa_factor": 1.0,
    "L_total": 33e9,
    "wage_rate": 16.9,
    "primal_tol": 1e-4,
    "dual_tol": 1e-4,
    "checkpoint_every": 5,
    "eta_K": 0.15,
    "eta_L": 0.15,
    "max_iter": 2000,
    "slim_history": None,
    "cybernetic_k_sigma": 1.0,
}

DATA_DIR = Path(__file__).parent.parent / "Data"
CONFIG_PATH = DATA_DIR / "config.json"

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            config.update(json.load(f))
        logger.info(f"[main] Loaded config from {CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"[main] Failed to load config.json: {e}. Using defaults.")

N_QUARTERS = config["n_quarters"]
DELTA = config["delta"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(__file__).parent.parent / "Results" / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "run_config.json", "w") as f:
    json.dump(config, f, indent=4)


# --- Main --------------------------------------------------------------------


def main():
    data = load_data(DATA_DIR)
    groups = sector_groups(data["sector_names"])

    state = calibrate(
        data,
        delta=config.get("delta", 0.01),
        pref_drift_rho=config.get("pref_drift_rho", 0.9),
        pref_drift_sigma=config.get("pref_drift_sigma", 0.02),
        pref_noise_sigma=config.get("pref_noise_sigma", 0.05),
        theta_drift=config.get("theta_drift", 0.03),
        epsilon=config.get("epsilon", 0.5),
        neumann_k=config.get("neumann_k", 25),
        kappa_factor=config.get("kappa_factor", 4.0),
        L_total=config.get("L_total", 39e9),
        wage_rate=config.get("wage_rate", 16.9),
        labor_mult=config.get("labor_mult", 1.0),
        primal_tol=config.get("primal_tol", 1e-3),
        dual_tol=config.get("dual_tol", 1e-5),
        eta_K=config.get("eta_K", 0.2),
        eta_L=config.get("eta_L", 0.15),
        max_iter=config.get("max_iter", 2000),
        g_step=config.get("g_step", 0.01),
        c_step=config.get("c_step", 0.01),
        habit_persistence=config.get("habit_persistence", 0.9),
        nominal_consumption_annual=config.get("nominal_consumption_annual", 807e9),
        slim_history=config.get("slim_history", None),
        n_households=config.get("n_households", 1000),
        hh_dispersion=config.get("hh_dispersion", 0.05),
        max_price_iter=config.get("max_price_iter", 8),
        n_firms=config.get("n_firms", 250),
        cybernetic_k_sigma=config.get("cybernetic_k_sigma", 1.0),
    )
    if "rng_seed" in config and config["rng_seed"] is not None:
        state.rng = np.random.default_rng(config["rng_seed"])

    state = run_simulation(
        state,
        n_quarters=N_QUARTERS,
        checkpoint_dir=RESULTS_DIR,
        checkpoint_every=config.get("checkpoint_every", 5),
    )

    # GDP summary table
    logger.info("\nQuarter   GDP (B EUR ann.)  YoY Growth   QoQ Growth")
    logger.info("-" * 55)
    for h in state.history:
        t = h["t"]
        gdp = annualise(h["GDP"]) / 1e9
        yoy = (
            (gdp / (annualise(state.history[t - 5]["GDP"]) / 1e9) - 1) * 100
            if t > 4
            else None
        )
        qoq = (
            (h["GDP"] / (state.history[t - 2]["GDP"] + 1e-30) - 1) * 100
            if t > 1
            else None
        )
        logger.info(
            f"Q{t:02d}   {gdp:10.2f}  "
            f"{'  ---' if yoy is None else f'{yoy:6.2f}%'}  "
            f"{'  ---' if qoq is None else f'{qoq:6.2f}%'}"
        )

    # Plots
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    r = RESULTS_DIR
    P_0 = state.P_0

    plot_gdp(
        state.history,
        r / "01_gdp.png",
        P_initial=state.pi_0_fixed,
        A=state.A,
        real_scale_factor=state.real_scale_factor,
    )
    plot_aggregate_demand_breakdown(state.history, r / "01b_ad_breakdown.png")
    plot_output_consumption(
        state.history, groups, r / "02_output_consumption.png", P_0=P_0
    )
    plot_investment(state.history, groups, r / "03_investment.png", P_0=P_0)
    plot_shadow_prices(
        state.history, data["sector_short"], groups, r / "04_shadow_prices.png"
    )
    plot_capital(state.history, groups, r / "05_capital.png", P_0=P_0)
    plot_alpha(state.history, r / "06_alpha_learning.png")
    plot_alpha_gap(state.history, r / "07_alpha_error.png")
    plot_capital_output_ratio(state.history, r / "08_capital_output_ratio.png")
    plot_shadow_price_index(state.history, r / "09_shadow_price_index.png")
    plot_capital_slack(state.history, r / "10_capital_slack.png")
    plot_labor_utilization(state.history, r / "11_labor_utilization.png")
    plot_cybernetic_signals(state.history, r / "12_cybernetic_signals.png")
    plot_real_income_index(state.history, r / "13_real_income_index.png")
    plot_labor_productivity(state.history, r / "14_labor_productivity.png")
    plot_growth_targets(state.history, r / "15_growth_targets.png")
    plot_excess_demand(state.history, r / "16_excess_demand.png")
    plot_inflation(state.history, r / "17_inflation.png")
    plot_investment_gdp_ratio(state.history, r / "18_investment_gdp_ratio.png")
    plot_mvps(state.history, r / "19_mvps.png")
    plot_firm_income_distribution(state.history, state.n, r / "20_firm_income.png")

    logger.info("\nAll charts and analysis saved to ./Results/")
    return state


if __name__ == "__main__":
    main()
