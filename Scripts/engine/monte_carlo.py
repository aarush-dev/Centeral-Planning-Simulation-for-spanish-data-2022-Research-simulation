"""
monte_carlo.py
--------------
Enhanced Monte-Carlo ensemble for the Spanish Central Planning Model.
Includes diagnostic plots for Alpha Gap, Price Drift, and Iterations.
"""

import os
# Mitigate segfaults by handling Julia/Python signal collisions
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import json
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent thread-safety crashes
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simulation components
from data.data_loader import load_data
from data.calibration import calibrate
from engine.simulation import run_simulation, SimulationError

logging.basicConfig(level=logging.WARNING, format="%(message)s")

class EnsembleConfig:
    def __init__(self, config_path="Data/config.json"):
        self.config = {}
        paths = [Path(config_path), Path("Data/config.json")]
        for p in paths:
            if p.exists():
                with open(p, "r") as f:
                    self.config = json.load(f)
                break
        
        self.defaults = {
            "n_quarters": 20,
            "n_runs": 1000,
            "delta": 0.01,
            "pref_drift_rho": 0.9,
            "pref_drift_sigma": 0.02,
            "pref_noise_sigma": 0.05,
            "theta_drift": 0.03,
            "epsilon": 0.5,
            "kappa_factor": 4.0,
            "L_total": 39e9,
            "labor_mult": 1.0,
            "neumann_k": 25,
            "wage_rate": 16.9,
            "g_step": 0.01,
            "c_step": 0.01,
            "habit_persistence": 0.9,
            "n_households": 1000,
            "n_firms": 250,
            "hh_dispersion": 0.05,
            "nominal_consumption_annual": 807e9,
            "inflation_target": 0.0,
            "price_tol": 0.0,
            "max_price_iter": 8,
            "primal_tol": 0.001,
            "dual_tol": 1e-5,
            "eta_K": 0.2,
            "eta_L": 0.15,
            "max_iter": 2000,
            "cybernetic_k_sigma": 1.0,
            "sigma_val": 1.0
        }

    def get(self, key):
        val = self.config.get(key, self.defaults.get(key))
        return val

class TrajectoryCollector:
    def __init__(self, n_runs, n_q):
        self.n_runs = n_runs
        self.n_q = n_q
        self.success_count = 0
        self.gdp_level = np.full((n_runs, n_q), np.nan)
        self.inflation = np.full((n_runs, n_q), np.nan)
        self.cap_slack = np.full((n_runs, n_q), np.nan)
        self.alpha_gap = np.full((n_runs, n_q), np.nan)
        self.price_drift = np.full((n_runs, n_q), np.nan)
        self.mvps = [] # Flat list for histogram

    def add_run(self, idx, history):
        if not isinstance(history, list): return
        self.success_count += 1
        gdp_q1 = history[0]["GDP"]
        for q, h in enumerate(history):
            self.inflation[idx, q]  = h.get("Inflation", 0.0)
            self.cap_slack[idx, q]   = (h.get("slack_val_Q1", 0.0) / max(h.get("K_val_Q1", 1.0), 1e-12)) * 100.0
            self.gdp_level[idx, q]  = (h["GDP"] / gdp_q1) * 100.0
            self.alpha_gap[idx, q]  = h.get("alpha_gap", 0.0) * 100.0 # to %
            self.price_drift[idx, q] = h.get("price_drift", 0.0) * 100.0 # to %
            self.mvps.append(h.get("mvps", h.get("iterations", 0) * 2))

class ProfessionalPlotter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_fan(self, data, title, ylabel, filename):
        valid = data[~np.isnan(data).any(axis=1)]
        if valid.size == 0: return
        x = np.arange(1, valid.shape[1] + 1)
        pcts = [np.percentile(valid, p, axis=0) for p in [5, 15, 25, 75, 85, 95]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(x, pcts[0], pcts[5], color="#d5e2f0", alpha=0.9, label='90% CI')
        ax.fill_between(x, pcts[1], pcts[4], color="#85a4cd", alpha=0.9, label='70% CI')
        ax.fill_between(x, pcts[2], pcts[3], color="#0b3060", alpha=0.9, label='50% CI')
        
        ax.set_title(f"Monte Carlo: {title}", fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Quarter")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(self.out_dir / f"fan_{filename}.png", dpi=300)
        plt.savefig(self.out_dir / f"fan_{filename}.pdf")
        plt.close()

    def plot_hist(self, data, title, xlabel, filename):
        if not data: return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=30, color="#0b3060", alpha=0.7, edgecolor='white')
        ax.set_title(f"Monte Carlo: {title}", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency (Quarters)")
        plt.tight_layout()
        plt.savefig(self.out_dir / f"hist_{filename}.png", dpi=300)
        plt.savefig(self.out_dir / f"hist_{filename}.pdf")
        plt.close()

def run_ensemble():
    config_handler = EnsembleConfig()
    n_runs = config_handler.get("n_runs")
    n_q    = config_handler.get("n_quarters")
    data   = load_data()
    
    import inspect
    sig = inspect.signature(calibrate)
    config_params = {k: config_handler.get(k) for k in sig.parameters.keys() if k != 'data'}

    collector = TrajectoryCollector(n_runs, n_q)
    out_dir = Path(__file__).parents[2] / "Results" / "MonteCarlo" / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)    
    print(f"Starting Sequential Monte-Carlo: {n_runs} runs.")
    print(f"Parameters: { {k:v for k,v in config_params.items() if v is not None} }")
    print(f"Diagnostics: GDP, Inflation, Slack, Alpha Gap, Price Drift, Iterations.")
    print(f"Results will save periodically to {out_dir}\n")
    
    start_time = time.time()
    for i in range(n_runs):
        run_start = time.time()
        try:
            state = calibrate(data, **config_params)
            state.rng = np.random.default_rng(1000 + i)
            state.slim_history = True
            
            state = run_simulation(state, n_quarters=n_q)
            collector.add_run(i, state.history)
            
            elapsed = time.time() - run_start
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Run {i+1}/{n_runs} complete ({elapsed:.1f}s/run). Total elapsed: {time.time()-start_time:.0f}s")
            
            if (i + 1) % 25 == 0:
                print(f"  Updating intermediate plots...")
                plotter = ProfessionalPlotter(out_dir)
                plotter.plot_fan(collector.gdp_level, "Real GDP Level", "Index (Q1=100)", "gdp_level")
                plotter.plot_fan(collector.inflation, "Geomean Inflation", "Rate", "inflation")
                plotter.plot_fan(collector.cap_slack, "Capital Capacity Slack", "Unused (%)", "capital_slack")
                plotter.plot_fan(collector.alpha_gap, "Alpha Tracking Error (Gap)", "Error (%)", "alpha_gap")
                plotter.plot_fan(collector.price_drift, "Price Drift (RMS)", "Drift (%)", "price_drift")
                plotter.plot_hist(collector.mvps, "Computational Effort", "Matrix-Vector Operations", "mvps")

        except Exception as e:
            print(f"  Run {i+1} failed: {e}")

    plotter = ProfessionalPlotter(out_dir)
    plotter.plot_fan(collector.gdp_level, "Real GDP Level", "Index (Q1=100)", "gdp_level")
    plotter.plot_fan(collector.inflation, "Geomean Inflation", "Rate", "inflation")
    plotter.plot_fan(collector.cap_slack, "Capital Capacity Slack", "Unused (%)", "capital_slack")
    plotter.plot_fan(collector.alpha_gap, "Alpha Tracking Error (Gap)", "Error (%)", "alpha_gap")
    plotter.plot_fan(collector.price_drift, "Price Drift (RMS)", "Drift (%)", "price_drift")
    plotter.plot_hist(collector.mvps, "Computational Effort", "Matrix-Vector Operations", "mvps")
    
    print(f"\nMonte-Carlo complete. Success: {collector.success_count}/{n_runs}")

if __name__ == "__main__":
    run_ensemble()
