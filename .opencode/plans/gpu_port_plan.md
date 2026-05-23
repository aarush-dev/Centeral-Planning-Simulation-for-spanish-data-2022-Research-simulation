# Plan: Optimize and Port to GPU (PyTorch)

## 1. Goal
Optimize the Central Planning Simulation code to run efficiently on Kaggle Dual T4 GPUs and eliminate the dependency on Julia (which causes long download times on Kaggle on every run). Also, fix the `matplotlib` Kaggle backend error.

## 2. Approach
- **Drop Julia**: Completely remove `Scripts/core/julia_bridge.py` and `Scripts/core/model_core.jl`.
- **PyTorch Port (`Scripts/core/torch_core.py`)**: 
  Implement `neumann_apply`, `solve_planner` (FISTA), `fast_loop` (Tatonnement), and `compute_investment` purely in PyTorch. PyTorch runs natively on GPU (`cuda`) and comes pre-installed on Kaggle, completely avoiding dependency downloads. The FISTA dual ascent algorithm and all tensor math map perfectly to PyTorch tensors.
- **LP Solver**: The firm production LP `solve_firm_lp` used JuMP + HiGHS. We will port this to `scipy.optimize.linprog`. Since there are multiple firms, we will parallelize this step across the Kaggle CPU cores (4 cores) using `concurrent.futures.ProcessPoolExecutor` while the heavy matrix math is done by the GPU.
- **Dual T4 Utilization**: Modify `Scripts/engine/monte_carlo.py` to use `multiprocessing` to dispatch independent Monte Carlo simulation runs across `cuda:0` and `cuda:1` in parallel.
- **Matplotlib Fix**: Inject `os.environ.pop("MPLBACKEND", None)` before any matplotlib imports to prevent the backend crash on Kaggle.

## 3. Required File Modifications (When Execution Mode is Enabled)

### A. Fix Kaggle Matplotlib Bug
In `Scripts/main.py` and `Scripts/engine/monte_carlo.py`, at the very top:
```python
import os
os.environ.pop("MPLBACKEND", None) # Fix Kaggle matplotlib error
import matplotlib
matplotlib.use("Agg")
```

### B. Replace Julia Bridge with PyTorch
**Create `Scripts/core/torch_core.py`**
This file will use `torch` for all mathematical arrays.
```python
import torch
import numpy as np
from scipy.optimize import linprog
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def neumann_apply(A, v, k=20):
    res = v.clone()
    term = v.clone()
    for _ in range(k):
        term = torch.matmul(A, term)
        res += term
    return res

# ... Complete PyTorch implementations of solve_planner, fast_loop, compute_investment ...
# solve_firm_lp will map to scipy.optimize.linprog and run in parallel on CPU.
```

### C. Update Simulation to use PyTorch
In `Scripts/engine/simulation.py`, change the import:
```python
from core.torch_core import (
    compute_investment, solve_planner,
    fast_loop, solve_firm_lp
)
```
Convert `numpy` arrays to `torch.Tensor` mapped to `device` before passing them to the solver functions.

### D. Multi-GPU Monte Carlo Execution
In `Scripts/engine/monte_carlo.py`:
Update `run_ensemble` to utilize `concurrent.futures.ProcessPoolExecutor` with max workers equal to the number of GPUs available or CPU cores, distributing `state.device = "cuda:0"` and `"cuda:1"` alternately among the runs.

## 4. Next Steps
Since the environment is currently restricted to **Plan Mode** (file edits are blocked), this plan outlines exactly what needs to be done. Please switch to **Execution Mode** to allow me to write `torch_core.py` and apply the necessary refactorings to the codebase.
