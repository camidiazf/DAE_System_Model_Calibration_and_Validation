# Codebase Guide

This document explains how this repository is organized, what the minimum pieces are to run it, and how the calibration and validation workflow moves through the code.

## What This Project Does

This repository is a Python framework for:

- simulating biological systems written as differential-algebraic equations (DAEs),
- calibrating model parameters against experimental data with particle swarm optimization (PSO),
- validating the calibrated model against a separate dataset,
- computing local sensitivities, Fisher-information-based statistics, and parameter correlations,
- saving figures and CSV summaries for each calibration scenario.

The current project includes three model variants:

- `M_smegmatis`
- `M_smegmatis_O2`
- `M_smegmatis_ammonia_def`

All three use the same shared analysis package in `DAE_analysis_package/`, and each model folder provides its own equations, parameters, and Excel data.

## Minimum Needed To Run It

At minimum, a runnable setup needs:

1. Python with the packages used by the repository.
2. One model folder containing:
   - `System_data.py`
   - `DAE_Systems_Simulations.py`
   - `Experimental_data.xlsx`
3. The shared package folder:
   - `DAE_analysis_package/`
4. A script or notebook that imports the selected model and calls the runner.

## Minimum Python Dependencies

The code imports these libraries directly:

- `casadi`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `mealpy`

The checked-in `requirements.txt` is not a normal pip requirements file. It lists module names, including standard-library modules like `os`, `sys`, `time`, and `copy`, which should not be installed with pip. For a fresh environment, the practical install command is closer to:

```bash
pip install casadi numpy pandas scipy matplotlib seaborn statsmodels scikit-learn mealpy openpyxl
```

`openpyxl` is also needed because `pandas.read_excel(...)` is used to load the experimental workbook.

## Recommended Way To Run

The intended entry point is a notebook in the repository root, especially `Main.ipynb`.

The execution pattern is:

1. Import `RUN_COMBOS` from `DAE_analysis_package.RUN_functions`.
2. Import `simulate_model` and `system_data` from the model folder you want to use.
3. Define the calibration bounds for the parameters you want to optimize.
4. Call `RUN_COMBOS(...)`.

Example using the current package API:

```python
from DAE_analysis_package.RUN_functions import RUN_COMBOS
from M_smegmatis_O2.DAE_Systems_Simulations import simulate_model
from M_smegmatis_O2.System_data import system_data

parameters_and_bounds = {
    "YX_N": [10.4, 10.6],
    "Xmax": [1.15, 1.17],
}

results = RUN_COMBOS(
    system_data=system_data,
    simulate_model=simulate_model,
    folder_model="M_smegmatis_O2",
    folder_results="Calibration_3",
    solve_calibrate="calibrate",
    iterations=30,
    parameters_and_bounds=parameters_and_bounds,
    min_k=1,
    max_k=2,
)
```

If you only want to run the original model without calibration, use:

```python
results = RUN_COMBOS(
    system_data=system_data,
    simulate_model=simulate_model,
    folder_model="M_smegmatis_O2",
    folder_results="Initial_Check",
    solve_calibrate="solve",
)
```

## Important Current Caveat

`Main.ipynb` still imports `RUN_ALL_COMBOS`, but the current package defines `RUN_COMBOS`.

That means the notebook, as committed right now, is partly outdated and will raise an import error unless that call is updated.

There are also old argument names in the notebook such as:

- `folder_calibration_results`
- `updated_parameters`

Those arguments are not accepted by the current `RUN_COMBOS(...)` function in `DAE_analysis_package/RUN_functions.py`.

So the source of truth for current behavior is the Python package, not the notebook text.

## High-Level Architecture

The repository is split into two layers:

### 1. Shared framework layer

Folder: `DAE_analysis_package/`

This contains the generic workflow and analysis utilities that can be reused across multiple models.

### 2. Model-specific layer

Folders:

- `M_smegmatis/`
- `M_smegmatis_O2/`
- `M_smegmatis_ammonia_def/`

Each model folder defines:

- the system equations,
- the initial conditions,
- the experimental datasets,
- the parameter list and constants.

The framework is designed so that changing the model usually means changing only these model-specific files, not the shared package.

## Folder-by-Folder Explanation

## `DAE_analysis_package/`

This is the main engine of the project.

### `RUN_functions.py`

This is the orchestration layer.

Main responsibilities:

- builds the output folder and CSV path,
- generates combinations of parameters to calibrate,
- runs the original model first,
- loops through each parameter combination,
- calls PSO calibration when requested,
- calls validation and parameter analysis,
- appends results to CSV,
- generates summary plots across iterations.

Main functions:

- `RUN_COMBOS(...)`
  - top-level runner
  - handles original-model execution and parameter-combination generation
- `RUN_COMBINATION(...)`
  - executes one parameter set or the original model
  - runs calibration, validation, sensitivity, and correlation analysis
- `init_csv_with_header(...)`
  - creates or resets the output CSV

### `Calibration_functions.py`

This file handles optimization.

Main responsibilities:

- defines the PSO problem,
- builds the objective function,
- runs Mealpy's `OriginalPSO`,
- merges optimized parameters back into the full parameter dictionary.

Main functions:

- `RUN_PSO_CALIBRATION(...)`
  - runs one PSO optimization
  - currently uses:
    - `epoch=100`
    - `pop_size=50`
    - `c1=1.5`
    - `c2=1.5`
    - `w=0.5`
- `define_cost_function(...)`
  - returns a closure used by the optimizer
  - the cost is the sum of squared differences between simulated and experimental values across the variables listed in `system_data["var_names"]`

### `Validation_functions.py`

This file evaluates model quality against the validation dataset.

Main responsibilities:

- runs the model on validation initial conditions,
- compares simulated outputs against validation measurements,
- computes:
  - RMSE
  - NRMSE
  - MAPE
  - AIC
  - BIC
  - Durbin-Watson statistic
  - Anderson-Darling normality statistic
- produces residual plots.

Main functions:

- `RUN_VALIDATION(...)`
- `residuals_error_equations(...)`

Note:

- The validation stage uses `x0_exp_v`, `t_exp_v`, and `df_val` from `system_data`.
- The original model solution is also stored in `system_data["original_sol_v"]` so later calibrated solutions can be plotted against it.

### `Param_analysis_functions.py`

This file performs parameter identifiability and sensitivity analysis.

Main responsibilities:

- local sensitivity analysis,
- finite-difference perturbation runs,
- Fisher Information Matrix construction,
- correlation matrix calculation,
- t-value calculation,
- structural identifiability checks.

Main functions:

- `PARAMETER_ANALYSIS(...)`
- `sensitivity_analysis(...)`
- `sim_plus_minus(...)`
- `identifiability_check(...)`
- `compute_correlation_matrix(...)`
- `compute_t_values(...)`

Core idea:

- each parameter is perturbed,
- the model is simulated again,
- output differences are used to approximate derivatives,
- those derivatives feed the sensitivity summary and FIM.

### `Plotting_functions.py`

This file generates and saves figures.

Main responsibilities:

- correlation heatmaps,
- sensitivity bar plots,
- residual histograms and Q-Q plots,
- single-run solution comparison plots,
- summary plots across many iterations.

Key functions:

- `plot_corr_matrix(...)`
- `plot_sensitivity_analysis(...)`
- `plot_residuals_analysis(...)`
- `plotting_single_solution(...)`
- `solution_somparison(...)`
- `plot_sensitivity_summary(...)`
- `plot_residuals_summary(...)`
- `plot_corr_summary(...)`

### `Aux_Functions.py`

Right now this file is very small in the current codebase.

It currently provides:

- `format_number(...)`

Some older documentation still describes this file as containing more analysis helpers, but those responsibilities now live mostly in `Validation_functions.py` and `Param_analysis_functions.py`.

## Model Folders

Each model folder provides the pieces needed by the generic framework.

### `System_data.py`

This file is the model configuration layer.

It defines:

- `model_folder_name`
- `var_names`
- `parameters`
- `parameters_og_list`
- `constants`
- initial conditions for simulation and for experimental runs
- simulation time grid
- Excel import logic
- sensitivity and correlation settings
- the final `system_data` dictionary

This file is where the shared runner gets nearly all of its configuration.

The most important output is:

```python
system_data = {...}
```

That dictionary is passed into almost every major function in the framework.

### `DAE_Systems_Simulations.py`

This file contains the actual mathematical model.

Each model defines:

- `DAE_system(...)`
  - normal simulation using fixed parameter values
- `DAE_system_calibrating(...)`
  - simulation form where selected parameters are replaced by optimization variables
- `simulate_model(...)`
  - builds the CasADi DAE system
  - calls the IDAS integrator
  - returns a Pandas DataFrame of time-series results

This is the only place where the biological process equations themselves are defined.

### `Experimental_data.xlsx`

This file contains the experimental data used for:

- parameter estimation or calibration,
- independent validation.

The expected sheets depend on the model. For example:

- `M_smegmatis/System_data.py` reads:
  - `PE_Normal`
  - `V_Normal`
- `M_smegmatis_ammonia_def/System_data.py` reads:
  - `PE_Amm_def`
  - `V_Amm_def_rep`

This means the workbook structure is model-specific and must match the code in `System_data.py`.

## How Data Flows Through The Code

Here is the runtime flow in plain language:

1. A notebook imports one model's `system_data` and `simulate_model`.
2. `RUN_COMBOS(...)` creates a results folder and CSV.
3. The original parameter set is evaluated first.
4. If calibration is enabled, parameter combinations are generated from `parameters_and_bounds`.
5. For each combination and each iteration:
   - PSO searches for better parameter values.
   - The optimized parameters are merged into the full parameter dictionary.
   - Validation is run on the validation dataset.
   - Sensitivity and correlation analysis are run.
   - Figures are saved.
   - A row is appended to the CSV.
6. After all iterations for one combination finish, summary plots are generated.

## How Calibration Works Internally

The calibration loop is built around `Mealpy` PSO:

1. `define_cost_function(...)` creates an objective function.
2. The objective function simulates the model at the parameter-estimation time points in `t_exp`.
3. It compares simulated outputs to `df_exp`.
4. The optimizer minimizes the total squared error across all observed variables.

Important detail:

- the simulation mode for calibration is `"calibrating"`,
- in that mode, only the selected parameters are exposed as optimization variables,
- all other parameters stay fixed at the values in `system_data["parameters"]`.

## How Validation Works

Validation is separate from calibration.

Calibration uses:

- `df_exp`
- `t_exp`
- `x0_exp`

Validation uses:

- `df_val`
- `t_exp_v`
- `x0_exp_v`

That separation is useful because it avoids judging the model only on the data it was fit to.

## How Sensitivity And Correlation Analysis Work

For each parameter:

1. The model is run with a small positive perturbation.
2. The model is run with a small negative perturbation.
3. Finite differences estimate how outputs change with that parameter.

Two slightly different perturbation schemes are used:

- `delta` for FIM-related calculations
- `perturbation` as a percentage for sensitivity plots

From those perturbation runs, the code computes:

- a sensitivity matrix,
- a Fisher Information Matrix,
- a pseudoinverse-based covariance approximation,
- a parameter correlation matrix,
- t-values for the calibrated parameters.

## Example: `M_smegmatis_O2`

This is a good example of how the framework is extended for a specific system.

Compared with the base `M_smegmatis` model, it adds:

- an oxygen state `O`,
- oxygen-related parameters:
  - `k_O`
  - `YX_O2`
- extra constants:
  - `O2_sat`
  - `k_La`

Its `simulate_model(...)` therefore integrates:

- differential states:
  - `X`
  - `C`
  - `N`
  - `CO2`
  - `O`
- algebraic state:
  - `pH`

This shows the pattern for extending the package: the shared framework stays the same, and the model folder changes.

## Output Structure

Results are written inside the selected model folder.

Typical pattern:

```text
M_smegmatis_O2/
  M_smegmatis_O2_Results/
    Calibration_3/
      M_smegmatis_O2_Calibration_3_data.csv
      original_model/
      ['YX_N']/
      ['Xmax']/
      ['YX_N', 'Xmax']/
```

Each parameter-combination folder typically contains:

- solution plots,
- sensitivity plots,
- residual analysis plots,
- correlation matrices,
- summary versions across iterations.

## What You Need To Change For A New Model

To adapt this framework to another DAE system, the minimum work is:

1. Create a new model folder.
2. Add `System_data.py`.
3. Add `DAE_Systems_Simulations.py`.
4. Add `Experimental_data.xlsx`.
5. Update your notebook or script imports to point to that folder.

In practice:

- put all model constants, parameter values, sheet names, time vectors, and initial conditions in `System_data.py`,
- put all differential and algebraic equations in `DAE_Systems_Simulations.py`,
- keep the shared analysis package unchanged unless you are intentionally changing framework behavior.

## Known Codebase Gaps And Mismatches

These are useful to know before running or editing the project:

- `Main.ipynb` still references `RUN_ALL_COMBOS`, but the code currently exposes `RUN_COMBOS`.
- the notebook mentions `System_info.py`, but the real file name in the repository is `System_data.py`.
- the notebook calls arguments that no longer exist in `RUN_functions.py`.
- `requirements.txt` is not ready for direct `pip install -r requirements.txt` use.
- `README.md` still describes some older structure and some helper locations that do not exactly match the current files.
- many functions rely on `os.getcwd()`, so the repository root should be your working directory when you run the notebooks or scripts.

That last point matters a lot because paths such as:

```python
os.path.join(os.getcwd(), model_folder_name, 'Experimental_data.xlsx')
```

assume the current working directory is the repository root.

## Practical Run Checklist

Before running the workflow, make sure:

1. Your terminal or notebook working directory is the repository root.
2. Your Python environment has all required packages installed.
3. Your chosen model folder contains:
   - `System_data.py`
   - `DAE_Systems_Simulations.py`
   - `Experimental_data.xlsx`
4. The Excel sheet names expected by `System_data.py` actually exist.
5. Your notebook imports `RUN_COMBOS`, not `RUN_ALL_COMBOS`.
6. Your call arguments match the current function signature in `DAE_analysis_package/RUN_functions.py`.

## Short Summary

The project is modular in a clean way:

- `DAE_analysis_package/` is the reusable framework.
- each model folder contains only model-specific equations and data.
- the notebook is meant to be a thin orchestration layer.

If you want to understand the code quickly, read it in this order:

1. `Main.ipynb`
2. `DAE_analysis_package/RUN_functions.py`
3. `DAE_analysis_package/Calibration_functions.py`
4. `DAE_analysis_package/Validation_functions.py`
5. `DAE_analysis_package/Param_analysis_functions.py`
6. the selected model folder's `System_data.py`
7. the selected model folder's `DAE_Systems_Simulations.py`

That order follows the actual execution path from user entry point down to the DAE equations.
