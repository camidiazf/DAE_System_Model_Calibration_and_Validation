# DAE_analysis_package

Generic toolkit to **calibrate, validate and diagnose** a dynamic (DAE/ODE) model against experimental time-series data. It was built for the *M. smegmatis* DAE growth model (biomass `X`, carbon `C`, nitrogen `N`, pH, plus unmeasured `O`, `CO2` and the Droop nitrogen quota `q`), but nothing in the package is organism-specific: the model enters only through a `simulate_model` function and a `system_data` dictionary (see [Contract with the model](#contract-with-the-model)).

---

## 1. Why all this? (the modelling idea)

A mechanistic model has many parameters and only a few measured states. Fitting it is not enough; you need to know whether the fit is **good**, **honest** and **identifiable**. The package answers these questions in a fixed pipeline:

| Question | Step | Module |
|---|---|---|
| What parameter values reproduce the data? | **Calibration**: a Particle Swarm Optimisation (PSO) minimises the squared error between simulation and the *calibration* experiment | `Calibration_functions` |
| Does the calibrated model predict data it has not seen? | **Validation**: the same parameters are evaluated on a separate *validation* experiment. Metrics are range-scaled (NRMSE), plus AIC/BIC, and residual tests (Anderson-Darling, Durbin-Watson) | `Validation_functions` |
| Can each parameter actually be estimated from this data? | **Parameter analysis**: relative sensitivities, Fisher Information Matrix (FIM), t-values, 95 % CI, correlations (determinability), condition number | `Param_analysis_functions` |
| Which subset of parameters should be calibrated? | **Combinatorial campaign**: every combination of candidate parameters is calibrated several times (independent PSO runs); runs are compared | `RUN_functions`, `Comparison_functions` |
| Is the extended model better than a simpler one? | **Baseline comparison** (e.g. Monod model) on the same data with the same formulas | `Comparison_functions` |
| Can I reproduce / resume / audit this months later? | **Results store**: tidy tables, stable run ids, seeds, config and versions on disk | `Results_store` |

Key design decisions (and why):

* **Two conditions, two datasets.** Every run is evaluated in both conditions (`normal` = nutrient replete, `N_def` = ammonium deficient) and on both datasets (`cal` = calibration experiment, `val` = validation experiment). The PSO is fitted only to the `cal` data of `calibration_condition` (default `N_def`); everything else is a genuine test. Rows in the tables carry a `fitted` flag so a fitted number is never mistaken for a prediction.
* **Range-scaled residuals.** `(y_exp - y_sim) / range(y_exp)` makes X (g/L), N, pH, ... dimensionless and comparable. NRMSE, AIC and BIC are all built on them. (The PSO cost can be switched from native-units SSE to range-scaled with `system_data['cost_scaling']='range'`; native is the default and lets the large-magnitude variable dominate.)
* **Parsimony via AIC/BIC.** Fit alone always improves with more parameters; `K = m + 1` (m calibrated parameters + residual variance) penalises that.
* **Identifiability is part of the result, not an afterthought.** A parameter set that fits well but has t < 2, `|kappa| > 0.95` or a huge FIM condition number is not trustworthy.
* **Reproducibility.** Every PSO run has an explicit, logged seed; seeds do not depend on the order of combinations; a run is "done" only when `run.json` is written last, so crashes are resumable.
* **Failures are data.** A failed run is recorded (status, error) and the campaign continues; the ranking lists excluded runs instead of silently dropping them.

---

## 2. Package layout and data flow

```
                      RUN_COMBOS  (RUN_functions.py)               <- user entry point
                           |
            +--------------+--------------------+
            |                                   |
      RUN_ORIGINAL                       RUN_COMBINATION  x every parameter subset
            |                                   |  x `iterations` independent runs
            +------------> _run_single <--------+
                              |
        +---------------------+------------------------+
        |                     |                        |
 RUN_PSO_CALIBRATION   RUN_VALIDATION (cal,val)  PARAMETER_ANALYSIS
 (Calibration_)        (Validation_)             (Param_analysis_)
        |                     |                        |
        +----- plots --> Plotting_functions <----------+
                              |
                    ResultsStore.save_run  (Results_store.py)  -> tables/, runs/, combos/
                              |
                 RUN_MODEL_COMPARISON (Comparison_functions.py) -> comparison/
```

| File | Role |
|---|---|
| `RUN_functions.py` | Orchestration of a campaign |
| `Calibration_functions.py` | PSO + cost function |
| `Validation_functions.py` | Error metrics, AIC/BIC, residual tests |
| `Param_analysis_functions.py` | Sensitivity, FIM, t-values, correlations |
| `Results_store.py` | On-disk storage, ids, resume, summary tables |
| `Plotting_functions.py` | All figures + saving helpers |
| `Comparison_functions.py` | Ranking and comparison of all runs (+ baseline) |
| `Best_models_functions.py` | **Legacy** best-model analysis from the old wide CSVs; still provides the baseline helpers |
| `Aux_Functions.py` | `format_number` |
| `test_campaign.py` | End-to-end plumbing test with a stub simulator |

Imports use the absolute package name (`from DAE_analysis_package.X import ...`), so run code from the folder that **contains** `DAE_analysis_package/`.

---

## 3. Contract with the model

The package never imports a model. You give it:

**`simulate_model(system_data, simulation_type, x0, parameters, time, p_vars=None)`** returning a DataFrame with one column per state in `system_data['var_names']` (or `None` on failure).

* `simulation_type='normal'`: simulate with the full `parameters` dict.
* `simulation_type='calibrating'`: PSO call; `p_vars` is the vector being optimised, mapped onto `system_data['calibrating_parameters']` inside the simulator, other values taken from `parameters`.

**`system_data`** (dict). Keys the package reads:

| Key | Meaning |
|---|---|
| `var_names`, `var_exp_names` | All states / states with experimental data |
| `parameters`, `parameters_og_list` | Base parameter dict / list of all parameter names |
| `x0_exp_<cond>`, `t_exp_<cond>`, `df_exp_<cond>` | Calibration experiment (initial state, times, data) |
| `x0_exp_v_<cond>`, `t_exp_v_<cond>`, `df_val_<cond>` | Validation experiment |
| `weights_exp_stack_<cond>` | Replicate-based weights (only used for `fim_mode='replicate'`) |
| `x0_sim_<cond>`, `time_stamps_sim` | Dense grid used for smooth trajectories and plots |
| `delta` / `delta_rel` | FIM finite-difference step (absolute / relative, relative preferred) |
| `perturbation` | Relative step (e.g. 0.05) for sensitivity |
| `correlation_threshold` | `|kappa|` above which a pair is "highly correlated" (0.95) |
| `fim_mode` | `'residual'` (default) or `'replicate'` |
| `sens_window` | `'exp'` (default) or `'sim'` |
| *optional* | `pso_epoch`, `pso_pop_size`, `pso_seed_base`, `cost_scaling`, `baseline_calibrated_parameters`, `condition_labels`, `dataset_names`, `xmax` |

`<cond>` is the extension `_normal` or `_N_def`. `calibrating_parameters` is set by the package for each run.

---

## 4. Module reference

### 4.1 `RUN_functions.py` - campaign orchestration

* **`RUN_COMBOS(system_data, simulate_model, folder_model, folder_results, solve_calibrate, iterations, parameters_and_bounds, min_k, max_k, baseline, resume, calibration_condition, rank_on, rank_dataset, top_frac, allow_config_change)`** - main entry point. Creates `<cwd>/<folder_model>/<folder_model>_Results/<folder_results>/`, writes the config, runs the original model first, builds all combinations of size `min_k..max_k` of the candidate parameters (largest first), runs each `iterations` times, refreshes the summary exports (Excel every 5 combinations) and finishes with `RUN_MODEL_COMPARISON`. `solve_calibrate='solve'` evaluates the original model only. Returns the `ResultsStore`. Raises if *every* run failed.
* **`RUN_COMBINATION`** - runs one combination `iterations` times. Computes the order-independent seed from `pso_seed_base`, the combination hash and the iteration index; skips runs already done (resume) and keeps the ETA honest; draws the per-combination summary plots when at least 2 iterations exist.
* **`RUN_ORIGINAL`** - evaluates the starting parameters (no calibration) as run `original`. Its trajectories become the dashed reference line in every figure.
* **`_run_single`** - the unit of work: (1) calibrate (skipped for `original`), (2) for each condition and dataset call `RUN_VALIDATION`, (3) call `PARAMETER_ANALYSIS` on the condition's calibration experiment, (4) assemble the tidy rows, (5) draw the single-solution figure, (6) save everything. On failure it stores the status/error (and calibrated parameters if the PSO had finished) but does **not** write `run.json`, so `resume=True` retries it. t-values of the condition the PSO was *not* fitted on are flagged `fitted_on=False` (they are not estimation statistics).
* Helpers: `suppress_all_output`, `_fmt_time`, `_progress_text` (ETA), `_to_float`.

### 4.2 `Calibration_functions.py` - PSO

* **`get_pso_settings(system_data)`** - single source of truth for the PSO settings, also written to `campaign.json`. Defaults: `epoch=50`, `pop_size=30`, `c1=c2=1.5`, `w=0.5` (override with `pso_epoch`, `pso_pop_size`; note some comments in the file still say 100/50).
* **`RUN_PSO_CALIBRATION`** - runs `mealpy` `OriginalPSO` within bounds `[lb, ub]` with an explicit seed (random if none). Returns the full refined parameter dict (`RESULTS`), final cost, seed, per-epoch best-cost history, **bound hits** (parameter within 1 % of the bound width of a bound; a hit suggests the bounds are too tight), number of failed evaluations, and wall time. Raises if all objective evaluations failed; warns if more than half did.
* **`define_cost_function`** - builds the objective: simulate with the candidate `p_vars` on the calibration experiment of the chosen condition, then `sum over measured variables of sum((sim - exp)/scale)^2`. `scale=1` (native units) or the data range (`cost_scaling='range'`). A failed simulation returns `1e6` and is counted, never hidden.

### 4.3 `Validation_functions.py` - metrics

* **`RUN_VALIDATION(..., dataset='val'|'cal', original, n_calibrated, ...)`** - simulates at the experimental time points (for metrics) and on the dense grid (for plots), computes per-variable and global metrics, runs the residual tests, draws `residuals_<dataset>.png`, and returns formatted rows plus tidy `Metrics_rows` / `Residuals_rows` and the dense solution.
  * `m` (number of calibrated parameters) = `len(calibrating_parameters)`, or `len(baseline_calibrated_parameters)` when `original=True` so a baseline gets AIC/BIC too.
* **`residuals_error_equations(y_val, y_sim)`** - per variable: RMSE, NRMSE = RMSE/range, MAPE (zero-valued data points excluded).
* **`model_error_equations(y_val_list, y_sim_list, n_params_updated)`** - pools range-scaled residuals of all variables:
  * `NRMSE_global = sqrt(RSS/n)`
  * `AIC = n ln(RSS/n) + 2K`, `BIC = n ln(RSS/n) + K ln(n)`, `K = m + 1`
  * `AICc = AIC + 2K(K+1)/(n-K-1)` (small-sample, not used in the paper)
  * AIC/BIC are `None` when `m = 0`.
* Residual diagnostics: Anderson-Darling normality on pooled residuals, Durbin-Watson autocorrelation per variable (averaged).

### 4.4 `Param_analysis_functions.py` - identifiability

* **`PARAMETER_ANALYSIS`** - main function. Runs the sensitivity analysis, then builds the **unweighted sensitivity matrix `G`** (rows = time x measured variable, columns = *all* parameters) by central differences (`sim_plus_minus`), and the **FIM** `= sum_v G_v^T G_v / sigma_v^2`, with `sigma_v^2` from the calibration residuals (`fim_mode='residual'`) or replicate weights applied once (`'replicate'`, cross-check only). Then calls correlation, t-value and diagnostics functions. Returns a row (t-values + rank ratio + number of highly correlated pairs) and the detailed objects.
* **`sensitivity_analysis`** - general relative sensitivity `S_ij = mean_t |(p_j / max|y_i|) dy_i/dp_j|` on measured variables, same x0 and time vector for base and perturbed runs. `S < 0.01` = non-sensitive (dashed line in the plot). Also plots the top-5 parameters.
* **`sim_plus_minus`** - simulates `p +/- delta` (FIM) or `p(1 +/- perturbation)` (sensitivity).
* **`identifiability_check`** - rank of `G` for the analysed parameters divided by their count (rough indicator; `matrix_rank` tolerance depends on scaling).
* **`compute_correlation_matrix`** - `kappa_ij = cov_ij / (sigma_i sigma_j)`, with `cov = pinv(FIM_sub)` where *FIM_sub* contains only the calibrated parameters (inverting the full FIM hides collinearity among them). Heatmap + list of pairs with `|kappa| > correlation_threshold`.
* **`compute_t_values`** - `t_j = |theta_j / SE_j|`, significant if `t > 2`; 95 % CI uses z = 1.96.
* **`compute_fim_diagnostics`** - SE, CI, relative error (%), and the **condition number** of the sub-FIM (very large = some parameter combination is not determined by the data even if single t-values look fine).
* `_analysed_parameters` decides which parameters are analysed: the calibrated ones, or `baseline_calibrated_parameters` for the original/baseline model. `_fim_step` selects absolute vs relative step.

### 4.5 `Results_store.py` - storage

One folder per campaign (one `RUN_COMBOS` call). **Tidy long CSV tables are the source of truth**; wide summaries, Excel and figures are derived.

```
<folder_model>/<folder_model>_Results/<campaign>/
  campaign.json         everything needed to reproduce: base params, bounds, PSO, cost scaling, seed base, versions, sessions
  README.md             table dictionary
  tables/               runs, combos, parameters, metrics, identifiability, correlations, sensitivity,
                        residuals, pso_history (.csv)  + summary_wide.csv / summary.xlsx (derived)
  runs/<run_id>/        run.json (written LAST = done marker), solutions/<cond>_<cal|val>.csv, figures/<cond>/
  combos/<combo_id>/    mean +/- sd over iterations of one combination
  comparison/           all-model figures and tables
```

* **`ResultsStore`**
  * `init_campaign` - creates a campaign, or on `resume=True` verifies the settings in `COMPAT_KEYS` (parameters/bounds, base parameters, PSO, cost scaling, calibration condition) are unchanged, so old and new runs stay comparable.
  * `combo_id(params, lb, ub)` -> `k<n>_<hash>`; `run_id(combo_id, i)` -> `<combo_id>__rNN`. Stable ids, independent of loop order.
  * `save_run` - upserts rows by `run_id` (idempotent) and writes `run.json` last. `is_done`, `register_combo`.
  * `save_solution` / `load_solution` - trajectories on the common time grid (interpolated on load if the grid differs).
  * `read(name)`, `build_summary_wide()` (one row per run: parameters, NRMSE/AIC/BIC for each condition x dataset, min t-value, max `|kappa|`, FIM condition number of the fitted condition), `refresh_exports`.
* Module helpers: constants `CONDITIONS`, `DATASETS`, `TABLE_SCHEMAS`; `condition_label` / `dataset_label` (display names from `system_data`), `environment_info` (package versions + git hash), `now_iso`, `jsonable`.

### 4.6 `Plotting_functions.py` - figures

* Saving: `save_fig_file` (saves and closes), `_ensure_dir`; legacy `save_fig`, `figures_dir_from_csv_path`, `save_solution_data`.
* Data helpers: `experiment_data`, `dataset_xmax` (x-axis limit from `system_data['xmax']` or the data).
* **`plotting_single_solution`** - one run, one condition: rows = states, columns = Calibration | Validation, calibrated vs original vs data, parameter table at the left, metrics in column titles, `[fitted]` / `[not fitted]` tags.
* Per run and condition: `plot_corr_matrix`, `plot_sensitivity_analysis`, `plot_residuals_analysis` (histogram + Q-Q).
* Per combination (mean +/- sd over iterations): `plot_sensitivity_summary`, `plot_residuals_summary`, `plot_corr_summary`, `plot_combo_solutions`, and the wrapper `plot_combo_summary`, which reads only the stored tables.

### 4.7 `Comparison_functions.py` - ranking and comparison

* **`RUN_MODEL_COMPARISON(system_data, store, rank_on, rank_dataset, top_frac, baseline)`** - reads stored tables/trajectories (no re-simulation) and writes into `comparison/`:
  * `<prefix>_{Calibration,Validation}_Solutions.png` (one line per run), `..._Solutions_Band.png` (min-max envelope + rank-1 line, *not* a confidence interval), `..._Metrics.png` (global NRMSE, per-variable NRMSE, AIC/BIC)
  * `<prefix>_models_ranking.csv`, `<prefix>_reference_metrics.csv`, `excluded_runs.csv`
  * prefix = `All` or `Top<N>pct`.
* **`rank_runs`** - orders finished runs by global NRMSE of `rank_dataset` (default `cal`, so the validation data never selects the model) for `rank_on` = `N_def`, `normal` or `both` (mean). Failed and non-finite runs go to the excluded table.
* Plot helpers: `plot_solutions`, `plot_solutions_band`, `plot_metrics`. The original model and optional baseline are always drawn on both datasets.

### 4.8 `Best_models_functions.py` - legacy

`RUN_BEST_MODELS` is the **old** selection step (it reads the two wide CSVs of the previous `RUN_COMBOS`, re-simulates the top X %, and ranks calibration and validation separately). New campaigns should use `RUN_MODEL_COMPARISON`. The file is still imported for the **baseline helpers**:

* `_baseline_sims` / `_baseline_metrics` - simulate a model of a different structure (e.g. Monod) and compute its metrics with the same formulas; or use precomputed ones.
* `load_baseline_xlsx` - builds a `baseline` dict from an Excel workbook (metrics sheet + trajectory sheets).

The `baseline` dict: `label`, plus either (`simulate_model`, `parameters`, `n_calibrated`, optional `system_data`) or precomputed (`metrics`, `solutions`).

### 4.9 `Aux_Functions.py` and `test_campaign.py`

* `format_number(x, decimals=10)` - safe float-to-string.
* `test_campaign.py` - runs a campaign with a small stub simulator (and stand-ins for `mealpy`/`statsmodels` if not installed) and checks storage, resume, figures and comparison. It does **not** test the PSO or the DAE model.
  ```bash
  MPLBACKEND=Agg python test_campaign.py
  ```

---

## 5. Typical use

```python
from DAE_analysis_package.RUN_functions import RUN_COMBOS
from DAE_analysis_package.Results_store import ResultsStore
from DAE_analysis_package.Comparison_functions import RUN_MODEL_COMPARISON

store = RUN_COMBOS(
    system_data=system_data, simulate_model=simulate_model,
    folder_model="M_smegmatis_extended_modified", folder_results="run_01",
    solve_calibrate="calibrate", iterations=5,
    parameters_and_bounds={"mu_max": [0.05, 0.5], "t_lag": [0, 30], "pH_opt": [6, 8]},
    min_k=2, max_k=3)

store.read("metrics")            # tidy DataFrames
store.read("identifiability")

# resume after a crash: same settings, same folder
RUN_COMBOS(..., folder_results="run_01", resume=True)

# redo / re-parametrise the comparison without recalibrating
store = ResultsStore.open(r".../run_01")
RUN_MODEL_COMPARISON(system_data, store, rank_on="both", rank_dataset="cal", top_frac=0.1)
```

## 6. Reading the results

1. `comparison/*_models_ranking.csv` - best runs and their parameters.
2. Check `bound_hits` (bounds too tight?), `n_failed_evals`, and `pso_history` (converged?).
3. Compare `cal` vs `val` NRMSE of the fitted condition (over-fit?) and the **other** condition (generalisation).
4. Check identifiability of the winner: `min_t > 2`, `max_abs_kappa < 0.95`, moderate `cond_FIM`.
5. Compare AIC/BIC against `Original` and the `Baseline`.

## 7. Dependencies

`numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, mealpy, openpyxl` (see `../requirements.txt`; the models themselves use `casadi`).
