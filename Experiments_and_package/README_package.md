# DAE_System_Model_Calibration_and_Validation

Python framework to **calibrate (PSO), validate and diagnose (FIM, t-values, correlations, information criteria)** DAE bioprocess models against time-series data. It was developed for *Mycobacterium smegmatis* batch cultivation on glycerol and ammonium and is used for the conference paper *"A Droop Quota DAE Model for Mycobacterium smegmatis Batch Cultivation Under Nutrient-Replete and Nitrogen-Limited Conditions"* and its Supplementary Material.

Two models are included: a **baseline** (Monod, Díaz-Figueroa et al. 2025) and an **extended** model (baseline + internal nitrogen quota, self-inhibition exponent, biomass-dependent maintenance, ammonium in the charge balance). See [What changed](CHANGELOG_final_vs_repo.md) for the differences with the previous version of this repository.

## 0. Repository layout and quick start

```
Experiments_and_package/
├── DAE_analysis_package/                       the toolkit (this document, sections 1-7)
├── M_smegmatis_extended_modified/              extended model: DAE_Systems_Simulations.py, System_data.py, <results>
├── M_smegmatis_comparisons_baseline_monod/     baseline model: DAE_Systems_Simulations.py, System_data.py, System_data_amm_def.py
├── Experimental_data.xlsx                      calibration and validation data (Apiyo et al. 2022)
├── baseline_model_data.xlsx                    baseline metrics and trajectories (read by load_baseline_xlsx)
└── requirements.txt
```

```bash
python -m venv .venv && source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
# run notebooks / scripts FROM Experiments_and_package (the data file is read from the current directory)
```

Minimal campaign (the reported-set start values are set in the notebook, **not** in `System_data.py`; see section 8):

```python
from DAE_analysis_package.RUN_functions import RUN_COMBOS
from M_smegmatis_extended_modified.DAE_Systems_Simulations import simulate_model
from M_smegmatis_extended_modified.System_data import system_data

system_data['cost_scaling'] = 'range'; system_data['pso_epoch'] = 200; system_data['pso_pop_size'] = 50
system_data['pso_seed_base'] = 238744
store = RUN_COMBOS(system_data=system_data, simulate_model=simulate_model,
                   folder_model='M_smegmatis_extended_modified', folder_results='my_campaign',
                   solve_calibrate='calibrate', iterations=10,
                   parameters_and_bounds={'mu_max': [0.05, 1.0], 'Xmax': [0.8, 2.5], 'YX_C': [0.05, 1.0],
                                          'YX_CO2': [0.05, 1.0], 'n_cap': [0.3, 5.0]},
                   min_k=5, max_k=5)
```

---

# DAE_analysis_package (the toolkit)

Generic toolkit to **calibrate, validate and diagnose** a dynamic (DAE/ODE) model against experimental time-series data. It was built for the *M. smegmatis* DAE growth model (biomass `X`, carbon `C`, nitrogen `N`, pH, plus unmeasured `O`, `CO2` and the Droop nitrogen quota `q`), but nothing in the package is organism-specific: the model enters only through a `simulate_model` function and a `system_data` dictionary (see [Contract with the model](#contract-with-the-model)).

---

## 1. Why all this? (the modelling idea)

A mechanistic model has many parameters and only a few measured states. Fitting it is not enough; you need to know whether the fit is **good**, **honest** and **identifiable**. The package answers these questions in a fixed pipeline:

| Question | Step | Module |
|---|---|---|
| What parameter values reproduce the data? | **Calibration**: a Particle Swarm Optimisation (PSO) minimises the squared error between simulation and the *calibration* experiment | `Calibration_functions` |
| Does the calibrated model predict data it has not seen? | **Validation**: the same parameters are evaluated on a separate *validation* experiment. Metrics are range-scaled (NRMSE), plus AIC/BIC/AICc, and residual tests (Anderson-Darling, Durbin-Watson) | `Validation_functions` |
| Can each parameter actually be estimated from this data? | **Parameter analysis**: relative sensitivities, Fisher Information Matrix (FIM), t-values, 95 % CI, correlations (determinability), condition number | `Param_analysis_functions` |
| Which subset of parameters should be calibrated? | **Combinatorial campaign**: every combination of candidate parameters is calibrated several times (independent PSO runs); runs are compared | `RUN_functions`, `Comparison_functions` |
| Is the extended model better than a simpler one? | **Baseline comparison** (e.g. Monod model) on the same data with the same formulas | `Comparison_functions` |
| Can I reproduce / resume / audit this months later? | **Results store**: tidy tables, stable run ids, seeds, config and versions on disk | `Results_store` |

Key design decisions (and why):

* **Two conditions, two datasets.** Every run is evaluated in both conditions (`normal` = nutrient replete, `N_def` = ammonium deficient) and on both datasets (`cal` = calibration experiment, `val` = validation experiment). The PSO is fitted only to the `cal` data of `calibration_condition` (default `N_def`); everything else is a genuine test. Rows in the tables carry a `fitted` flag so a fitted number is never mistaken for a prediction.
* **Range-scaled residuals.** `(y_exp - y_sim) / range(y_exp)` makes X (g/L), N, pH, ... dimensionless and comparable. NRMSE, AIC and BIC are all built on them. (The PSO cost can be switched from native-units SSE to range-scaled with `system_data['cost_scaling']='range'`; native is the default and lets the large-magnitude variable dominate.)
* **Parsimony via information criteria.** Fit alone always improves with more parameters; AIC, BIC and AICc with `K = m + 1` (m calibrated parameters + residual variance) penalise that. They assume independent Gaussian residuals, which time series do not satisfy: they rank models, they do not test them.
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
| `Campaign_compare.py` | Joins several campaigns into one table; top-N figures across campaigns |
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
  * `AICc = AIC + 2K(K+1)/(n-K-1)` (small-sample correction; reported in the paper)
  * Akaike weights, nested F-tests and the model ladder of the paper are **not** computed by the package; they come from the analysis scripts of the paper.
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

### 4.8 `Campaign_compare.py` - several campaigns

Joins the results of **several `RUN_COMBOS` calls** (possibly different models) into one table, keeping the trace back to campaign, run and figures. It reads only `campaign.json`, `tables/summary_wide.csv`, `tables/identifiability.csv` and `tables/runs.csv` of each campaign (pandas / numpy / matplotlib only), so **those files must be kept when the repository is committed**.

* Keys: `campaign` (folder name, prefixed with the model if two campaigns share a name); `run_id` (not unique across campaigns); `uid = '<campaign>::<run_id>'` (global key).
* Loading: `find_campaigns(root)`, `load_campaigns(paths, labels)` -> `{'summary', 'identifiability', 'runs', 'settings'}`; `settings_differences` shows which settings differ between campaigns (NRMSE/AIC are only comparable if the relevant ones are the same).
* Tables: `summary_with_identifiability` (NRMSE + identifiability, one row per run), `identifiability_wide`, `identifiability_summary`, `best_per_campaign`, `select_top`, `top_tvalues`, `save_combined` (one Excel workbook).
* Tracing: `trace(table, uid)` (where is this run on disk?), `show_run(table, uid, condition)` (display a stored figure in Jupyter).
* Figures: `plot_nrmse_vs_identifiability`, `plot_top_nrmse_global`, `plot_top_nrmse_per_variable`, `plot_top_tvalues`.
* **`RUN_TOP_COMPARISON(paths, outdir, n, rank_by, ...)`** - one call: takes the n best *calibrated* runs over all campaigns (ranked on the calibration data of the fitted condition, so validation data never selects the models) and writes `Top<n>_NRMSE_global.png`, `Top<n>_NRMSE_per_variable.png`, `Top<n>_tvalues_heatmap.png`, `Top<n>_models.csv`, `Top<n>_tvalues.csv`, `campaigns_used.csv`.

```python
from DAE_analysis_package.Campaign_compare import *
camps = load_campaigns('M_smegmatis_extended_modified')      # searches all campaigns below the folder
table = summary_with_identifiability(camps['summary'], camps['identifiability'])
table.sort_values('N_def_cal_NRMSE').head(10)
RUN_TOP_COMPARISON('M_smegmatis_extended_modified', outdir='Compare_top10', n=10)
```

### 4.9 `Best_models_functions.py` - legacy

`RUN_BEST_MODELS` is the **old** selection step (it reads the two wide CSVs of the previous `RUN_COMBOS`, re-simulates the top X %, and ranks calibration and validation separately). New campaigns should use `RUN_MODEL_COMPARISON`. The file is still imported for the **baseline helpers**:

* `_baseline_sims` / `_baseline_metrics` - simulate a model of a different structure (e.g. Monod) and compute its metrics with the same formulas; or use precomputed ones.
* `load_baseline_xlsx` - builds a `baseline` dict from an Excel workbook (metrics sheet + trajectory sheets).

The `baseline` dict: `label`, plus either (`simulate_model`, `parameters`, `n_calibrated`, optional `system_data`) or precomputed (`metrics`, `solutions`).

### 4.10 `Aux_Functions.py` and `test_campaign.py`

* `format_number(x, decimals=10)` - safe float-to-string.
* `test_campaign.py` - runs a campaign with a small stub simulator (and stand-ins for `mealpy`/`statsmodels` if not installed) and checks storage, resume, figures and comparison. It does **not** test the PSO or the DAE model.
  ```bash
  # from Experiments_and_package, with that folder on the path:
  MPLBACKEND=Agg PYTHONPATH=. python DAE_analysis_package/test_campaign.py
  ```
  (As shipped, the script inserts *its own* folder in `sys.path`, so it only runs without `PYTHONPATH` if it is moved one level up, next to `DAE_analysis_package/`.)

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

Pinned in `../requirements.txt`: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, **mealpy 3.0.2** (PSO), **casadi 3.7.2** (the models, IDAS integrator), openpyxl, and the Jupyter stack. Tested with Python 3.13.

## 8. The models

Both models follow the contract of section 3 (`simulate_model` + `system_data`) and carry **both conditions** in one `System_data.py` (suffixes `_normal` = nutrient replete, `_N_def` = ammonium deficient). Initial states are `[X, C, N, CO2, O, (q), pH]` with the algebraic pH last.

| | Baseline (`M_smegmatis_comparisons_baseline_monod`) | Extended (`M_smegmatis_extended_modified`) |
|---|---|---|
| Source | Díaz-Figueroa et al. 2025 (Monod, fixed biomass-per-nitrogen yield) | this work |
| Differential states | 5: X, C, N, CO2, O2 | 6: adds the internal nitrogen quota q (g N / g X) |
| Algebraic variable | pH from the charge balance | pH from the charge balance, ammonium included |
| Parameters (dictionary) | 14 | 21 (adds n_cap, m_cap, ...) |
| Nitrogen limitation of growth | Monod on external N | Droop factor `max(1 - q_min/q, 0)`; uptake `rho` with feedback fills the quota |
| Capacity term | `1 - X/Xmax` | `max(1 - X/Xmax, 0)^n_cap` |
| Maintenance | none | `m_C * max(1 - X/Xmax, 0)^m_cap` (carbon, CO2 and O2 balances) |
| Calibrated in the paper | on the nutrient-replete calibration dataset (3 parameters) | on the ammonium-deficient calibration dataset only (5 parameters) |

Data (`Experimental_data.xlsx`, Apiyo et al. 2022): `PE_Normal` / `V_Normal` (replete calibration / validation), `PE_Amm_def` / `V_Amm_def_rep` (ammonium-deficient calibration / validation). `V_Amm_def` is a spiking experiment and is not used. The other sheets are present in the workbook but not used by these models.

Equations of the extended model (state `[X, C, N, CO2, O, q]`, algebraic `pH`):

```
mu    = mu_max*(1-exp(-t/t_lag)) * C/(C+k_C) * O/(O+k_O) * max(1-X/Xmax,0)^n_cap * exp(-((pH-pH_opt)/pH_w)^2) * max(1-q_min/q,0)
m(X)  = m_C*max(1-X/Xmax,0)^m_cap
rho   = rho_N_max * N/(N+k_Nup) * max((q_max-q)/(q_max-q_min),0)
dX/dt = (mu-k_d)X          dC/dt = -(mu/YX_C + m(X))X          dq/dt = rho - mu*q
dN/dt = -rho*X + f_Nrec*k_d*q*X
dCO2/dt = (mu/YX_CO2 + Y_CO2_m*m(X))X - gas exchange     dO/dt = kLa(O2_sat-O) - (mu/YX_O2 + Y_O2_m*m(X))X
0     = [OH-]+[HCO3-]+[HPO4^2-]+3[Cit3-]+2[HCit2-]+[H2Cit-] - pH_alk - [H+] - [NH4+]
```

`n_cap = m_cap = 1` recovers the logistic capacity term of the baseline structure; `q_max -> infinity` removes the uptake feedback. The charge balance mixes units (phosphate, citrate and CO2 in g/L, ammonium in mol/L); this is a known limitation. The initial quota `q(0) = 0.096 g N/g X` is assumed, not measured.

### Important: defaults versus the reported set

`System_data.py` holds **starting values**, not the reported solution. The reported set of the paper (campaign `seventh_run_0`) is obtained by overriding the dictionary before the run and calibrating five parameters:

| Start values used in `seventh_run_0` | |
|---|---|
| `k_C 0.08, k_O 0.001, k_d 0.001, t_lag 7, m_C 0.008, m_cap 1.5, rho_N_max 0.5, k_Nup 1e-4, q_min 1e-4, q_max 0.096, f_Nrec 0.5, pH_opt 7, pH_w 10, YX_O2 1.352, Y_CO2_m 0.05, Y_O2_m 1.0` | assumed (fixed) values |
| `mu_max, Xmax, YX_C, YX_CO2, n_cap` | calibrated; bounds 0.05-1, 0.8-2.5, 0.05-1, 0.05-1, 0.3-5; PSO 200 epochs x 50 particles, `cost_scaling='range'`, 10 runs |

The assumed values are the authors' choices, not measurements and not fits.

## 9. Known limitations of this version

* The baseline `System_data.x0_sim_N_def` has 7 entries (a stray quota value) although the baseline has 5 states; the dense-grid simulation of the baseline in the ammonium-deficient condition therefore returns `None`. Baseline curves in the figures come from `baseline_model_data.xlsx`.
* The package default PSO is 50 epochs x 30 particles and `cost_scaling='native'`; set both explicitly for paper-style campaigns.
* `.gitignore` must not exclude `campaign.json` / `run.json` (section 4.5, 4.8).
* A full-size PSO campaign was not part of the checks made while writing this document; only the plumbing test and a small end-to-end run were executed.
