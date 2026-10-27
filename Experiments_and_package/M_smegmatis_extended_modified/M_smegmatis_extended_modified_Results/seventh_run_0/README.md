# Calibration campaign

* `campaign.json`  everything needed to reproduce the runs (base parameters, bounds, PSO settings, cost scaling, seed base, versions).
* `tables/`        tidy CSV tables = source of truth (one row per run / parameter / metric ...). `summary_wide.csv` and
                   `summary.xlsx` are DERIVED (one row per run) and are rebuilt automatically.
* `runs/<run_id>/` one folder per run: `run.json` (written last = done marker), `solutions/` (trajectories), `figures/<condition>/`.
* `combos/`        mean +- sd over the iterations of one parameter combination.
* `comparison/`    all-models figures and tables.

run_id = `<combo_id>__rNN`; combo_id = `k<number of calibrated parameters>_<hash of parameters+bounds>` (see tables/combos.csv).
condition: `normal` = nutrient replete, `N_def` = ammonium deficient. dataset: `cal` = calibration (parameter estimation)
experiment, `val` = validation experiment. `fitted` = TRUE only for the calibration dataset of the condition the PSO was fitted on.
Metrics are computed on range-scaled residuals (see Validation_functions.py); NRMSE global = RMS of all scaled residuals.
