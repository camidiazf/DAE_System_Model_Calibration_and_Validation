from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import os
import sys
import logging


from sklearn.metrics import mean_squared_error

from scipy import stats
from statsmodels.stats.stattools import durbin_watson # type: ignore

from DAE_analysis_package.Plotting_functions import plot_residuals_analysis
from DAE_analysis_package.Aux_Functions import format_number
from DAE_analysis_package.Results_store import dataset_label

# Suppress all output from PSO Optimization to avoid cluttering the console during optimization
# Comment this out if you want to see the output from PSO
# Note: This will suppress all output, including errors in the optimization, so use with caution.
@contextmanager
def suppress_all_output():
    """
    Context manager to suppress all stdout and stderr output.
    
    """
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# =====================================================================================
# CHANGELOG vs. the previous version  (search for "[FIX n]")
#  [FIX 1] AIC/BIC also for the BASELINE: before, original=True forced n_updated_parameters = 0 -> AIC/BIC = None.
#          Now K = m + 1 with m = len(system_data['baseline_calibrated_parameters']) when original=True
#          (define it in the baseline System_data, e.g. ['YX_N', 'Xmax', 'mu_max'] -> m = 3). Optional override: n_calibrated.
#  [FIX 2] dataset='cal' evaluates the SAME metrics on the calibration experiment (x0_exp, t_exp, df_exp), scaled by the
#          range of that dataset. Default dataset='val' = old behaviour.
#  [FIX 3] Extra outputs: 'Metrics' (per-variable NRMSE + global NRMSE + AIC/BIC/AICc) and AICc in model_error_equations.
#  The formulas of NRMSE / global NRMSE / AIC / BIC are UNCHANGED (they were correct).
#  [STORE] RUN_VALIDATION also returns 'Metrics_rows' (tidy rows: one per variable + 'global') and 'Residuals_rows'
#          (range-scaled residual per variable and time) for Results_store; residual figures are drawn for BOTH datasets
#          (residuals_cal.png / residuals_val.png, no more overwriting); the 'original_sol_v' side effect on system_data
#          is gone (the original trajectories are stored as run 'original' in the campaign).
# =====================================================================================

def RUN_VALIDATION(system_data: Dict[str, any],
                simulate_model:callable,
                fig_outdir: str,
                parameters: Dict[str, float], # We give if they are orginal or updated parameters
                iteration: int, # None if original model, integer if post-calibration
                ext: str, # Extension for the validation results
                original: bool = False,
                dataset: str = 'val',                  # [FIX 2] 'val' = validation experiment, 'cal' = calibration experiment
                n_calibrated: Optional[int] = None,    # [FIX 1] optional manual override of m (number of calibrated parameters)
                make_plots: Optional[bool] = None,     # default: residual figure for every dataset (file name carries the dataset)
                label: Optional[str] = None,           # run label for figure titles, e.g. 'k4_a1b2c3__r02'
                ) -> List[Any]:

    # Load system information
    var_exp_names = system_data['var_exp_names']
    # [FIX 2] choose the dataset the metrics are computed on
    if dataset == 'cal':
        x0_exp_v = system_data['x0_exp'+ext]
        t_exp_v = system_data['t_exp'+ext]
        df_val = system_data['df_exp'+ext]
    else:
        x0_exp_v = system_data['x0_exp_v'+ext]
        t_exp_v = system_data['t_exp_v'+ext]
        df_val = system_data['df_val'+ext]
    time_stamps_sim = system_data['time_stamps_sim']
    condition = ext.lstrip('_')           # '_normal' -> 'normal', '_N_def' -> 'N_def'
    if make_plots is None:
        make_plots = True

    calibrating_parameters = system_data['calibrating_parameters']

    # Initialize lists to store data from validation experiment and simulation with parameters given
    y_val = []
    y_sim = []

    # Simulation with experimental validation data time points and initial validation conditions, to compare with experimental data (same size)
    sol = simulate_model(system_data=system_data, 
                            simulation_type='normal', 
                            x0=x0_exp_v, 
                            parameters=parameters, 
                            time=t_exp_v)
    
    sol_smooth = simulate_model(system_data=system_data, 
                            simulation_type='normal', 
                            x0=x0_exp_v, 
                            parameters=parameters, 
                            time=time_stamps_sim)
    
    # Solution check
    if sol is None:
        print("!!!!!!!!!! Simulation for validation failed. Please check the parameters and initial conditions.")
        return None
    
    for var in var_exp_names:
        var_exp_v = var
        y_val.append(df_val[var_exp_v].values)
        y_sim.append(sol[var].values)


    # Number of parameters whose value was chosen with data (m).  [FIX 1]
    # Before: original=True -> 0 -> no AIC/BIC for the baseline, although the baseline had 3 calibrated parameters.
    if n_calibrated is not None:
        n_updated_parameters = n_calibrated
    elif original is True:
        n_updated_parameters = len(system_data.get('baseline_calibrated_parameters', []))   # 0 (-> no AIC) if not defined
    else:
        n_updated_parameters = len(calibrating_parameters)

    # ---------- Per-variable metrics (RMSE, NRMSE, MAPE) ----------
    val_results_df_rows = []
    for var, y_v, y_s in zip(var_exp_names, y_val, y_sim):
        m = residuals_error_equations(y_v, y_s)
        val_results_df_rows.append({'Variable': var,
                                    'RMSE': m['RMSE'],
                                    'NRMSE': m['NRMSE'],
                                    'MAPE': m['MAPE'],
                                    'AIC': None,
                                    'BIC': None})

    # ---------- Whole-model metrics (scaled residuals) ----------
    model = model_error_equations(y_val, y_sim, n_updated_parameters)
    val_results_df_rows.append({'Variable': 'Model',
                                'RMSE': None,    # not meaningful when pooling different units
                                'NRMSE': model['NRMSE_global'],
                                'MAPE': None,    # not meaningful when pooling different units
                                'AIC': model['AIC'],
                                'BIC': model['BIC']})
 
    val_results_df = pd.DataFrame(val_results_df_rows)


    # ---------- Statistical tests on residuals ----------
    residuals = model['residuals']  # pooled, range-scaled residuals
 
    # Durbin-Watson: one variable at a time (pooled residuals mix time series)
    dw_by_var = []
    for var, r in zip(var_exp_names, model['residuals_by_var']):
        dw = durbin_watson(r)
        dw_by_var.append(dw)
    dw_statistic = float(np.mean(dw_by_var))  # single number kept for the results row
 
        # Anderson-Darling test for normality (on scaled residuals)
    try:
        result = stats.anderson(residuals, dist='norm', method='interpolate')   # SciPy >= 1.17
    except TypeError:                                                           # older SciPy without 'method'
        result = stats.anderson(residuals)

    if make_plots:
        ttl = (f"Residuals analysis - {label if label else ('Original model' if original else 'Calibrated model')}"
               f"\n{dataset_label(system_data, condition, dataset)}")
        plot_residuals_analysis(fig_outdir, residuals=residuals, title=ttl, filename=f"residuals_{dataset}.png")
 
    # ---------- Results row, fixed order, no NaN dropping ----------
    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ' '          # placeholder keeps every value in its own column
        return format_number(x)
 
    validation_results_formatted = []
    for row in val_results_df_rows:
        validation_results_formatted += [_fmt(row['RMSE']), _fmt(row['NRMSE']), _fmt(row['MAPE'])]
        if row['Variable'] == 'Model':
            validation_results_formatted += [_fmt(row['AIC']), _fmt(row['BIC'])]
 
    validation_results_formatted.extend([format_number(result.statistic), format_number(dw_statistic)])
 
    # [FIX 3] raw numbers (unformatted) for tables: per-variable NRMSE, global NRMSE, AIC, BIC, AICc, m
    metrics_raw = {'dataset': dataset, 'm': n_updated_parameters, 'n': len(residuals),
                   'NRMSE_global': model['NRMSE_global'], 'AIC': model['AIC'], 'BIC': model['BIC'], 'AICc': model['AICc']}
    for row in val_results_df_rows[:-1]:
        metrics_raw['NRMSE_' + row['Variable']] = row['NRMSE']

    # [STORE] tidy rows for the results tables (raw numbers, NaN instead of ' ' placeholders).
    # Caller adds run_id / condition / fitted.
    def _f(x):
        return float(x) if x is not None else np.nan
    metrics_rows = []
    for var, y_v, y_s, dwv in zip(var_exp_names, y_val, y_sim, dw_by_var):
        mm = residuals_error_equations(y_v, y_s)
        metrics_rows.append({'dataset': dataset, 'variable': var, 'n': int(len(y_v)), 'm': n_updated_parameters,
                             'RMSE': _f(mm['RMSE']), 'NRMSE': _f(mm['NRMSE']), 'MAPE': _f(mm['MAPE']),
                             'DW': _f(dwv), 'AD': np.nan, 'AIC': np.nan, 'BIC': np.nan, 'AICc': np.nan})
    metrics_rows.append({'dataset': dataset, 'variable': 'global', 'n': int(len(residuals)), 'm': n_updated_parameters,
                         'RMSE': np.nan, 'NRMSE': _f(model['NRMSE_global']), 'MAPE': np.nan,
                         'DW': _f(dw_statistic), 'AD': _f(result.statistic),
                         'AIC': _f(model['AIC']), 'BIC': _f(model['BIC']), 'AICc': _f(model['AICc'])})
    residuals_rows = []
    t_arr = np.asarray(t_exp_v, dtype=float)
    for var, r in zip(var_exp_names, model['residuals_by_var']):
        for ti, ri in zip(t_arr, r):
            residuals_rows.append({'dataset': dataset, 'variable': var, 't': float(ti), 'residual_scaled': float(ri)})

    return {"Row_val_analysis": validation_results_formatted,
        "System_sol": sol_smooth,
        "Residuals": residuals,
        "AIC": model['AIC'] if model['AIC'] is not None else ' ',
        "Metrics": metrics_raw,
        "Metrics_rows": metrics_rows,
        "Residuals_rows": residuals_rows}
    


def residuals_error_equations(y_val: np.ndarray, 
                              y_sim: np.ndarray) -> Dict[str, object]:
    """
    Per-variable metrics. y_val and y_sim are 1D arrays of ONE variable
    (e.g. only X, or only C), validation experimental data vs simulation.
 
    RMSE  : root mean squared error, in the variable's own units.
    NRMSE : RMSE / (max - min of the experimental data).
    MAPE  : mean absolute percentage error. Points where the experimental
            value is exactly 0 are excluded (division by zero), e.g. N in
            the ammonium plateau.
    """
    y_val = np.asarray(y_val, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)
 
    res = y_val - y_sim
    rmse = np.sqrt(np.mean(res**2))
 
    y_range = np.ptp(y_val)
    nrmse = rmse / y_range if y_range > 0 else np.nan
 
    nonzero = y_val != 0
    mape = np.mean(np.abs(res[nonzero] / y_val[nonzero])) * 100 if nonzero.any() else np.nan
 
    return {"RMSE": rmse, "NRMSE": nrmse, "MAPE": mape, "residuals": res}
 
 
def model_error_equations(y_val_list: List[np.ndarray],
                          y_sim_list: List[np.ndarray],
                          n_params_updated: int = 0) -> Dict[str, object]:
    """
    Whole-model metrics. Receives ONE array per variable (lists, not concatenated),
    so each variable's residuals can be scaled by its own range before pooling.
 
    Scaled residual = (y_exp - y_sim) / range(y_exp)  -> dimensionless, so
    X, C, N and pH all weigh comparably.
 
    NRMSE_global : root mean square of all scaled residuals
                   (= sqrt(mean(NRMSE_i^2)) when every variable has the same number of points).
    AIC, BIC     : Gaussian least-squares form, using the SCALED residuals.
                   K = k + 1 (the +1 is the residual variance).
                   Only computed if n_params_updated > 0.
    """
    res_by_var = []
    for yv, ys in zip(y_val_list, y_sim_list):
        yv = np.asarray(yv, dtype=float)
        ys = np.asarray(ys, dtype=float)
        res_by_var.append((yv - ys) / np.ptp(yv))
 
    res_scaled = np.concatenate(res_by_var)
    n = len(res_scaled)
    RSS = np.sum(res_scaled**2)
 
    out = {
        "NRMSE_global": np.sqrt(RSS / n),
        "residuals": res_scaled,          # pooled, scaled -> use for Q-Q plot / Anderson-Darling
        "residuals_by_var": res_by_var,   # use for Durbin-Watson, one variable at a time
        "AIC": None,
        "BIC": None,
        "AICc": None,
    }
 
    if n_params_updated > 0:
        K = n_params_updated + 1
        out["AIC"] = n * np.log(RSS / n) + 2 * K
        out["BIC"] = n * np.log(RSS / n) + K * np.log(n)
        # [FIX 3] small-sample correction (advisor slide: AIC + 2(m+1)(m+2)/(N-m-2) = AIC + 2K(K+1)/(N-K-1)).
        # Not used in the paper (AIC and BIC are reported); available for completeness.
        if n - K - 1 > 0:
            out["AICc"] = out["AIC"] + 2 * K * (K + 1) / (n - K - 1)
 
    return out