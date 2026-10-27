import os
import sys
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import copy
import logging

from numpy.linalg import cond, matrix_rank, pinv

from DAE_analysis_package.Plotting_functions import plot_sensitivity_analysis, plot_corr_matrix


logger = logging.getLogger(__name__)

# =====================================================================================
# CHANGELOG vs. the previous version  (search for "[FIX n]" in the code)
#  [FIX 1] FIM built as  sum_v G_v^T G_v / sigma_v^2  (sigma_v^2 from calibration residuals, advisor slide).
#          Before: J = G*w ; FIM = J^T J  ->  the weights entered SQUARED (w^2) -> t-values of 1e3-1e4.
#  [FIX 2] Sensitivity analysis: base run and perturbed runs now use the SAME times (before: perturbed runs
#          at t_exp, base run on time_stamps_sim -> pandas aligned them by row number, not by time).
#          Uses the GENERAL relative sensitivity  p/max|y| * dy/dp  (advisor slide) on the measured variables.
#  [FIX 3] Correlations (determinability) from the inverse of the sub-FIM of the CALIBRATED parameters
#          (before: pinv of the full 19x19 FIM, which hides the collinearity of the calibrated parameters).
#  [FIX 4] t-values: z = 1.96, CI and relative error returned/logged, FIM condition number reported.
#  [FIX 5] Optional relative finite-difference step for the FIM (system_data['delta_rel']).
#  [FIX 6] identifiability_check moved out of the loop and applied to the analysed parameters only.
#  [FIX 7] Baseline (original=True) can get t-values/correlations if
#          system_data['baseline_calibrated_parameters'] is defined (list of the parameters it calibrated).
# =====================================================================================


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


# [FIX 7] Which parameters are the ones actually estimated (and therefore analysed in FIM / t-values / correlations)?
def _analysed_parameters(system_data: Dict[str, any], original: bool) -> Optional[List[str]]:
    """
    - Calibrated model (original=False): the parameters calibrated in this run.
    - 'original' model: normally nothing was estimated (None). If the 'original' model is the BASELINE that was
      calibrated elsewhere (Diaz-Figueroa 2025), define in System_data
          baseline_calibrated_parameters = ['YX_N', 'Xmax', 'mu_max']
      and its t-values/correlations are computed exactly like for the extended model.
    """
    if original is False:
        return list(system_data['calibrating_parameters'])
    return system_data.get('baseline_calibrated_parameters', None)


# [FIX 5] Step for finite differences in the FIM
def _fim_step(system_data: Dict[str, any], parameters: Dict[str, float], key: str) -> float:
    """
    If system_data['delta_rel'] exists (e.g. 1e-3): step = delta_rel*|p|  (scale-free, works for tiny parameters).
    Otherwise falls back to the old absolute step system_data['delta'] (1e-4).
    """
    delta_rel = system_data.get('delta_rel', None)
    if delta_rel is not None and parameters[key] != 0:
        return delta_rel * abs(parameters[key])
    return system_data['delta']


def PARAMETER_ANALYSIS(system_data: Dict[str, any],
                        simulate_model:callable,
                        fig_outdir: str,
                        iteration: int,
                        parameters: Dict[str, float],
                        ext: str, # Extension for the validation results
                        original: bool = False,
                        label: Optional[str] = None   # [STORE] run label for figure titles (files are named by the caller's folder)
                        ):


    # Sensitivity Analysis
    sensitivity_analysis_results = sensitivity_analysis(system_data=system_data,
                                                        simulate_model=simulate_model,
                                                        fig_outdir=fig_outdir,
                                                        iteration=iteration,
                                                        parameters=parameters,
                                                        ext=ext,
                                                        original=original,
                                                        label=label)

    # Correlation Matrix and t-values via FIM
    parameters_og_list = system_data['parameters_og_list']
    weights_exp = system_data['weights_exp_stack'+ext]      # only used now for the optional replicate-variance FIM
    var_exp_names = system_data['var_exp_names']

    n_params = len(parameters_og_list)
    n_t = len(system_data['t_exp'+ext])
    n_v = len(var_exp_names)

    # [FIX 1] Unweighted sensitivity matrix G (n_t*n_v rows, n_params columns).
    # Row order = flatten of an (n_t, n_v) array -> index = time_index*n_v + variable_index.
    # (The old weighted matrix J = G*w is gone: it caused the w^2 problem.)
    G = np.zeros((n_t * n_v, n_params))

    for i, key in enumerate(parameters_og_list):
        h = _fim_step(system_data, parameters, key)                           # [FIX 5]
        result = sim_plus_minus(system_data=system_data,
                                simulate_model=simulate_model,
                                key=key,
                                parameters=parameters,
                                ext=ext,
                                var_names = var_exp_names,                   # FIM only on MEASURED variables (as before)
                                step=h)                                      # [FIX 5]
        if result is None:
            logger.error("Simulation failed for parameter %s", key)
            failed_row = [None] * n_params
            failed_row.extend([None, None])
            return {'Row_param_analysis': failed_row,
                    'Sensitivity_results': sensitivity_analysis_results,
                    'Correlation_matrix': None}
        sim_plus, sim_minus = result
        dY_dp = (np.vstack(sim_plus).T - np.vstack(sim_minus).T) / (2 * h)  # (n_t, n_v), central difference
        G[:, i] = dY_dp.flatten()                                           # Unweighted sensitivities

    # [FIX 6] Identifiability check ONCE (it was inside the loop, re-run on a half-filled G at every iteration)
    analysed = _analysed_parameters(system_data, original)
    cols = [parameters_og_list.index(p) for p in analysed] if analysed else list(range(n_params))
    rank_ratio = identifiability_check(system_data=system_data, G=G[:, cols], names=[parameters_og_list[c] for c in cols])

    # [FIX 1] Fisher Information Matrix  FIM = sum_t G(t)^T Q^-1 G(t),  Q = diag(sigma_v^2)
    Gc = G.reshape(n_t, n_v, n_params)
    mode = system_data.get('fim_mode', 'residual')
    if mode == 'residual':
        # sigma_v^2 = sum_i e_vi^2/(N-1) from the calibration residuals (advisor slide, 'errors estimated from residuals')
        sol = simulate_model(system_data=system_data, simulation_type='normal',
                             x0=system_data['x0_exp'+ext], parameters=parameters, time=system_data['t_exp'+ext])
        if sol is None:
            logger.error("Base simulation for residual variance failed.")
            return {'Row_param_analysis': [None] * (n_params + 2),
                    'Sensitivity_results': sensitivity_analysis_results,
                    'Correlation_matrix': None}
        e = np.column_stack([system_data['df_exp'+ext][v].values - sol[v].values for v in var_exp_names])
        sigma2 = (e ** 2).sum(axis=0) / (n_t - 1)
        FIM = sum(Gc[:, v, :].T @ Gc[:, v, :] / sigma2[v] for v in range(n_v))
    else:
        # 'replicate': weights (1/(3 SE^2) from the replicate variance) applied ONCE (not squared). Use only as a cross-check:
        # these weights span ~9 orders of magnitude (0.57 ... 6.8e8) and ignore model misfit.
        w = np.asarray(weights_exp, float).reshape(-1, 1)
        FIM = (G * w).T @ G

    t_values = [None] * n_params
    correlation_matrix = None
    n_highly_corr = None
    fim_diag = None

    if np.any(np.isnan(FIM)) or np.any(np.isinf(FIM)):
        print("!!!!!!!!!!!!! FIM contains NaNs or infinities. Simulation or sensitivities failed.")
        sensitivity_analysis_results = None
    else:
        # [FIX 3] Correlation matrix for determinability (calibrated parameters only)
        correlation_matrix = compute_correlation_matrix(system_data=system_data, fig_outdir=fig_outdir, iteration=iteration, FIM=FIM, original=original,
                                                        label=label, ext=ext)

        n_highly_corr = len(correlation_matrix['highly_correlated_pairs'])

        # [FIX 4] t-values for parameter significance
        t_values = compute_t_values(system_data=system_data, parameters=parameters, FIM=FIM, original=original)
        fim_diag = compute_fim_diagnostics(system_data=system_data, parameters=parameters, FIM=FIM, original=original)

    row_param_analysis = list(t_values)
    row_param_analysis.extend([rank_ratio, n_highly_corr])
    return {"Row_param_analysis": row_param_analysis,
            'Sensitivity_results': sensitivity_analysis_results,
            "Correlation_matrix": correlation_matrix['matrix'] if correlation_matrix is not None else None,
            "Correlation_names": correlation_matrix['names'] if correlation_matrix is not None else None,   # [STORE]
            "Correlation_threshold": system_data.get('correlation_threshold'),                              # [STORE]
            "FIM_diagnostics": fim_diag}   # extra key (SE, CI, relative error, condition number); ignore if not needed

def identifiability_check(system_data: Dict[str, any],
                        G: np.ndarray,
                        names: Optional[List[str]] = None):
    """
    Rank of the unweighted sensitivity matrix of the ANALYSED parameters / number of those parameters.
    [FIX 6] Receives only the columns of the analysed parameters (before: all 19, many of them fixed with zero columns).
    NOTE: matrix_rank's default tolerance depends on the scale of G (pH vs g/L); read rank_ratio as a rough indicator,
    and use the FIM condition number / correlations as the quantitative evidence.
    """
    names = names if names is not None else system_data['parameters_og_list']
    col_norms = np.linalg.norm(G, axis=0)

    for name, norm in zip(names, col_norms):
        status = "OK" if norm > 1e-6 else "VERY SMALL -> may be unidentifiable"
        logger.debug("  %-12s %.3e  %s", name, norm, status)

    rank_G = np.linalg.matrix_rank(G)
    return rank_G / G.shape[1]


def sensitivity_analysis(system_data: Dict[str, any],
                        simulate_model:callable,
                        fig_outdir: str,
                        iteration: int,
                        parameters: Dict[str, float],
                        ext: str, # Extension for the validation results
                        original: bool = False,
                        label: Optional[str] = None
                        ):
    """
    GENERAL relative sensitivity (advisor slides):   S_ij = mean_t | (p_j / max_t|y_i|) * dy_i/dp_j |
    Values < 0.01 are considered non-sensitive.

    [FIX 2]  (a) base run and perturbed runs now share the same x0 and the same time vector
                 (before: perturbed runs at t_exp, base run on time_stamps_sim; the two pandas Series were then
                  multiplied aligned by ROW NUMBER, i.e. row k of the experimental run was divided by the state at
                  t = 0.2*k h, essentially by the initial condition).
             (b) general relative form p/max|y| instead of local p/y(t), which diverges when a state -> 0 (e.g. N).
             (c) only the measured variables (var_exp_names), as in the FIM.
    Window: system_data.get('sens_window','exp')
        'exp' : experimental times of the calibration experiment (same points as the FIM)   [default]
        'sim' : 0-100 h grid (system_data['x0_sim'], ['time_stamps_sim'])
    """
    # Compute sensitivity by simulating with given parameters
    perturbation = system_data['perturbation']
    parameters_og_list = system_data['parameters_og_list']
    var_names = system_data['var_exp_names']                                       # [FIX 2c]

    window = system_data.get('sens_window', 'exp')
    if window == 'sim':
        x0_s = system_data['x0_sim'+ext]
        t_s = system_data['time_stamps_sim']
    else:
        x0_s = system_data['x0_exp'+ext]
        t_s = system_data['t_exp'+ext]

    sensitivity_df = pd.DataFrame(index = parameters_og_list,
                                columns=var_names)

    # Get base simulation (same x0 and time as the perturbed runs)
    model_sim_sensitivity = simulate_model(system_data=system_data,
                                            simulation_type='normal',
                                            x0=x0_s,
                                            parameters=parameters,
                                            time=t_s)

    if model_sim_sensitivity is None:
        print("!!!!!!!!!!!!! Simulation for base model sensitivity failed. Please check the parameters and initial conditions.")
        return None

    # max |y| of each measured variable over the window (normalisation of the general relative sensitivity)
    y_max = [np.max(np.abs(model_sim_sensitivity[var].values)) for var in var_names]

    # Perform sensitivity analysis for each parameter
    for key in parameters_og_list:
        base_val = parameters[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        sim_plus_minus_results = sim_plus_minus(system_data=system_data,
                                                simulate_model=simulate_model,
                                                key=key,
                                                parameters=parameters,
                                                var_names = var_names,
                                                ext=ext,
                                                base_val=base_val,
                                                x0=x0_s, time=t_s)                # [FIX 2a] same window as the base run

        if sim_plus_minus_results is None:
            print(f"!!!!!!!!!!!!! Sensitivity Analysis for parameter {key} failed. Please check the parameters and initial conditions.")
            return None

        sim_plus = sim_plus_minus_results[0]
        sim_minus = sim_plus_minus_results[1]

        for i, var in enumerate(var_names):
            # .values -> plain arrays: no more silent index alignment between Series of different length/time
            dxdp = (sim_plus[i].values - sim_minus[i].values)/(2*perturbation*base_val)
            Gij = dxdp * (base_val / y_max[i])                                     # [FIX 2b] general relative sensitivity
            sensitivity_df.loc[key, var] = np.mean(np.abs(Gij))

    # Process and plot sensitivity results
    sensitivity_df = sensitivity_df.astype(float)

    # Parameters below 0.01 in every variable are non-sensitive (slide threshold) -> logged, not removed
    non_sens = sensitivity_df.index[(sensitivity_df.max(axis=1) < 0.01)].tolist()
    logger.info("Non-sensitive parameters (max general relative sensitivity < 0.01): %s", non_sens)

    top5_df = sensitivity_df.sum(axis=1).nlargest(5)   # (removed the dead 'Mean' sort that was overwritten right after)
    top5_keys = top5_df.index
    top5_plot_df = sensitivity_df.loc[top5_keys]

    plot_sensitivity_analysis(fig_outdir=fig_outdir, sensitivity_df=sensitivity_df, top5_plot_df=top5_plot_df, var_names=var_names,
                              title=f"Parameter sensitivities - {label if label else ('Original model' if original else 'Calibrated model')}"
                                    f" ({ext.lstrip('_')})",
                              filename="sensitivity.png")

    return sensitivity_df

def sim_plus_minus(system_data: Dict[str, any],
                    simulate_model:callable,
                    key: str,
                    parameters: Dict[str, float],
                    ext: str,
                    var_names: List[str] = None,
                    base_val: Optional[float] = None,
                    step: Optional[float] = None,
                    x0: Optional[np.ndarray] = None,
                    time: Optional[np.ndarray] = None
                    ) -> Optional[List[np.ndarray]]:
    """
    Function to simulate the model with perturbed parameters for sensitivity analysis or FIM analysis.
    Parameters:
        - key: parameter name to be perturbed.
        - base_val: base value of the parameter for sensitivity analysis (only if performing sensitivity analysis).
        - step: [FIX 5] absolute step for the FIM (default: system_data['delta']).
        - x0, time: [FIX 2] initial conditions / time vector of the simulations
                    (default: calibration experiment, system_data['x0_exp'] and ['t_exp'], as before).
    Returns:
        - [Y_plus, Y_minus]: list containing the results of the simulation with perturbed parameters.
        - None: if the simulation fails.

    """

    x0 = system_data['x0_exp'+ext] if x0 is None else x0
    time_stamps = system_data['t_exp'+ext] if time is None else time

    params_plus = copy.deepcopy(parameters)
    params_minus = copy.deepcopy(parameters)

    if base_val is None:                             # FIM analysis
        delta = system_data['delta'] if step is None else step
        # Perturb the parameter for FIM analysis by adding/subtracting delta
        params_plus[key] += delta
        params_minus[key] -= delta
    else:                                            # Sensitivity analysis
        perturbation = system_data['perturbation']
        # Perturb the parameter for sensitivity analysis by a percentage of its base value
        params_plus[key] = base_val * (1 + perturbation)
        params_minus[key] = base_val * (1 - perturbation)

    # Simulate the model with perturbed parameters
    sim_plus = simulate_model(system_data=system_data,
                                simulation_type='normal',
                                x0=x0,
                                parameters=params_plus,
                                time=time_stamps)

    sim_minus = simulate_model(system_data=system_data,
                                simulation_type='normal',
                                x0=x0,
                                parameters=params_minus,
                                time=time_stamps)

    # Check for simulation failure
    if sim_plus is None:
        print(f"!!!!!!!!!!!!! Simulation plus with parameter {key} perturbed up failed. Please check the parameters and initial conditions.")
    elif sim_minus is None:
        print(f"!!!!!!!!!!!!! Simulation minus with parameter {key} perturbed down failed. Please check the parameters and initial conditions.")

    if sim_plus is None or sim_minus is None:
        return None

    # Extract results for each variable
    Y_plus = []
    Y_minus = []
    for var in var_names:
        Y_plus.append(sim_plus[var])
        Y_minus.append(sim_minus[var])

    return [Y_plus, Y_minus]



def compute_correlation_matrix(system_data: Dict[str, any],
                                fig_outdir: str,
                                iteration: int,
                                FIM: np.ndarray,
                                original: bool = False,
                                label: Optional[str] = None,
                                ext: str = ''
                                ) -> np.ndarray:
    """
    Parameter correlation (determinability, advisor slides) kappa_ij = cov_ij / (sigma_i sigma_j), cov = FIM^-1.
    Calls for plotting function to visualize the correlation matrix.

    [FIX 3] Computed from the sub-FIM of the ANALYSED (calibrated) parameters only. Before: pinv of the full 19x19 FIM,
            where the fixed parameters take part in the inversion; with the extended model it gave
            mu_max-t_lag 0.59 (sub-FIM: 1.00) and mu_max-pH_opt -0.58 (sub-FIM: 0.97), and counted 14 pairs > 0.80
            among all 19 parameters instead of the pairs among the calibrated ones.
    Returned 'matrix' is therefore n_calibrated x n_calibrated (19x19 only if nothing is defined as calibrated).
    Threshold: system_data['correlation_threshold'] -> set it to 0.95 (slides: highly correlated if |kappa| > 0.95).
    """
    parameters_og_list = system_data['parameters_og_list']
    correlation_threshold = system_data['correlation_threshold']

    analysed = _analysed_parameters(system_data, original)
    if analysed:
        names = list(analysed)
        idx = [parameters_og_list.index(p) for p in names]
        Cov = pinv(FIM[np.ix_(idx, idx)])
    else:                                    # nothing was estimated: legacy behaviour (all parameters)
        names = list(parameters_og_list)
        Cov = pinv(FIM)

    variances = np.diag(Cov)
    variances = np.where(variances > 0.0, variances, np.nan)
    std = np.sqrt(variances)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = std[:, None] * std[None, :]
        corr = Cov / denom
        corr = np.where(np.isfinite(corr), corr, np.nan)

    # Symmetrize, clip values, set diagonal to 1
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Plot heatmap (single plot now: the analysed parameters)
    plot_corr_matrix(fig_outdir=fig_outdir, corr=corr, parameters_og_list=names,
                     title=f"Parameter correlation matrix - {label if label else ('Original model' if original else 'Calibrated model')}"
                           f" ({ext.lstrip('_')})",
                     filename="corr_matrix.png")

    # Extract highly correlated pairs
    i_upper, j_upper = np.triu_indices(len(names), k=1)
    corr_vals = corr[i_upper, j_upper]

    # Mask to get correlations above threshold
    mask = ~np.isnan(corr_vals) & (np.abs(corr_vals) > correlation_threshold)

    if np.any(mask):
        high_pairs = pd.DataFrame({
            "Param 1": [names[i] for i in i_upper[mask]],
            "Param 2": [names[j] for j in j_upper[mask]],
            "Correlation": corr_vals[mask]
        })
        high_pairs["|Correlation|"] = high_pairs["Correlation"].abs()
        high_pairs = high_pairs.sort_values(by="|Correlation|", ascending=False)
        high_pairs = high_pairs.drop(columns="|Correlation|")
        logger.info("Highly correlated pairs (|kappa| > %s):\n%s", correlation_threshold, high_pairs.to_string(index=False))
    else:
        high_pairs = pd.DataFrame(columns=["Param 1", "Param 2", "Correlation"])

    # (removed: unused 'dict_corr_groups' grouping block)

    return {'matrix': corr, 'highly_correlated_pairs': high_pairs, 'corr_calibrated': corr if analysed else None,
            'names': names}


def compute_t_values(system_data: Dict[str, any],
                    parameters: Dict[str, float],
                    FIM: np.ndarray,
                    original: bool = False
                    ) -> List[Optional[float]]:
    """
    t-values  t_j = |theta_j / sigma_j|  from the sub-FIM of the analysed parameters (same as before), where
    sigma_j^2 = (FIM_sub^-1)_jj.  Significant if t > 2 (slides). CI = theta +- 1.96 sigma.
    [FIX 4] z = 1.96 (was 2, and the CI was computed but never returned/used); CI and relative error logged.
    [FIX 7] works for the baseline when system_data['baseline_calibrated_parameters'] is defined.
    Returns a list with one entry per parameter in parameters_og_list (None for fixed parameters).
    """
    parameters_og_list = system_data['parameters_og_list']

    params_list = _analysed_parameters(system_data, original)
    if not params_list:
        return [None] * len(parameters_og_list)

    # Determine indices of calibrated parameters
    indices = [parameters_og_list.index(p) for p in params_list]
    FIM_sub = FIM[np.ix_(indices, indices)]
    cov_sub = pinv(FIM_sub)

    theta = np.array([parameters[p] for p in params_list])

    std_err = np.sqrt(np.diag(cov_sub))
    std_err[std_err <= 0] = np.nan

    t_vals = np.abs(theta / std_err)          # |theta/sigma| as defined in the slides

    # 95% CI: theta +- 1.96 sigma
    z = 1.96
    ci_low = theta - z * std_err
    ci_high = theta + z * std_err
    rel_err = 100 * z * std_err / np.abs(theta)

    lines = [f"{'Parameter':<12}{'theta':>14}{'SE':>14}{'t':>10}{'signif(t>2)':>13}{'CI low':>14}{'CI high':>14}{'rel.err %':>11}"]
    for p, th, se, tv, lo, hi, re in zip(params_list, theta, std_err, t_vals, ci_low, ci_high, rel_err):
        signif = "YES" if (tv > 2) else "NO"
        lines.append(f"{p:<12}{th:14.6g}{se:14.6g}{tv:10.2f}{signif:>13}{lo:14.6g}{hi:14.6g}{re:11.1f}")
    logger.info("t-values (FIM, z=1.96):\n%s", "\n".join(lines))

    # Build complete list  (None for fixed parameters)
    t_values_complete: List[Optional[float]] = []
    for lbl in parameters_og_list:
        if lbl in params_list:
            t_values_complete.append(float(t_vals[params_list.index(lbl)]))
        else:
            t_values_complete.append(None)

    # Error check
    if len(t_values_complete) != len(parameters_og_list):
        logger.error("Length mismatch in t-values computation.")
        return [None] * len(parameters_og_list)

    return t_values_complete


def compute_fim_diagnostics(system_data: Dict[str, any],
                            parameters: Dict[str, float],
                            FIM: np.ndarray,
                            original: bool = False) -> Optional[Dict[str, object]]:
    """
    [FIX 4] Extra information for the paper: SE, 95% CI, relative error (%), and condition number of the sub-FIM.
    A condition number ~1e10 (extended model) means that some linear combination of the calibrated parameters is not
    determined by the data, even if each individual t-value is computed.
    """
    params_list = _analysed_parameters(system_data, original)
    if not params_list:
        return None
    parameters_og_list = system_data['parameters_og_list']
    idx = [parameters_og_list.index(p) for p in params_list]
    F = FIM[np.ix_(idx, idx)]
    se = np.sqrt(np.clip(np.diag(pinv(F)), 0, None))
    theta = np.array([parameters[p] for p in params_list])
    return {'parameters': params_list, 'theta': theta, 'SE': se,
            'CI_low': theta - 1.96 * se, 'CI_high': theta + 1.96 * se,
            'rel_error_pct': 100 * 1.96 * se / np.abs(theta),
            'cond_FIM': float(np.linalg.cond(F))}