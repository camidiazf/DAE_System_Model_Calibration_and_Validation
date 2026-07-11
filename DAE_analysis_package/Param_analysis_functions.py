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

def PARAMETER_ANALYSIS(system_data: Dict[str, any],
                        simulate_model:callable,
                        fig_outdir: str,
                        iteration: int, 
                        parameters: Dict[str, float],
                        original: bool = False
                        ):
    

    print(f"\n------------------------------------")
    print(">>> Running model PARAMETER ANALYSIS")
    print("------------------------------------")
    
    # Sensitivity Analysis 
    sensitivity_analysis_results = sensitivity_analysis(system_data=system_data,
                                                        simulate_model=simulate_model,
                                                        fig_outdir=fig_outdir,
                                                        iteration=iteration,
                                                        parameters=parameters,
                                                        original=original)
    
    # Correlation Matrix and t-values via FIM
    parameters_og_list = system_data['parameters_og_list']
    weights_exp = system_data['weights_exp_stack']
    delta = system_data['delta']

    n_params = len(parameters_og_list)
    n_outputs = weights_exp.size

    # Sensitivity matrix J
    J = np.zeros((n_outputs, n_params))

    # Structural identifiability matrix
    G = np.zeros((n_outputs, n_params))

    for i, key in enumerate(parameters_og_list):
        result = sim_plus_minus(system_data=system_data, 
                                simulate_model=simulate_model,
                                key=key, 
                                parameters=parameters,
                                var_names = system_data['var_exp_names'])
        if result is None:
            logger.error("Simulation failed for parameter %s", key)
            failed_row = [None] * n_params
            failed_row.extend([None, None])
            return {'Row_param_analysis': failed_row,
                    'Sensitivity_results': sensitivity_analysis_results,
                    'Correlation_matrix': None}
        sim_plus, sim_minus = result
        dY_dp = (np.vstack(sim_plus).T - np.vstack(sim_minus).T) / (2 * delta)
        J[:, i] = (dY_dp * weights_exp).flatten() # Weighted sensitivities
        G[:, i] = dY_dp.flatten() # Unweighted sensitivities

    # Identifiability check
        rank_ratio = identifiability_check(system_data=system_data, G=G)


    # Compute FIM
    FIM = J.T @ J
    t_values = [None] * n_params
    correlation_matrix = None
    n_highly_corr = None

    if np.any(np.isnan(FIM)) or np.any(np.isinf(FIM)):
        print("!!!!!!!!!!!!! FIM contains NaNs or infinities. Simulation or sensitivities failed.")
        sensitivity_analysis_results = None
    else:
        # Correlation matrix for determinability
        correlation_matrix = compute_correlation_matrix(system_data=system_data, fig_outdir=fig_outdir, iteration=iteration, FIM=FIM, original=original)

        n_highly_corr = len(correlation_matrix['highly_correlated_pairs'])

        # Compute t-values for parameter significance
        t_values = compute_t_values(system_data=system_data, parameters=parameters, FIM=FIM, original=original)

    row_param_analysis = list(t_values)
    row_param_analysis.extend([rank_ratio, n_highly_corr])
    return {"Row_param_analysis": row_param_analysis,
            'Sensitivity_results': sensitivity_analysis_results,
            "Correlation_matrix": correlation_matrix['matrix'] if correlation_matrix is not None else None}

def identifiability_check(system_data: Dict[str, any],
                        G: np.ndarray,):
    
    # 1. Sensitivity magnitude 
    parameters_og_list = system_data['parameters_og_list']
    n_params = len(parameters_og_list)
    col_norms = np.linalg.norm(G, axis=0)
    print("\nColumn norms (parameter sensitivities):")

    for name, norm in zip(parameters_og_list, col_norms):
        status = "OK" if norm > 1e-6 else "VERY SMALL → may be unidentifiable"
        print(f"  {name:20s}  {norm:.3e}   {status}")

    # 2. Rank test (linear independence) 
    rank_G = np.linalg.matrix_rank(G)

    if rank_G == n_params:
        print(f"\nRank(G): {rank_G} / {n_params}: G is full rank → STRUCTURALLY IDENTIFIABLE")
    else:
        print(f"\nRank(G): {rank_G} / {n_params}: G is rank deficient → NOT STRUCTURALLY IDENTIFIABLE")

    return rank_G/n_params

def sensitivity_analysis(system_data: Dict[str, any],
                        simulate_model:callable,
                        fig_outdir: str,
                        iteration: int, 
                        parameters: Dict[str, float],
                        original: bool = False
                        ):
    
    # Compute sensitivity by simulating with given parameters
    perturbation = system_data['perturbation']
    parameters_og_list = system_data['parameters_og_list']
    var_names = system_data['var_names']

    x0_sim = system_data['x0_sim']
    time_stamps_sim = system_data['time_stamps_sim']

    sensitivity_df = pd.DataFrame(index = parameters_og_list, 
                                columns=var_names)
    
    # Get base simulation
    Y_base = []

    model_sim_sensitivity = simulate_model(system_data=system_data, 
                                            simulation_type='normal', 
                                            x0=x0_sim, 
                                            parameters=parameters,
                                            time=time_stamps_sim)
    
    if model_sim_sensitivity is None:
        print("!!!!!!!!!!!!! Simulation for base model sensitivity failed. Please check the parameters and initial conditions.")
        return None
    
    for var in var_names:
        Y_base.append(model_sim_sensitivity[var])

    # Perform sensitivity analysis for each parameter
    for key in parameters_og_list:
        base_val = parameters[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        sim_plus_minus_results = sim_plus_minus(system_data=system_data,
                                                simulate_model=simulate_model,
                                                key=key,
                                                parameters=parameters,
                                                var_names = system_data['var_names'],
                                                base_val=base_val)
        
        if sim_plus_minus_results is None:
            print(f"!!!!!!!!!!!!! Sensitivity Analysis for parameter {key} failed. Please check the parameters and initial conditions.")
            return None
        
        sim_plus = sim_plus_minus_results[0]
        sim_minus = sim_plus_minus_results[1] 

        for i, var in enumerate(var_names):
            dxdp = (sim_plus[i] - sim_minus[i])/(2*perturbation*base_val)
            Gij = dxdp * (parameters[key]/Y_base[i])
            Gij = np.mean(np.abs(Gij))
            sensitivity_df.loc[key, var] = Gij

    # Process and plot sensitivity results
    sensitivity_df = sensitivity_df.astype(float)
    sensitivity_df['Mean'] = sensitivity_df.mean(axis=1)
    sensitivity_sorted = sensitivity_df.sort_values('Mean', ascending=False)
    top5_df = sensitivity_sorted.head(5)
    sensitivity_df.drop(columns='Mean', inplace=True)

    top5_df = sensitivity_df.sum(axis=1).nlargest(5)
    top5_keys = top5_df.index
    top5_plot_df = sensitivity_df.loc[top5_keys]

    plot_sensitivity_analysis(fig_outdir=fig_outdir, iteration=iteration, sensitivity_df=sensitivity_df, top5_plot_df=top5_plot_df, var_names=var_names, original=original)

    return sensitivity_df

def sim_plus_minus(system_data: Dict[str, any],
                    simulate_model:callable,
                    key: str, 
                    parameters: Dict[str, float],
                    var_names: List[str] = None, 
                    base_val: Optional[float] = None
                    ) -> Optional[List[np.ndarray]]:
    """
    Function to simulate the model with perturbed parameters for sensitivity analysis or FIM analysis.
    Parameters:
        - key: parameter name to be perturbed.
        - x0: initial conditions for the simulation.
        - parameters: dictionary of model parameters and their values.
        - time_stamps: time points for the simulation.
        - base_val: base value of the parameter for sensitivity analysis (only if performing sensitivity analysis).
    Returns:
        - [Y_plus, Y_minus]: list containing the results of the simulation with perturbed parameters.
        - None: if the simulation fails.

    """
    
    x0 = system_data['x0_exp']
    time_stamps = system_data['t_exp']
    
    params_plus = copy.deepcopy(parameters)
    params_minus = copy.deepcopy(parameters)

    if base_val is None:                             # FIM analysis
        delta = system_data['delta']
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
                                original: bool = False
                                ) -> np.ndarray:
    """
    Compute the parameter correlation matrix from the Fisher Information Matrix (FIM).
    Calls for plotting function to visualize the correlation matrix.
    """
    parameters_og_list = system_data['parameters_og_list']
    correlation_threshold = system_data['correlation_threshold']

    # Covariance matrix approximation
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

    # Plot heatmap
    plot_corr_matrix(fig_outdir=fig_outdir, iteration=iteration, corr=corr, parameters_og_list=parameters_og_list, original=original)

    # Extract highly correlated pairs
    i_upper, j_upper = np.triu_indices(len(parameters_og_list), k=1)
    corr_vals = corr[i_upper, j_upper]

    # Mask to get correlations above threshold
    mask = ~np.isnan(corr_vals) & (np.abs(corr_vals) > correlation_threshold)
    
    if np.any(mask):
        high_pairs = pd.DataFrame({
            "Param 1": [parameters_og_list[i] for i in i_upper[mask]],
            "Param 2": [parameters_og_list[j] for j in j_upper[mask]],
            "Correlation": corr_vals[mask]
        })
        high_pairs["|Correlation|"] = high_pairs["Correlation"].abs()
        high_pairs = high_pairs.sort_values(by="|Correlation|", ascending=False)
        high_pairs = high_pairs.drop(columns="|Correlation|")
        print("\nHighly correlated parameter pairs:")
        # print(high_pairs.to_string(index=False))
        # pairs in lists
        
        dict_corr_groups = {}
        i = 0
        for _, row in high_pairs.iterrows():
            p1 = row["Param 1"]
            p2 = row["Param 2"]
            corr_value = row["Correlation"]

            # Find existing group
            found_group = None
            for group in dict_corr_groups.values():
                if p1 in group or p2 in group:
                    found_group = group
                    break

            if found_group is not None:
                found_group.update([p1, p2])
            else:
                dict_corr_groups[f"group_{i}"] = set([p1, p2])
                i += 1
    else:
        high_pairs = pd.DataFrame(columns=["Param 1", "Param 2", "Correlation"])
        print(f"\nNo correlated parameter pairs above threshold |r| > {correlation_threshold}.")

    # Now corr matrix only for calibrated parameters if not original
    if original is False:
        params_list = system_data['calibrating_parameters']
        indices = [parameters_og_list.index(p) for p in params_list]
        corr_calibrated = corr[np.ix_(indices, indices)]

        plot_corr_matrix(fig_outdir=fig_outdir, iteration=iteration, corr=corr_calibrated, parameters_og_list=params_list, original=original, calibrated_only=True)

    return {'matrix': corr, 'highly_correlated_pairs': high_pairs, 'corr_calibrated': corr_calibrated if original is False else None}


def compute_t_values(system_data: Dict[str, any],
                    parameters: Dict[str, float],
                    FIM: np.ndarray,
                    original: bool = False
                    ) -> List[Optional[float]]:
    """
    Function to compute t-values for the parameters based on the Fisher Information Matrix (FIM).
    """
    parameters_og_list = system_data['parameters_og_list']

    if original is True:
        return [None] * len(parameters_og_list)
    else:
        params_list = system_data['calibrating_parameters']

    # Determine indices of calibrated parameters
    indices = [parameters_og_list.index(p) for p in params_list]
    FIM_sub = FIM[np.ix_(indices, indices)]
    cov_sub = pinv(FIM_sub)

    theta = np.array([parameters[p] for p in params_list])

    std_err = np.sqrt(np.diag(cov_sub))
    std_err[std_err <= 0] = np.nan   

    t_vals = theta / std_err

    # 95% CI: theta ± 2σ 
    ci_low = theta - 2 * std_err
    ci_high = theta + 2 * std_err

    print("\nComputed t-values and confidence intervals:")
    print(f"{'Parameter':<15}{'θ':>12}{'SE':>12}{'t-value':>12}{'Signif?':>10}{'CI low':>15}{'CI high':>15}")

    for p, th, se, tv, lo, hi in zip(params_list, theta, std_err, t_vals, ci_low, ci_high):
        signif = "YES" if (abs(tv) > 2) else "NO"
        print(f"{p:<15}{th:12.6f}{se:12.6f}{tv:12.2f}{signif:>10}{lo:15.6f}{hi:15.6f}")
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
    if t_values_complete is None:
        logger.error("t-values computation failed.")
        return [None] * len(parameters_og_list)
    
    return t_values_complete


# def RUN_SUMMARY(system_data: Dict[str, any],
#                         fig_outdir: str, 
#                         sensitivity_df_all: pd.DataFrame, 
#                         corr_matrix_all: np.ndarray) -> None:

#     """
#     Run summary analysis with all collected data from iterations for sensitivity, correlation matrix.
#     """
    
#     # Sensitivity Analysis 
#     if sensitivity_df_all is None or len(sensitivity_df_all) == 0 or sensitivity_df_all == []:
#         print("No sensitivity data available for plotting.")
#         # pass
#     else:
#         plot_sensitivity_summary(fig_outdir, sensitivity_df_all)

#     # Correlation Matrix 
#     if corr_matrix_all is None or len(corr_matrix_all) == 0 or corr_matrix_all == []:
#         print("No correlation matrix data available for plotting.")
#         # pass
#     else:
#         parameters_og_list = system_data['parameters_og_list']

        
#         corr_stack = np.stack(corr_matrix_all, axis=0)
#         mean_corr  = corr_stack.mean(axis=0)
#         std_corr   = corr_stack.std(axis=0)
#         plot_corr_summary(fig_outdir, corr_stack, mean_corr, parameters_og_list)

#         print("Most correlated parameters:")
#         corr_pairs = []
#         for i in range(len(parameters_og_list)):
#             for j in range(i+1, len(parameters_og_list)):  # ensures i < j
#                 corr_pairs.append((
#                     abs(mean_corr[i, j]),
#                     i,
#                     j,
#                     mean_corr[i, j],
#                     std_corr[i, j]
#                 ))
#         corr_pairs_sorted = sorted(corr_pairs, key=lambda x: x[0], reverse=True)
#         for _, i, j, mean_val, std_val in corr_pairs_sorted:
#             if mean_val > 0.9:  # threshold for "most correlated"
#                 print(f"{parameters_og_list[i]} ↔ {parameters_og_list[j]}: {mean_val:.2f} ± {std_val:.2f}")