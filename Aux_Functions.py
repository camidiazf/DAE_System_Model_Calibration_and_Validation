import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import copy
import logging

from scipy import stats
from numpy.linalg import cond, matrix_rank, pinv
from statsmodels.stats.stattools import durbin_watson # type: ignore
from itertools import combinations

from DAE_Systems_Simulations import simulate_model
from Plotting_functions import plot_sensitivity_analysis, plot_corr_matrix, plotting_comparison, plot_residuals_analysis
from System_info import system_info as system_data

logger = logging.getLogger(__name__)

# Combination function
def all_param_combos(params, min_k=1, max_k=None):
    n = len(params)
    if max_k is None:
        max_k = n
    combos = []
    for k in range(min_k, max_k + 1):
        for combo in combinations(params, k):
            combos.append(list(combo))
    combos.reverse()  # Reverse the list to have larger combinations first
    return combos   

# Analysis functions
def validation_analysis(iteration: int, 
                        parameters: Dict[str, float], 
                        fig_outdir: str
                        ) -> Optional[Dict[str, any]]:
    """
    Function to perform validation analysis of the DAE system model.
    If new_og is provided, it updates the original parameters with new values for comparison.
    Analisis includes residuals, RMSE, NRMSE, MAPE, AIC, BIC, and statistical tests on residuals.
    As well as plotting comparison of original and new parameters with validation data.
    """
    # print("        [--------------- Validation - Residual Analysis ---------------]                       ")
    # print(" ")

    # Load system information
    x0_sim_v = system_data['x0_sim_v']
    var_names = system_data['var_names']
    x0_exp_v = system_data['x0_exp_v']
    t_exp_v = system_data['t_exp_v']
    df_val = system_data['df_val']

    # Initialize lists to store data from validation experiment and simulation with parameters given
    y_val = []
    y_sim = []

    # Simulation with experimental validation data time points and initial validation conditions, to compare with experimental data (same size)
    sol = simulate_model(simulation_type='normal', 
                            x0=x0_exp_v, 
                            parameters=parameters, 
                            time=t_exp_v)
    
    # Error check
    if sol is None:
        # print("!!!!!!!!!! Simulation for validation failed. Please check the parameters and initial conditions.")
        return None

    for var in var_names:
        var_exp_v = var
        y_val.append(df_val[var_exp_v].values)
        y_sim.append(sol[var].values)
        
    y_val_c = np.concatenate(y_val)
    y_sim_c = np.concatenate(y_sim)

    val_results_df_rows = []

    # Residuals, RMSE, NRMSE, MAPE, AIC and BIC for each variable
    for i in range(len(var_names)):
        estado = var_names[i]
        y_v = y_val[i]
        y_s = y_sim[i]
        res_results_var = residuals_equations(y_v, y_s) 
        val_results_df_rows.append({'Variable': estado,
                                        'RMSE': res_results_var[0],
                                        'MAPE': res_results_var[1],
                                        'NRMSE': res_results_var[2],
                                        'AIC': None,
                                        'BIC': None})
    if iteration is None:
        n_updated_parameters = system_data['n_updated_parameters']
    else:
        n_updated_parameters = len(system_data['calibrating_parameters'])  
    res_results_model = residuals_equations(y_val_c, y_sim_c, n_updated_parameters)

    # AIC, BIC, RMSE, MAPE y NRMSE for the whole model
    val_results_df_rows.append({'Variable': 'Model',
                                'RMSE': res_results_model[0][2],
                                'MAPE': res_results_model[0][3],
                                'NRMSE': res_results_model[0][4],
                                'AIC': res_results_model[0][0],
                                'BIC': res_results_model[0][1]})

    # Create DataFrame with validation results
    val_results_df = pd.DataFrame(val_results_df_rows)
    # print(val_results_df)

    val_results_df_numeric = val_results_df.drop(columns=["Variable"]).to_numpy().flatten()
    val_results_df_numeric = [x for x in val_results_df_numeric if not pd.isna(x)]
    
    # Statistical tests on residuals
    residuals = res_results_model[1]
    result = stats.anderson(residuals)

    # Anderson-Darling test
    print(f"\nAnderson-Darling test statistic: {result.statistic}\n")
    print("Critical values and significance levels (Null hypothesis = H0):")
    for i in range(len(result.critical_values)):
        level = result.significance_level[i]
        critical_value = result.critical_values[i]
        null_hypothesis = "not rejected (normality)" if result.statistic < critical_value else "rejected (non-normality)"
        print(f"      Significance level {level}%: Critical value {critical_value} -> H0 {null_hypothesis}")

    # Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    print(f"\nDurbin-Watson statistic: {dw_statistic}\n")
    if dw_statistic < 1.5:
        print("      Positive autocorrelation detected in residuals -> Residual may not be independent.")
    elif dw_statistic > 2.5:
        print("      Negative autocorrelation detected in residuals -> Residual may not be independent.")
    else:
        print("      No autocorrelation detected in residuals -> Residual may be independent.")

    plot_residuals_analysis(residuals, fig_outdir, iteration)

    # Plotting comparison of original and new parameters with validation data
    plotting_comparison(iteration = iteration,
                        parameters_updated=parameters,
                        AIC = res_results_model[0][0],
                        fig_outdir=fig_outdir)

    return {'Validation results': val_results_df_numeric,
            'Residuals': residuals}


def parameter_analysis(iteration: int, 
                        parameters: Dict[str, float],
                        fig_outdir: str
                        ) -> Optional[Dict[str, any]]:
    """
    Function to perform parameter analysis of the DAE system model.
    Analysis includes sensitivity analysis, Fisher Information Matrix (FIM), parameter correlation matrix, and t-values.
    """
    
    # Load system information
    x0_sim = system_data['x0_sim']
    x0_exp = system_data['x0_exp']
    time_stamps_sim = system_data['time_stamps_sim']
    t_exp = system_data['t_exp']

    # print(" ")
    # print("        [----------------- Parameter Analysis -----------------]")

    # Compute sensitivity by simulating with given parameters
    sensitivity = compute_sensitivity(x0 = x0_sim,
                                        parameters = parameters,
                                        time_stamps = time_stamps_sim,
                                        fig_outdir = fig_outdir,
                                        iteration = iteration) 
    
    
    # Compute FIM using given parameters and experimental PE data
    FIM = compute_FIM(iteration = iteration,
                        x0= x0_exp, 
                        parameters = parameters, 
                        time_stamps = t_exp,
                        fig_outdir = fig_outdir)

    # Error check
    if FIM is None:
        # print("!!!!!!!!!!!!! FIM Analysis failed. Please check the parameters and initial conditions.")
        return None
    
    corr_matrix = FIM['correlation_matrix']
    t_values = FIM['t_values']
    FIM = FIM['FIM']  

    return {'correlation_matrix':corr_matrix,
            't_values': t_values,
            'sensitivity': sensitivity}

# Optimization auxiliary function

def define_cost_function(base_parameters: Dict[str, float]
                        ) -> callable:
    """
    Function to define the cost function for calibration using experimental data.
    Parameters:
        - params_list: list of parameter names to be calibrated.
    Returns:
        - cost_function: function that computes the cost based on the difference between simulated and experimental data.
    """
    # Load system information
    var_names = system_data['var_names']
    x0_exp = system_data['x0_exp']

    t_exp = system_data['t_exp']
    df_exp = system_data['df_exp']


    def cost_function(p_vars: np.ndarray) -> float: # COST FUNCTION USING PE DATA
        """
        Computes the cost function based on the difference between simulated and experimental data.
        Parameters:
            - p_vars: array of parameter values to be calibrated
        Returns:
            - err: total error between simulated and experimental data
        """
        try:
            df_results = simulate_model(simulation_type='calibrating', 
                                        x0=x0_exp, 
                                        parameters=base_parameters,
                                        time=t_exp,
                                        p_vars=p_vars
                                    )
            
            err = 0
            for var in var_names:
                var_new = df_results[var]
                var_exp = df_exp[var]

                err += np.sum((var_new - var_exp)**2)
        
            return err
        
        except:
            err = 1e6
            return err
        
    return cost_function

# Analysis auxiliary functions

def sim_plus_minus(key: str, 
                    x0: np.ndarray, 
                    parameters: Dict[str, float], 
                    time_stamps: np.ndarray, 
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
    var_names = system_data['var_names']
    
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
    sim_plus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_plus, 
                                time=time_stamps)
    
    sim_minus = simulate_model(simulation_type='normal', 
                                x0=x0, 
                                parameters=params_minus, 
                                time=time_stamps)

    # Check for simulation failure
    # if sim_plus is None:
    #     print(f"!!!!!!!!!!!!! Simulation plus with parameter {key} perturbed up failed. Please check the parameters and initial conditions.")
    # elif sim_minus is None:
    #     print(f"!!!!!!!!!!!!! Simulation minus with parameter {key} perturbed down failed. Please check the parameters and initial conditions.")

    if sim_plus is None or sim_minus is None:
        return None
    
    # Extract results for each variable
    Y_plus = []
    Y_minus = []
    for var in var_names:
        Y_plus.append(sim_plus[var])
        Y_minus.append(sim_minus[var])

    return [Y_plus, Y_minus]


def residuals_equations(y_val: np.ndarray, 
                        y_sim: np.ndarray, 
                        n_params_updated: Optional[List[str]] = None
                        ) -> Tuple[List[float], Optional[np.ndarray]]:
    """
    Function to compute residuals, RMSE, NRMSE, and MAPE between experimental and simulated data.
    Uses validation experimental data.
    """

    y_val_range = np.max(y_val) - np.min(y_val)

    res = y_val - y_sim

    rmse = np.sqrt(np.mean(res**2))

    nmrse = np.sqrt(np.mean(res**2)) / y_val_range
    
    mape = np.mean(np.abs(res/ y_val)) * 100
    if n_params_updated is None:
        return [rmse, nmrse, mape]
    elif n_params_updated == 0:
        return [[' ', ' ', rmse, nmrse, mape], res]
    else:
        n = len(y_val)
        k = n_params_updated

        rss = np.sum(res**2)

        aic = 2 * k + n * np.log(rss / n)

        bic = k * np.log(n) + n * np.log(rss / n)

        return [[aic, bic, rmse, nmrse, mape], res]

def compute_FIM(iteration: int, 
                x0: np.ndarray, 
                parameters: Dict[str, float], 
                time_stamps: np.ndarray, 
                fig_outdir: str
                ) -> Dict[str, Optional[np.ndarray]]:

    """
    Compute the Fisher Information Matrix (FIM), parameter correlation matrix, and t-values for sensitivity analysis.
    """

    # print(" ")
    # print("                    >>>> FIM Analysis <<<<")
    # print(" ")

    # Load system information and initialize variables
    weights_exp = system_data['weights_exp_stack']
    parameters_og_list = system_data['parameters_og_list']
    n_params = len(parameters_og_list)
    n_outputs = weights_exp.shape[0] * weights_exp.shape[1]
    delta = system_data['delta']
    J = np.zeros((n_outputs, n_params))

    parameters_og_list = system_data['parameters_og_list']
    weights_exp = system_data['weights_exp_stack']
    delta = system_data['delta']
    n_params = len(parameters_og_list)
    n_outputs = weights_exp.size
    J = np.zeros((n_outputs, n_params))

    # Build Jacobian via finite differences
    for i, key in enumerate(parameters_og_list):
        result = sim_plus_minus(key, x0, parameters, time_stamps)
        if result is None:
            logger.error("Simulation failed for parameter %s", key)
            return {'FIM': None, 'correlation_matrix': None, 't_values': None}
        sim_plus, sim_minus = result
        # Flatten and weight
        dY_dp = (np.vstack(sim_plus).T - np.vstack(sim_minus).T) / (2 * delta)
        J[:, i] = (dY_dp * weights_exp).flatten()

    # Compute FIM
    FIM = J.T @ J
    cond_num = cond(FIM)
    rank = matrix_rank(FIM)
    logger.info("FIM condition number: %.2e | rank: %d", cond_num, rank)

    # Correlation matrix
    corr_matrix = compute_correlation_matrix(fig_outdir, iteration, FIM)

    # Compute t-values
    t_values_complete = compute_t_values(iteration, parameters, FIM)
        
    return {'FIM': FIM,
            'correlation_matrix': corr_matrix,
            't_values': t_values_complete,}

def compute_correlation_matrix(fig_outdir: str, 
                                iteration: int, 
                                FIM: np.ndarray
                                ) -> np.ndarray:
    """
    Compute the parameter correlation matrix from the Fisher Information Matrix (FIM).
    Calls for plotting function to visualize the correlation matrix.
    """
    parameters_og_list = system_data['parameters_og_list']
    correlation_threshold = system_data['correlation_threshold']

    # Inverse FIM (pseudo-inverse for stability)
    Finv = pinv(FIM)

    # Standard deviations from diagonal
    diag = np.diag(Finv)
    diag = np.where(diag > 0.0, diag, np.nan)
    std = np.sqrt(diag)

    with np.errstate(divide="ignore", invalid="ignore"): # Handle division by zero
        scale = std[:, None] * std[None, :]
        corr = np.divide(Finv, scale, out=np.full_like(Finv, np.nan, dtype=float), where=~np.isnan(scale))

    # Symmetrize, clip, set diagonal = 1
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Plot heatmap
    plot_corr_matrix(corr, parameters_og_list, fig_outdir, iteration)

    # Extract highly correlated pairs
    iu = np.triu_indices(FIM.shape[0], k=1)
    vals = corr[iu]
    mask_pairs = ~np.isnan(vals) & (np.abs(vals) > correlation_threshold)

    high_pairs = pd.DataFrame({
        "Param 1": [parameters_og_list[i] for i in iu[0][mask_pairs]],
        "Param 2": [parameters_og_list[j] for j in iu[1][mask_pairs]],
        "Correlation": vals[mask_pairs]
    })

    # Sort by absolute correlation
    if not high_pairs.empty:
        high_pairs["|Correlation|"] = high_pairs["Correlation"].abs()
        high_pairs = high_pairs.sort_values(by="|Correlation|", ascending=False).drop(columns="|Correlation|")

    print(f"\nHighly correlated parameter pairs (|r| > {correlation_threshold}):")
    if high_pairs.empty:
        print("  None found.")
    else:
        print(high_pairs.to_string(index=False))

    return corr

def compute_t_values(iteration: int, 
                    parameters: Dict[str, float],
                    FIM: np.ndarray
                    ) -> List[Optional[float]]:
    """
    Function to compute t-values for the parameters based on the Fisher Information Matrix (FIM).
    """
    if iteration is None:
        params_list = []
    else:
        params_list = system_data['calibrating_parameters']
    
    parameters_og_list = system_data['parameters_og_list']

    # If no parameters were calibrated, return None, as it is the initial parameter set
    if iteration is None:
        return [None] * len(parameters_og_list)
    
    # print(" ")
    # print("                    >>>> t-values <<<<") 
    # print(" ")

    # Determine indices of calibrated parameters
    indices = [parameters_og_list.index(p) for p in params_list]
    FIM_sub = FIM[np.ix_(indices, indices)]
    cov_sub = pinv(FIM_sub)

    theta = np.array([parameters[p] for p in params_list])
    std_err = np.sqrt(np.diag(cov_sub))
    std_err[std_err == 0] = np.nan
    t_vals = theta / std_err

    # print("Computed t-values for calibrated parameters:")
    # print(f"{'Parameter':<15}{'θ':>12}{'SE':>12}{'t-value':>12}")

    # for p, th, se, tv in zip(params_list, theta, std_err, t_vals):
    #     print(f"{p:<15}{th:12.6f}{se:12.6f}{tv:12.2f}")

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


def compute_sensitivity(x0, parameters, time_stamps, fig_outdir, iteration):
    """
    Function to compute the sensitivity of the model parameters using perturbation analysis.
    Returns a DataFrame with sensitivity values and generates sensitivity plots.
    """

    # print(" ")
    # print("                    >>>> Sensitivity Analysis <<<<")
    # print(" ")

    perturbation = system_data['perturbation']
    parameters_og_list = system_data['parameters_og_list']
    var_names = system_data['var_names']

    sensitivity_df = pd.DataFrame(index = parameters_og_list, 
                                columns=var_names)
    
    # Get base simulation
    Y_base = []

    model_sim_sensitivity = simulate_model(simulation_type='normal', 
                                            x0=x0, 
                                            parameters=parameters,
                                            time=time_stamps)
    
    if model_sim_sensitivity is None:
        # print("!!!!!!!!!!!!! Simulation for base model sensitivity failed. Please check the parameters and initial conditions.")
        return None
    
    for var in var_names:
        Y_base.append(model_sim_sensitivity[var])

    # Perform sensitivity analysis for each parameter
    for key in parameters_og_list:
        base_val = parameters[key]
        if base_val == 0 or np.isnan(base_val):
            continue

        sim_plus_minus_results = sim_plus_minus(key=key,
                                                x0=x0,
                                                parameters=parameters,
                                                time_stamps=time_stamps,
                                                base_val=base_val)
        
        if sim_plus_minus_results is None:
            # print(f"!!!!!!!!!!!!! Validation Analysis for parameter {key} failed. Please check the parameters and initial conditions.")
            return None
        
        sim_plus = sim_plus_minus_results[0]
        sim_minus = sim_plus_minus_results[1]    

        for i, var in enumerate(var_names):
            delta_Y = sim_plus[i] - sim_minus[i]
            rel_Y = delta_Y / (2 * perturbation * Y_base[i])
            mean_S = np.mean(np.abs(rel_Y))
            sensitivity_df.loc[key, var] = mean_S

    # Process and plot sensitivity results
    sensitivity_df = sensitivity_df.astype(float)
    sensitivity_df['Mean'] = sensitivity_df.mean(axis=1)
    sensitivity_sorted = sensitivity_df.sort_values('Mean', ascending=False)
    top5_df = sensitivity_sorted.head(5)
    sensitivity_df.drop(columns='Mean', inplace=True)

    top5_df = sensitivity_df.sum(axis=1).nlargest(5)
    top5_keys = top5_df.index
    top5_plot_df = sensitivity_df.loc[top5_keys]

    plot_sensitivity_analysis(sensitivity_df, top5_plot_df, var_names, fig_outdir, iteration)

    return sensitivity_df

# General auxiliary functions

def format_number(x: float | list | None, decimals: int = 10) -> Optional[str]:
    """
    Function to format a number to a specified number of decimal places.
    Returns None if input is None or cannot be converted to float, and formatted string otherwise.
    """

    if x is None:
        return None
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    try:
        return f"{float(x):.{decimals}f}"
    except (ValueError, TypeError):
        return x

def init_csv_with_header(path: str, column_names: list[str]) -> None:
    """ 
    Initialize a CSV file with a header.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        os.remove(path)
        print(f">>> Existing file {path!r} removed | starting fresh")

    pd.DataFrame(columns=column_names).to_csv(path, index=False, sep=",", decimal=".")
    
    print(">>> CSV initialized with header")

def append_csv_row(row_values: list, path: str, column_names: list[str]) -> None:
    """Append a row to the CSV, respecting column order."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row_values], columns=column_names)
    df.to_csv(path, mode="a", header=False, index=False, sep=",", decimal=".")