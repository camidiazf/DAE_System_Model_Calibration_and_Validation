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

from DAE_analysis_package.Plotting_functions import plot_residuals_analysis, plotting_single_solution
from DAE_analysis_package.Aux_Functions import format_number

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


def RUN_VALIDATION(system_data: Dict[str, any],
                simulate_model:callable,
                fig_outdir: str,
                parameters: Dict[str, float], # We give if they are orginal or updated parameters
                iteration: int, # None if original model, integer if post-calibration
                original: bool = False,
                ) -> List[Any]:
    
    print("------------------------------")
    print(">>> Running model VALIDATION")
    print("------------------------------")

    # Load system information
    var_exp_names = system_data['var_exp_names']
    x0_exp_v = system_data['x0_exp_v']
    t_exp_v = system_data['t_exp_v']
    df_val = system_data['df_val']
    time_stamps_sim = system_data['time_stamps_sim']

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
    
    if original:
        system_data['original_sol_v'] = sol_smooth

    # Solution check
    if sol is None:
        print("!!!!!!!!!! Simulation for validation failed. Please check the parameters and initial conditions.")
        return None
    
    for var in var_exp_names:
        var_exp_v = var
        y_val.append(df_val[var_exp_v].values)
        y_sim.append(sol[var].values)
        
    y_val_c = np.concatenate(y_val)
    y_sim_c = np.concatenate(y_sim)

    # Residuals, RMSE, NRMSE, MAPE for each variable
    val_results_df_rows = []
    for i in range(len(var_exp_names)):
        estado = var_exp_names[i]
        y_v = y_val[i]
        y_s = y_sim[i]
        res_results_var = residuals_error_equations(y_v, y_s) 
        val_results_df_rows.append({'Variable': estado,
                                        'RMSE': res_results_var[0],
                                        'MAPE': res_results_var[1],
                                        'NRMSE': res_results_var[2],
                                        'AIC': None,
                                        'BIC': None})
    if original is True:
        # initial model validation
        n_updated_parameters = 0
    else:
        # post-calibration model validation
        n_updated_parameters = len(calibrating_parameters)  

    
    # AIC, BIC, RMSE, MAPE y NRMSE for the whole model
    res_results_model = residuals_error_equations(y_val_c, y_sim_c, n_updated_parameters)
    val_results_df_rows.append({'Variable': 'Model',
                                'RMSE': res_results_model[0][2],
                                'MAPE': res_results_model[0][3],
                                'NRMSE': res_results_model[0][4],
                                'AIC': res_results_model[0][0],
                                'BIC': res_results_model[0][1]})

    # Create DataFrame with validation results
    val_results_df = pd.DataFrame(val_results_df_rows)
    print(val_results_df)

    val_results_df_numeric = val_results_df.drop(columns=["Variable"]).to_numpy().flatten()
    val_results_df_numeric = [x for x in val_results_df_numeric if not pd.isna(x)]
    
    # Statistical tests on residuals
    residuals = res_results_model[1]

    # Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    print(f"\nDurbin-Watson statistic: {dw_statistic}")
    if dw_statistic < 1.5:
        print(" > Positive autocorrelation detected in residuals -> Residual may not be independent.\n")
    elif dw_statistic > 2.5:
        print(" > Negative autocorrelation detected in residuals -> Residual may not be independent.\n")
    else:
        print(" > No autocorrelation detected in residuals -> Residual may be independent.\n")

    # Anderson-Darling test for normality
    result = stats.anderson(residuals)
    print(f"Anderson-Darling test statistic: {result.statistic}\n")
    print("Critical values and significance levels (Null hypothesis = H0):")
    for i in range(len(result.critical_values)):
        level = result.significance_level[i]
        critical_value = result.critical_values[i]
        null_hypothesis = "not rejected (normality)" if result.statistic < critical_value else "rejected (non-normality)"
        print(f"      Significance level {level}%: Critical value {critical_value} -> H0 {null_hypothesis}")

    plot_residuals_analysis(fig_outdir, original=original, residuals=residuals, iteration=iteration)

    validation_results = val_results_df_numeric

    if validation_results is None:
        validation_results_formatted = ['failed'] * (3 * len(var_exp_names) + 5) # 3 metrics per variable + 5 overall metrics
    else:
        validation_results_formatted = [format_number(x) for x in validation_results]

    validation_results_formatted.extend([format_number(result.statistic), format_number(dw_statistic)])

    return {"Row_val_analysis": validation_results_formatted, 
            "System_sol": sol_smooth,
            "Residuals": residuals,
            'AIC': res_results_model[0][0],}
    

def residuals_error_equations(y_val: np.ndarray, 
                                y_sim: np.ndarray, 
                                n_params_updated: Optional[int] = None
                                ) -> Tuple[List[float], Optional[np.ndarray]]:
    """
    Function to compute residuals, RMSE, NRMSE, and MAPE between experimental and simulated data.
    Uses validation experimental data.
    """

    y_val_range = np.max(y_val) - np.min(y_val)

    res = y_val - y_sim

    mse = mean_squared_error(y_val, y_sim)

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

        RSS = np.sum(res**2)

        aic = n * np.log(RSS/n) + 2 * (k + 1) 
        bic = RSS/n + k * np.log(n)


        return [[aic, bic, rmse, nmrse, mape], res]
    