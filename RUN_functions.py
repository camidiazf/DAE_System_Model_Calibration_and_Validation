import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import copy
import time

from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from System_info import system_info as system_data
from Aux_Functions import define_cost_function, format_number, append_csv_row, init_csv_with_header, validation_analysis, parameter_analysis, all_param_combos
from Plotting_functions import plot_residuals_summary, plot_sensitivity_summary, plot_corr_summary, _ensure_dir

import sys
import contextlib

# Suppress all output from PSO Optimization to avoid cluttering the console during optimization
# Comment this out if you want to see the output from PSO
# Note: This will suppress all output, including errors in the optimization, so use with caution.
@contextlib.contextmanager
def suppress_all_output():
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

def RUN_ALL_COMBOS(iterations: int,
                    initial_given_data: Dict[str, List[float]],
                    folder: str,
                    file_name: str,
                    updated_parameters: Optional[Dict[str, float]] = None,
                    min_k: int = 1,
                    max_k: Optional[int] = None) -> pd.DataFrame:

    """
    
    """
    
    os.makedirs(folder, exist_ok=True)

    all_params = list(initial_given_data.keys())
    combos = all_param_combos(all_params, min_k=min_k, max_k=max_k)

    csv_path = os.path.join(os.getcwd(), folder, file_name)
    column_names = system_data['column_names']
    if not os.path.exists(csv_path):
        init_csv_with_header(csv_path, column_names)

    initial_run = RUN_PARAMETER_COMBO(
        iterations=iterations,          
        params_list=[],                 
        lb=[], ub=[],                   
        folder=folder,
        file_name=file_name,
        combo_name="Original",          
        updated_parameters=updated_parameters,
        original_model=True                 
    )

    for combo in combos:
        # bounds in the order of this combo
        lb = [float(initial_given_data[p][0]) for p in combo]
        ub = [float(initial_given_data[p][1]) for p in combo]

        # per-combo name (folder-safe, sorted)
        cname = str(combo)

        # call the (slightly tweaked) single-combo runner
        RUN_PARAMETER_COMBO(iterations=iterations,
                            params_list=combo,
                            lb=lb,
                            ub=ub,
                            folder=folder,
                            file_name=file_name,
                            combo_name=cname,
                            updated_parameters=updated_parameters,
                            original_model=False
                            )

    # return the aggregated CSV
    return pd.read_csv(csv_path)


def RUN_PARAMETER_COMBO(iterations: int, 
                                params_list: List[str], 
                                lb: List[float], 
                                ub: List[float], 
                                folder: str, 
                                file_name: str,
                                original_model:bool = True,
                                combo_name: Optional[str] = None, 
                                updated_parameters: Optional[Dict[str, float]] = None
                                ) -> pd.DataFrame:
    """
    Main function to run given parameter calibration iteration and save results to a CSV file.
    Along with figures in a specified folder.
    Parameters:
        - iterations: Number of iterations to run.  
        - params_list: List of parameter names to calibrate.
        - lb: List of lower bounds for the parameters.
        - ub: List of upper bounds for the parameters.
        - folder: Folder to save the results and figures.
        - file_name: Name of the CSV file to save results.
        - combo_name: Optional name for the parameter combination. If None, a name will be generated from params_list.
        - updated_parameters: Optional dictionary of new original parameters to use instead of the default ones.
    """
    
    system_data['calibrating_parameters'] = params_list
    

    # Get original parameters
    parameters_og = system_data['parameters']
    column_names = system_data['column_names']

    # Update original parameters if updated_parameters is provided
    if updated_parameters is None:
        system_data['n_updated_parameters'] = 0
        base_parameters = parameters_og.copy()

    else:
        base_parameters = {}
        system_data['n_updated_parameters'] = len(updated_parameters.keys())
        for key, value in parameters_og.items():
            if key in list(updated_parameters.keys()):
                base_parameters[key] = updated_parameters[key]
            else:
                base_parameters[key] = value
    system_data['base_parameters'] = base_parameters
    # Print given initial parameters
    # print(pd.DataFrame(base_parameters, index=[0]).T.rename(columns={0: 'Original Parameters'}))

    # Prepare CSV path and figure output directory
    csv_path = os.path.join(os.getcwd(), folder, file_name)
    os.makedirs(folder, exist_ok=True)
    need_header = not os.path.exists(csv_path)
    if need_header:
        init_csv_with_header(csv_path, column_names)

    # Decide per-combo figures directory
    if combo_name is None:
        combo_name = str(params_list)

    fig_outdir = os.path.join(folder, combo_name)
    _ensure_dir(fig_outdir)


    # Run initial analysis with original parameters if specified
    if original_model:
        initial_results = RUN_INITIAL(fig_outdir=fig_outdir, base_parameters=base_parameters) 
        row_data = ['Original', '', '', '']
        row_data.extend(initial_results)
        append_csv_row(row_data, csv_path, column_names)
    else:
        

        # Run calibration scenario
        # print(f"\n>>> Calibrating parameters {params_list}, for {iterations} iterations.")
        # print(f"    lb = {lb}")
        # print(f"    ub = {ub}")

        RUN_SCENARIO(fig_outdir=fig_outdir,
                    iterations=iterations,
                    lb=lb,
                    ub=ub,
                    csv_path=csv_path,
                    column_names=column_names,
                    base_parameters=base_parameters
                    )
    print(f"\n>>> Scenario results saved to {csv_path}")

    return pd.read_csv(csv_path)




def RUN_INITIAL(fig_outdir: str, base_parameters: Dict[str, float]) -> List[Any]:
    """
    Run initial analysis with initial given parameters.
    """

    # print(" ")
    # print("-------------------------------------------------------------------------------")
    # print("------------------------------- INITIAL MODEL ---------------------------------")
    # print("-------------------------------------------------------------------------------")
    # print(" ")

    # Run validation analysis with initial given parameters
    ANALYSIS_RESULTS = RUN_ANALYSIS(iteration = None,
                                    parameters=base_parameters,
                                    fig_outdir=fig_outdir)
    

    validation_results = ANALYSIS_RESULTS['validation_results']
    if validation_results is None:
        validation_results_formatted = ['failed'] * (3 * len(system_data['var_names']) + 5) # 3 metrics per variable + 5 overall metrics
    else:
        validation_results_formatted = [format_number(x) for x in validation_results]
    t_values = ANALYSIS_RESULTS['t_values']
    if t_values is None:
        t_values_formatted = ['failed'] * len(system_data['parameters_og_list'])
    else:
        t_values_formatted = [format_number(x) for x in t_values]
    corr_matrix = ANALYSIS_RESULTS['correlation_matrix']
    sensitivity_df = ANALYSIS_RESULTS['sensitivity']
    residuals = ANALYSIS_RESULTS['residuals']

    parameters_og_values = [base_parameters[key] for key in base_parameters.keys()]
    final_results_initial = parameters_og_values + t_values_formatted + validation_results_formatted

    return final_results_initial

def RUN_SCENARIO(fig_outdir: str, 
                iterations: int, 
                lb: List[float], 
                ub: List[float], 
                csv_path: str, 
                column_names: List[str], 
                base_parameters: Dict[str, float]
                ) -> None:
    """
    Run calibration scenario for a given number of iterations and save results to the CSV file.
    Currently, only PSO optimization is implemented.
    It also collects sensitivity, correlation matrix, and residuals for all iterations for summary analysis.
    """

    params_list = system_data['calibrating_parameters']
    sensitivity_df_all = []
    corr_matrix_all = []
    residuals_all = []

    # Iterate for the given number of iterations
    for i in range(iterations):       
        Results = RUN_PSO_CALIBRATION(iteration = i, 
                                lb = lb,
                                ub = ub,
                                fig_outdir=fig_outdir,
                                base_parameters=base_parameters
                                )
                                
        final_result_iteration = Results['FINAL RESULTS']

        row_data = [f"Model_iteration_{i+1}", str(params_list), lb, ub]
        row_data.extend(final_result_iteration)

        append_csv_row(row_data, csv_path, column_names)

        # If different than None, collect sensitivity, correlation matrix, and residuals for summary analysis
        sensitivity_df = Results['SENSITIVITY']
        corr_matrix = Results['CORRELATION MATRIX']
        residuals = Results['RESIDUALS']

        if sensitivity_df is not None:
            sensitivity_df_all.append(sensitivity_df)
        else:
            pass
            # print('Not adding sensitivity data for this iteration, it is None')
        if corr_matrix is not None:
            corr_matrix_all.append(corr_matrix)
        else:
            pass
            # print('Not adding correlation matrix for this iteration, it is None')
        if residuals is not None:
            residuals_all.append(residuals)
        else:
            pass
            # print('Not adding residuals for this iteration, it is None')

    # print(" ")
    # print(f"                    ...... ALL ITERATIONS FOR SCENARIO DONE ......")    
    # print(" ")

    # print("-------------------------------------------------------------------------------")
    # print(f"----------- SUMMARY OF {params_list} CALIBRATION ---------------------")
    # print(f"----------- Lower Bounds: {lb} -----------------------------------")
    # print(f"----------- Upper Bounds: {ub} -----------------------------------")
    # print("-------------------------------------------------------------------------------")
    # print(" ")

    # Run summary analysis with all collected data from iterations
    RUN_SUMMARY_ANALYSIS(fig_outdir, sensitivity_df_all, corr_matrix_all, residuals_all)


def RUN_PSO_CALIBRATION(iteration: int, 
                        lb: List[float], 
                        ub: List[float], 
                        fig_outdir: str, 
                        base_parameters: Dict[str, float]
                        ) -> Dict[str, Any]:
    """
    Run PSO optimization for parameter calibration.
    Returns a dictionary with final results, sensitivity, correlation matrix, and residuals.
    """
    params_list = system_data['calibrating_parameters']
    # print(" ")
    # print("------------------------------------------------------------------------------")
    # print(f"------ PSO OPTIMIZATION FOR PARAMETER CALIBRATION | Iteration {iteration + 1} -------")
    # print("------------------------------------------------------------------------------")
    # print(" ")

    # Define the optimization problem
    problem = {
    "obj_func": define_cost_function(base_parameters=base_parameters),
    "bounds": FloatVar(lb=lb, ub=ub),
    "minmax": "min"
    }

    # Initialize PSO optimizer (you can adjust parameters as needed)
    pso = PSO.OriginalPSO(epoch=100, pop_size=50, c1=1.5, c2=1.5, w=0.5)
    
    # Solve the optimization problem, suppressing output, and measure time taken
    start = time.perf_counter()
    with suppress_all_output():
        g_best = pso.solve(problem)
    end = time.perf_counter()
    
    # print("Optimization Results:")
    # print("     Best Solutions: ", g_best.solution)
    # print("     Minimum Error:", g_best.target.fitness)
    # print(f"     Optimization Time: {end - start:.2f} s")

    # print(" ")
    # print("---------------------------------------------------------------------------")
    # print(f"----------------------- NEW PARAMETERS MODEL {iteration +1} -----------------")
    # print("---------------------------------------------------------------------------")
    # print(" ")

    # Update parameters with optimized values
    new_params = g_best.solution
    new_params_dict = dict(zip(params_list, new_params))

    parameters_updated = copy.deepcopy(base_parameters)
    parameters_values = []
    
    i_param = 0
    for param in base_parameters.keys():
        if param not in params_list:
            parameters_updated[param] = base_parameters[param]
            parameters_values.append(None)
        else:
            new_value = new_params_dict[param]
            parameters_updated[param] = new_value
            parameters_values.append(new_value)
            upper_limit = True
            lower_limit = True
            # Check if the new parameter value is at limit of given bounds
            if abs((abs(new_value) - abs(ub[i_param]))) < 1e-3:
                # print(f"!!!       Warning: Parameter '{param}' reached its upper limit ({ub[i_param]}).")
                upper_limit = False
            if abs((abs(new_value) - abs(lb[i_param]))) < 1e-3:
                # print(f"!!!       Warning: Parameter '{param}' reached its lower limit ({lb[i_param]}).")                
                lower_limit = False
            if upper_limit and lower_limit:
                pass
                # print(f"Parameter '{param}' is between the limits ({lb[i_param]}, {ub[i_param]}).")
            i_param += 1

        parameters_values_formatted = [format_number(x) for x in parameters_values]

    df_new_params = pd.DataFrame({
        "Parameter      ": params_list,
        "Original / Base": [base_parameters[key] for key in params_list],
        "New"            : [parameters_updated[key] for key in params_list]})
    # print(" ")
    # print(df_new_params)
    # print(" ")

    # Run analysis with updated parameters
    ANALYSIS_RESULTS = RUN_ANALYSIS(iteration = iteration,
                                    parameters=parameters_updated,
                                    fig_outdir=fig_outdir)
    validation_results = ANALYSIS_RESULTS['validation_results']
    if validation_results is None:
        validation_results_formatted = ['failed'] * (3 * len(system_data['var_names']) + 5) # 3 metrics per variable + 5 overall metrics
    else:
        validation_results_formatted = [format_number(x) for x in validation_results]
    t_values = ANALYSIS_RESULTS['t_values']
    if t_values is None:
        t_values_formatted = ['failed'] * len(system_data['parameters_og_list'])
    else:
        t_values_formatted = [format_number(x) for x in t_values]

    corr_matrix = ANALYSIS_RESULTS['correlation_matrix']
    sensitivity_df = ANALYSIS_RESULTS['sensitivity']
    residuals = ANALYSIS_RESULTS['residuals']

    final_results_escenario = parameters_values_formatted + t_values_formatted + validation_results_formatted

    return {'FINAL RESULTS' : final_results_escenario,
                'SENSITIVITY' : sensitivity_df,
                'CORRELATION MATRIX' : corr_matrix,
                'RESIDUALS' : residuals
                }

def RUN_ANALYSIS(iteration: int, parameters: dict, fig_outdir: str) -> Dict[str, Any]:
    """
    Run validation and parameter analysis for given parameters.
    Returns a dictionary with validation results, residuals, t-values, correlation matrix, and sensitivity
    """
    # Get system data
    parameters_og_list = system_data['parameters_og_list']

    # Run validation analysis with given parameters
    val_analysis = validation_analysis(iteration = iteration,
                                        parameters = parameters,
                                        fig_outdir=fig_outdir)
    if val_analysis is None:
        # print("\n!!!!!!!!!!!!! Validation Analysis failed. Please check the parameters and initial conditions.")
        validation_results = None
        residuals = None
    else:
        validation_results = val_analysis['Validation results']
        residuals = val_analysis['Residuals']

    # Run parameter analysis with given parameters
    param_analysis = parameter_analysis(iteration = iteration,
                                        parameters = parameters,
                                        fig_outdir = fig_outdir)
    
    # Error check
    if param_analysis is None:
        # print("\n!!!!!!!!!!!! Parameter Analysis failed. Please check the parameters and initial conditions.")
        t_values = None
        corr_matrix = None
        sensitivity_df = None
    else:
        t_values = param_analysis['t_values']
        corr_matrix = param_analysis['correlation_matrix']
        sensitivity_df = param_analysis['sensitivity']

    return {'validation_results': validation_results,
            'residuals': residuals,
            't_values': t_values,
            'correlation_matrix': corr_matrix,
            'sensitivity': sensitivity_df}

def RUN_SUMMARY_ANALYSIS(fig_outdir: str, sensitivity_df_all: pd.DataFrame, corr_matrix_all: np.ndarray, residuals_all: np.ndarray) -> None:

    """
    Run summary analysis with all collected data from iterations for sensitivity, correlation matrix, and residuals.
    """
    
    # Sensitivity Analysis 
    if sensitivity_df_all is None or len(sensitivity_df_all) == 0 or sensitivity_df_all == []:
        # print("No sensitivity data available for plotting.")
        pass
    else:
        plot_sensitivity_summary(fig_outdir, sensitivity_df_all)

    # Residuals Histogram and Q-Q 
    if residuals_all is None or len(residuals_all) == 0 or residuals_all == []:
        pass
        # print("No residuals data available for plotting.")
    else:
        plot_residuals_summary(fig_outdir, residuals_all)

    # Correlation Matrix 
    if corr_matrix_all is None or len(corr_matrix_all) == 0 or corr_matrix_all == []:
        # print("No correlation matrix data available for plotting.")
        pass
    else:
        parameters_og_list = system_data['parameters_og_list']
        corr_stack = np.stack(corr_matrix_all, axis=0)
        mean_corr  = corr_stack.mean(axis=0)
        std_corr   = corr_stack.std(axis=0)
        plot_corr_summary(fig_outdir, corr_stack, mean_corr, parameters_og_list)

        # print("Most correlated parameters:")
        # corr_flat = mean_corr.flatten()
        # indices = np.argsort(np.abs(corr_flat))[-10:] 
        # for idx in indices:
        #     i, j = np.unravel_index(idx, mean_corr.shape)
        #     print(f"{parameters_og_list[i]} ↔ {parameters_og_list[j]}: {mean_corr[i,j]:.2f} ± {std_corr[i,j]:.2f}")