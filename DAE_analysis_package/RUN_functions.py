import os
import sys
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

import pandas as pd

from itertools import combinations

from DAE_analysis_package.Param_analysis_functions import PARAMETER_ANALYSIS
from DAE_analysis_package.Aux_Functions import format_number
from DAE_analysis_package.Validation_functions import RUN_VALIDATION
from DAE_analysis_package.Calibration_functions import RUN_PSO_CALIBRATION
from DAE_analysis_package.Plotting_functions import _ensure_dir, plot_corr_summary, plot_residuals_summary, plot_sensitivity_summary, plotting_single_solution, solution_comparison


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

def RUN_COMBOS(system_data: Dict[str, any],
                    simulate_model:callable,
                    folder_model: str,
                    folder_results: str,
                    solve_calibrate: str,
                    iterations: int = None,
                    parameters_and_bounds: Dict[str, List[float]] = None,
                    min_k: int = 1,
                    max_k: Optional[int] = None) -> pd.DataFrame:

    """
    Run all combinations of parameter calibrations and save results to a CSV file.
    """    
    # Create output directories and CSV file
    os.makedirs(folder_model, exist_ok=True)

    results_folder = folder_model +"_Results"
    csv_name = folder_model + "_" + folder_results + "_data.csv"

    calibration_results_path = os.path.join(os.getcwd(), folder_model, results_folder, folder_results)
    csv_path = os.path.join(os.getcwd(), folder_model, results_folder, folder_results, csv_name)

    parameters_og_list = system_data['parameters_og_list']
    var_exp_names = system_data['var_exp_names']

    # Columns for data export
    column_names = []
    columns_values = []
    columns_t_values = []
    for key in parameters_og_list:
        columns_values.append(key)
        columns_t_values.append('t_value_' + key)

    
    columns_var_validation = []
    for var in var_exp_names:
        columns_var_validation.append('RMSE_' + var)
        columns_var_validation.append('NMRSE_' + var)
        columns_var_validation.append('MAPE_' + var)

    column_names.extend(['Model', 'Param Combo', 'lb', 'ub'])
    column_names.extend(columns_values)
    column_names.extend(columns_t_values)
    column_names.extend(['Rank_Ratio', 'Highly_Corr'])
    column_names.extend(columns_var_validation)
    column_names.extend(['RMSE', 'NMRSE', 'MAPE', 'AIC', 'BIC','D-W','A-D'])

    if not os.path.exists(csv_path):
        init_csv_with_header(csv_path, column_names)


    # Print given initial parameters
    print(pd.DataFrame(system_data['parameters'], index=[0]).T.rename(columns={0: 'Original Parameters'}))
    print(" ")
    # Run original first (automatic with iteration 0)
    RUN_COMBINATION(system_data=system_data,
                        simulate_model=simulate_model,
                        model_id = [0,0],
                        iterations=None,
                        params_list=[],
                        lb=[],
                        ub=[],
                        calibration_results_path=calibration_results_path,
                        csv_path=csv_path,
                        combo_name='original_model',
                        column_names=column_names,
                        )
    
    if solve_calibrate == 'solve':
        return pd.read_csv(csv_path)
    else:
        all_params = list(parameters_and_bounds.keys())

        n = len(all_params)
        if max_k is None:
            max_k = n
        combos = []
        for k in range(min_k, max_k + 1):
            for combo in combinations(all_params, k):
                combos.append(list(combo))
        combos.reverse()  # Reverse the list to have larger combinations first


        model_id = [1,1]

        for combo in combos:
            # bounds in the order of this combo
            lb = [float(parameters_and_bounds[p][0]) for p in combo]
            ub = [float(parameters_and_bounds[p][1]) for p in combo]

            # per-combo name (folder-safe, sorted)
            cname = str(combo)

            # call the (slightly tweaked) single-combo runner
            RUN_COMBINATION(system_data=system_data,
                            model_id = model_id,
                                simulate_model=simulate_model,
                                iterations=iterations,
                                params_list=combo,
                                lb=lb,
                                ub=ub,
                                calibration_results_path=calibration_results_path,
                                csv_path=csv_path,
                                combo_name=cname,
                                column_names=column_names,)
            model_id[0] += 1

        # return the aggregated CSV
        return pd.read_csv(csv_path)


def RUN_COMBINATION(system_data: Dict[str, any],
                        simulate_model:callable, 
                        model_id: Any,
                        params_list: List[str], 
                        lb: List[float], 
                        ub: List[float], 
                        calibration_results_path: str,
                        csv_path: Optional[str],
                        combo_name: Optional[str],
                        column_names: List[str], 
                        iterations: int = None
                        ) -> pd.DataFrame:
    
    if iterations is None:
        run_calibration = False
        iterations = 1 # just run once for original model
        print("\n>>> Running original model validation...")
    else:
        run_calibration = True

        print(f"\nMODEL ID: {model_id}")
        print(f"\n>>> Calibrating parameters {params_list}, for {iterations} iterations.")
        print(f"    lb = {lb}")
        print(f"    ub = {ub}")

    
    system_data['calibrating_parameters'] = params_list
    
    # Prepare CSV path and figure output directory
    need_header = not os.path.exists(csv_path)
    if need_header:
        init_csv_with_header(csv_path, column_names)

    # Decide per-combo figures directory
    if combo_name is None:
        combo_name = str(params_list)

    fig_outdir = os.path.join(calibration_results_path, combo_name)
    _ensure_dir(fig_outdir)

    model_id[1] = 1

    model_summary_dict = {}

    model_summary_dict['Model'] = []
    model_summary_dict['Solution'] = []
    model_summary_dict['Residuals'] = []
    model_summary_dict['Sensitivity'] = []
    model_summary_dict['Correlation'] = []
    model_summary_dict['AIC'] = []
    
    # Iterate for the given number of iterations
    for iteration in range(iterations):  
        # Calibration step
        params_list = system_data['calibrating_parameters']
        if run_calibration:
            print(f"\n>>> Calibration iteration {iteration + 1} for parameters {params_list}")
            calibration_results_full = RUN_PSO_CALIBRATION(system_data=system_data,
                                                    simulate_model=simulate_model,
                                                    iteration=iteration,
                                                    lb=lb,
                                                    ub=ub)
            refined_parameters = calibration_results_full['RESULTS']

            


            parameter_values_results_to_row = calibration_results_full['FORMATTED_VALUES']
            row_data = [f"Model_{model_id[0]}_{iteration+1}", str(params_list), lb, ub]
        
        # If original, no calibration, just validation and parameter analysis
        else:           
            refined_parameters = system_data['parameters']
            parameter_values_results_to_row = [format_number(x,3) for x in refined_parameters.values()]
            row_data = [f"Original_Model", ' ', ' ', ' ']
        
        
        row_data.extend(parameter_values_results_to_row)

        validation_results = RUN_VALIDATION(system_data=system_data,
                                            simulate_model=simulate_model,
                                            fig_outdir=fig_outdir,
                                            parameters=refined_parameters,
                                            iteration=iteration,
                                            original=not run_calibration
                                            )
        
        if validation_results is None:
            print(">>> Validation failed, skipping to next iteration.")
            continue
        row_validation = validation_results['Row_val_analysis']
        system_sol = validation_results['System_sol']
        residuals = validation_results['Residuals']
        aic = validation_results['AIC']

        if not run_calibration:
            system_sol = []
        else:
            system_sol = validation_results['System_sol']
    
        plotting_single_solution(system_data=system_data,
                            fig_outdir = fig_outdir,
                            iteration=iteration,
                            parameters_updated =refined_parameters,
                            AIC = aic,
                            new_sol_v = system_sol,
                            original = not run_calibration
                            )

        parameter_analysis_results = PARAMETER_ANALYSIS(system_data=system_data,
                                                        simulate_model=simulate_model,
                                                        fig_outdir=fig_outdir,
                                                        iteration=iteration,
                                                        parameters=refined_parameters,
                                                        original=not run_calibration
                                                        )
        
        row_param_analysis = parameter_analysis_results['Row_param_analysis']
        sensitivity_analysis_results = parameter_analysis_results['Sensitivity_results']
        corr_matrix = parameter_analysis_results['Correlation_matrix']
        
        row_data.extend(row_param_analysis)
        row_data.extend(row_validation)

        model_summary_dict['Model'].append(f"Model_{model_id[0]}_{iteration+1}" if run_calibration else "Original_Model")
        model_summary_dict['Solution'].append(system_sol)
        model_summary_dict['Residuals'].append(residuals)
        model_summary_dict['Sensitivity'].append(sensitivity_analysis_results)
        model_summary_dict['Correlation'].append(corr_matrix)
        model_summary_dict['AIC'].append(aic)

        model_id[1] += 1

        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame([row_data], columns=column_names)
        df.to_csv(csv_path, mode="a", header=False, index=False, sep=",", decimal=".")
    # print("\n>>> Model Summary:")
    # print(model_summary_dict)

    # print(' ')
    # print(len(model_summary_dict['Model']))
    # print(len(model_summary_dict['Solution']))
    # print(len(model_summary_dict['Residuals']))
    # print(len(model_summary_dict['Sensitivity']))
    # print(len(model_summary_dict['Correlation']))
    # print(' ')
    if run_calibration:
        plot_sensitivity_summary(fig_outdir=fig_outdir,
                                model_summary_dict=model_summary_dict)
        plot_corr_summary(system_data=system_data,
                            fig_outdir=fig_outdir,
                            model_summary_dict=model_summary_dict)
        plot_residuals_summary(fig_outdir=fig_outdir,
                                model_summary_dict=model_summary_dict)
        solution_comparison(system_data=system_data,
                            fig_outdir=fig_outdir,
                            model_summary_dict=model_summary_dict)
        

    
    return pd.read_csv(csv_path)




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