import os
import sys
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

import csv
import pandas as pd

from itertools import combinations

from DAE_analysis_package.Param_analysis_functions import PARAMETER_ANALYSIS
from DAE_analysis_package.Aux_Functions import format_number
from DAE_analysis_package.Validation_functions import RUN_VALIDATION
from DAE_analysis_package.Calibration_functions import RUN_PSO_CALIBRATION
from DAE_analysis_package.Plotting_functions import _ensure_dir, plot_corr_summary, plot_residuals_summary, plot_sensitivity_summary, plotting_single_solution, solution_comparison
from DAE_analysis_package.Best_models_functions import RUN_BEST_MODELS

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

import time

def _fmt_time(seconds: float) -> str:
    s = int(round(seconds)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec:02d}s"

def _progress_text(progress) -> str:
    if progress is None or progress['done'] == 0:
        return " | ETA: calculating..."
    el = time.time() - progress['t0']
    eta = el / progress['done'] * (progress['total'] - progress['done'])
    return f" | elapsed {_fmt_time(el)} | ETA {_fmt_time(eta)}"


def RUN_COMBOS(system_data: Dict[str, any],
                    simulate_model:callable,                    folder_model: str,
                    folder_results: str,
                    solve_calibrate: str,
                    iterations: int = None,
                    parameters_and_bounds: Dict[str, List[float]] = None,
                    min_k: int = 1,
                    max_k: Optional[int] = None,
                    baseline: Optional[Dict[str, any]] = None) -> pd.DataFrame:

    """
    Run all combinations of parameter calibrations and save results to a CSV file.
    """    
    # Create output directories and CSV file
    os.makedirs(folder_model, exist_ok=True)

    results_folder = folder_model +"_Results"
    csv_name_normal = "normal" + folder_model + "_" + folder_results + "_data.csv"
    csv_name_N_def = "N_def" + folder_model + "_" + folder_results + "_data.csv"

    folder_results_N_def = "N_def_" + folder_results
    folder_results_normal = "normal_" + folder_results

    calibration_results_path_N_def = os.path.join(os.getcwd(), folder_model, results_folder, folder_results_N_def)
    csv_path_N_def = os.path.join(os.getcwd(), folder_model, results_folder, folder_results_N_def, csv_name_N_def)

    calibration_results_path_normal = os.path.join(os.getcwd(), folder_model, results_folder, folder_results_normal)
    csv_path_normal = os.path.join(os.getcwd(), folder_model, results_folder, folder_results_normal, csv_name_normal)


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
        columns_var_validation.append('NRMSE_' + var)   # [FIX] was 'NMRSE_' (typo)
        columns_var_validation.append('MAPE_' + var)

    column_names.extend(['Model', 'Param Combo', 'lb', 'ub'])
    column_names.extend(columns_values)
    column_names.extend(columns_t_values)
    column_names.extend(['Rank_Ratio', 'Highly_Corr'])
    column_names.extend(columns_var_validation)
    # [FIX] 'NMRSE' -> 'NRMSE'.  [FIX] 'D-W','A-D' were SWAPPED: RUN_VALIDATION appends [Anderson-Darling, Durbin-Watson] in that order,
    # so the old column called 'D-W' actually held the A-D statistic and 'A-D' held D-W. (Old CSV/Excel files: swap the two columns.)
    column_names.extend(['RMSE', 'NRMSE', 'MAPE', 'AIC', 'BIC','A-D','D-W'])
    # [NEW] calibration-set metrics and PSO bookkeeping (before, the CSV only had VALIDATION metrics, so runs could not be
    # compared/selected on the data they were actually fitted to). Order must match 'row_extra' in RUN_COMBINATION.
    column_names.extend(['Cal_NRMSE_' + v for v in var_exp_names])
    column_names.extend(['Cal_NRMSE', 'Cal_AIC', 'Cal_BIC',
                         'Cal_cost',            # final PSO objective: SSE in native units on the calibration set
                         'PSO_seed', 'Bound_hits', 'N_failed_evals'])

    # [WARNING] if csv_path already exists the header is NOT rewritten and new rows are APPENDED to the old file.
    # After changing the analysis code (t-values, correlations, sensitivities) use a NEW folder_results name or delete the old CSV,
    # otherwise old and new rows get mixed.
    if not os.path.exists(csv_path_normal):
        init_csv_with_header(csv_path_normal, column_names)

    if not os.path.exists(csv_path_N_def):
        init_csv_with_header(csv_path_N_def, column_names)


    # Print given initial parameters

    # Run original first (automatic with iteration 0)
    RUN_COMBINATION(system_data=system_data,
                        simulate_model=simulate_model,
                        model_id = [0,0],
                        iterations=None,
                        params_list=[],
                        lb=[],
                        ub=[],
                        calibration_results_path_normal=calibration_results_path_normal,
                        calibration_results_path_N_def=calibration_results_path_N_def,
                        csv_path_normal=csv_path_normal,
                        csv_path_N_def=csv_path_N_def,
                        combo_name='original_model',
                        column_names=column_names
                        )
    
    if solve_calibrate == 'solve':
        return pd.read_csv(csv_path_normal), pd.read_csv(csv_path_N_def)
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
        progress = {'t0': time.time(), 'done': 0, 'total': len(combos) * iterations}

        
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
                                calibration_results_path_normal=calibration_results_path_normal,
                                csv_path_normal=csv_path_normal,
                                calibration_results_path_N_def=calibration_results_path_N_def,
                                csv_path_N_def=csv_path_N_def,
                                combo_name=cname,
                                column_names=column_names,
                                n_combos=len(combos),
                                progress=progress
                                )
            model_id[0] += 1

        # return the aggregated CSV
                # [NEW] top-10 % analysis (both rankings: calibration and validation NRMSE)
        RUN_BEST_MODELS(system_data=system_data,
                        simulate_model=simulate_model,
                        csv_path_normal=csv_path_normal,
                        csv_path_N_def=csv_path_N_def,
                        outdir=os.path.join(os.getcwd(), folder_model, results_folder, "Best_models_" + folder_results),
                        top_frac=0.10,
                        rank_on='N_def',
                        baseline=baseline)

        # return the aggregated CSV
        return pd.read_csv(csv_path_normal), pd.read_csv(csv_path_N_def)


def RUN_COMBINATION(system_data: Dict[str, any],
                        simulate_model:callable,
                        model_id: Any,
                        params_list: List[str], 
                        lb: List[float], 
                        ub: List[float], 
                        calibration_results_path_normal: str,
                        csv_path_normal: Optional[str],
                        calibration_results_path_N_def: str,
                        csv_path_N_def: Optional[str],
                        combo_name: Optional[str],
                        column_names: List[str], 
                        n_combos: int = None,
                        iterations: int = None,
                        progress: Optional[dict] = None
                        ) -> pd.DataFrame:
    
    if iterations is None:
        run_calibration = False
        iterations = 1 # just run once for original model
        progress = {'t0': time.time(), 'done': 0, 'total': iterations}
    else:
        run_calibration = True

    
    system_data['calibrating_parameters'] = params_list
    
    # Prepare CSV path and figure output directory
    need_header = not os.path.exists(csv_path_normal)
    if need_header:
        init_csv_with_header(csv_path_normal, column_names)

    need_header_N_def = not os.path.exists(csv_path_N_def)
    if need_header_N_def:
        init_csv_with_header(csv_path_N_def, column_names)

    # Decide per-combo figures directory
    if combo_name is None:
        combo_name = str(params_list)

    fig_outdir_normal = os.path.join(calibration_results_path_normal, combo_name)
    fig_outdir_N_def = os.path.join(calibration_results_path_N_def, combo_name)
    _ensure_dir(fig_outdir_normal)
    _ensure_dir(fig_outdir_N_def)
    model_id[1] = 1

    model_summary_dict_normal = {}

    model_summary_dict_normal['Model'] = []
    model_summary_dict_normal['Solution'] = []
    model_summary_dict_normal['Residuals'] = []
    model_summary_dict_normal['Sensitivity'] = []
    model_summary_dict_normal['Correlation'] = []
    model_summary_dict_normal['AIC'] = []



    model_summary_dict_N_def = {}
    model_summary_dict_N_def['Model'] = []
    model_summary_dict_N_def['Solution'] = []
    model_summary_dict_N_def['Residuals'] = []
    model_summary_dict_N_def['Sensitivity'] = []
    model_summary_dict_N_def['Correlation'] = []
    model_summary_dict_N_def['AIC'] = []
    
    # Iterate for the given number of iterations
    for iteration in range(iterations):
        if run_calibration:
            tot = f"/{n_combos}" if n_combos else ""
            print(f">>> Model {model_id[0]}{tot} | iteration {iteration+1}/{iterations}" + _progress_text(progress), flush=True)
            progress['done'] += 1
        else:
            print(">>> Original model (no calibration)", flush=True)

        # Calibration step
        params_list = system_data['calibrating_parameters']
        if run_calibration:
            # [NEW] explicit seed per run. If system_data['pso_seed_base'] is defined the seeds are deterministic
            # (re-running reproduces the same candidates); otherwise random, but always recorded in the CSV.
            seed_base = system_data.get('pso_seed_base', None)
            seed = None if seed_base is None else int(seed_base) + 1000 * int(model_id[0]) + iteration
            calibration_results_full = RUN_PSO_CALIBRATION(system_data=system_data,
                                                    simulate_model=simulate_model,
                                                    iteration=iteration,
                                                    lb=lb,
                                                    ub=ub,
                                                    ext="_N_def",
                                                    seed=seed)
            refined_parameters = calibration_results_full['RESULTS']

            


            parameter_values_results_to_row = calibration_results_full['FORMATTED_VALUES']
            row_data_normal = [f"normal_Model_{model_id[0]}_{iteration+1}", str(params_list), lb, ub]
            row_data_N_def = [f"N_def_Model_{model_id[0]}_{iteration+1}", str(params_list), lb, ub]
        
        
        # If original, no calibration, just validation and parameter analysis
        else:           
            refined_parameters = system_data['parameters']
            parameter_values_results_to_row = [format_number(x,3) for x in refined_parameters.values()]
            row_data_normal = [f"normal_Model_{model_id[0]}_{iteration+1}", str(params_list), lb, ub]
            row_data_N_def = [f"N_def_Model_{model_id[0]}_{iteration+1}", str(params_list), lb, ub]
        
        
        row_data_normal.extend(parameter_values_results_to_row)
        row_data_N_def.extend(parameter_values_results_to_row)

        
# Run validation and parameter analysis for the normal model
        validation_results_normal = RUN_VALIDATION(system_data=system_data,
                                            simulate_model=simulate_model,
                                            fig_outdir=fig_outdir_normal,
                                            parameters=refined_parameters,
                                            iteration=iteration,
                                            ext='_normal',
                                            original=not run_calibration
                                            )
        
        if validation_results_normal is None:
            print(">>> Validation failed, skipping to next iteration.")
            continue
        row_validation_normal = validation_results_normal['Row_val_analysis']
        system_sol_normal = validation_results_normal['System_sol']
        residuals_normal = validation_results_normal['Residuals']
        aic_normal = validation_results_normal['AIC']

        if not run_calibration:
            system_sol_normal = []
        else:
            system_sol_normal = validation_results_normal['System_sol']

        plotting_single_solution(system_data=system_data,
                            fig_outdir = fig_outdir_normal,
                            iteration=iteration,
                            parameters_updated =refined_parameters,
                            AIC = aic_normal,
                            new_sol_v = system_sol_normal,
                            original = not run_calibration,
                            ext='_normal'
                            )

        parameter_analysis_results_normal = PARAMETER_ANALYSIS(system_data=system_data,
                                                        simulate_model=simulate_model,
                                                        fig_outdir=fig_outdir_normal,
                                                        iteration=iteration,
                                                        parameters=refined_parameters,
                                                        ext='_normal',
                                                        original=not run_calibration
                                                        )
        
        row_param_analysis_normal = parameter_analysis_results_normal['Row_param_analysis']
        sensitivity_analysis_results_normal = parameter_analysis_results_normal['Sensitivity_results']
        corr_matrix_normal = parameter_analysis_results_normal['Correlation_matrix']
        
        row_data_normal.extend(row_param_analysis_normal)
        row_data_normal.extend(row_validation_normal)

        # [NEW] the SAME metrics on the CALIBRATION experiment (dataset='cal': no residual plots, does not overwrite validation figures)
        cal_results = RUN_VALIDATION(system_data=system_data,
                                     simulate_model=simulate_model,
                                     fig_outdir=fig_outdir_normal,
                                     parameters=refined_parameters,
                                     iteration=iteration,
                                     ext='_normal',
                                     original=not run_calibration,
                                     dataset='cal')
        var_exp_names = system_data['var_exp_names']
        cm = cal_results['Metrics'] if cal_results is not None else {}
        row_extra = [format_number(cm.get('NRMSE_' + v)) for v in var_exp_names]
        row_extra += [format_number(cm.get('NRMSE_global')), format_number(cm.get('AIC')), format_number(cm.get('BIC'))]
        if run_calibration:
            row_extra += [format_number(calibration_results_full['COST']), calibration_results_full['SEED'],
                          ';'.join(calibration_results_full['BOUND_HITS']), calibration_results_full['N_FAILED_EVALS']]
            # [NEW] best-so-far PSO objective per epoch -> side file (convergence check; ~100 numbers per run)
            hist_path = csv_path_normal.replace('_data.csv', '_pso_history.csv')
            n_ep = len(calibration_results_full['HISTORY'])
            if not os.path.exists(hist_path):
                with open(hist_path, 'w', newline='') as fh:
                    csv.writer(fh).writerow(['Model', 'Seed'] + [f'epoch_{i+1}' for i in range(n_ep)])
            with open(hist_path, 'a', newline='') as fh:
                csv.writer(fh).writerow([f"Model_{model_id[0]}_{iteration+1}", calibration_results_full['SEED']]
                                        + calibration_results_full['HISTORY'])
        else:
            row_extra += [None, None, None, None]     # original model: nothing was calibrated here
        row_data_normal.extend(row_extra)

        model_summary_dict_normal['Model'].append(f"normal_Model_{model_id[0]}_{iteration+1}" if run_calibration else "Original_Model")
        model_summary_dict_normal['Solution'].append(system_sol_normal)
        model_summary_dict_normal['Residuals'].append(residuals_normal)
        model_summary_dict_normal['Sensitivity'].append(sensitivity_analysis_results_normal)
        model_summary_dict_normal['Correlation'].append(corr_matrix_normal)
        model_summary_dict_normal['AIC'].append(aic_normal)



# Run validation for the n def
        validation_results_N_def = RUN_VALIDATION(system_data=system_data,
                                                    simulate_model=simulate_model,
                                                    fig_outdir=fig_outdir_N_def,
                                                    parameters=refined_parameters,
                                                    iteration=iteration,
                                                    ext='_N_def',
                                                    original=not run_calibration
                                                    )
        
        if validation_results_N_def is None:
            print(">>> Validation failed, skipping to next iteration.")
            continue
        row_validation_N_def = validation_results_N_def['Row_val_analysis']
        system_sol_N_def = validation_results_N_def['System_sol']
        residuals_N_def = validation_results_N_def['Residuals']
        aic_N_def = validation_results_N_def['AIC']

        if not run_calibration:
            system_sol_N_def = []
        else:
            system_sol_N_def = validation_results_N_def['System_sol']

        plotting_single_solution(system_data=system_data,
                            fig_outdir = fig_outdir_N_def,
                            iteration=iteration,
                            parameters_updated =refined_parameters,
                            AIC = aic_N_def,
                            new_sol_v = system_sol_N_def,
                            original = not run_calibration, 
                            ext='_N_def'
                            )

        parameter_analysis_results_N_def = PARAMETER_ANALYSIS(system_data=system_data,
                                                        simulate_model=simulate_model,
                                                        fig_outdir=fig_outdir_N_def,
                                                        iteration=iteration,
                                                        parameters=refined_parameters,
                                                        ext='_N_def',
                                                        original=not run_calibration
                                                        )
        
        row_param_analysis_N_def = parameter_analysis_results_N_def['Row_param_analysis']
        sensitivity_analysis_results_N_def = parameter_analysis_results_N_def['Sensitivity_results']
        corr_matrix_N_def = parameter_analysis_results_N_def['Correlation_matrix']

        row_data_N_def.extend(row_param_analysis_N_def)
        row_data_N_def.extend(row_validation_N_def)

        # [NEW] the SAME metrics on the CALIBRATION experiment (dataset='cal': no residual plots, does not overwrite validation figures)
        cal_results = RUN_VALIDATION(system_data=system_data,
                                     simulate_model=simulate_model,
                                     fig_outdir=fig_outdir_N_def,
                                     parameters=refined_parameters,
                                     iteration=iteration,
                                     ext='_N_def',
                                     original=not run_calibration,
                                     dataset='cal')
        var_exp_names = system_data['var_exp_names']
        cm = cal_results['Metrics'] if cal_results is not None else {}
        row_extra = [format_number(cm.get('NRMSE_' + v)) for v in var_exp_names]
        row_extra += [format_number(cm.get('NRMSE_global')), format_number(cm.get('AIC')), format_number(cm.get('BIC'))]
        if run_calibration:
            row_extra += [format_number(calibration_results_full['COST']), calibration_results_full['SEED'],
                          ';'.join(calibration_results_full['BOUND_HITS']), calibration_results_full['N_FAILED_EVALS']]
            # [NEW] best-so-far PSO objective per epoch -> side file (convergence check; ~100 numbers per run)
            hist_path = csv_path_N_def.replace('_data.csv', '_pso_history.csv')
            n_ep = len(calibration_results_full['HISTORY'])
            if not os.path.exists(hist_path):
                with open(hist_path, 'w', newline='') as fh:
                    csv.writer(fh).writerow(['Model', 'Seed'] + [f'epoch_{i+1}' for i in range(n_ep)])
            with open(hist_path, 'a', newline='') as fh:
                csv.writer(fh).writerow([f"N_def_Model_{model_id[0]}_{iteration+1}", calibration_results_full['SEED']]
                                        + calibration_results_full['HISTORY'])
        else:
            row_extra += [None, None, None, None]     # original model: nothing was calibrated here
        row_data_N_def.extend(row_extra)

        model_summary_dict_N_def['Model'].append(f"Model_{model_id[0]}_{iteration+1}" if run_calibration else "Original_Model")
        model_summary_dict_N_def['Solution'].append(system_sol_N_def)
        model_summary_dict_N_def['Residuals'].append(residuals_N_def)
        model_summary_dict_N_def['Sensitivity'].append(sensitivity_analysis_results_N_def)
        model_summary_dict_N_def['Correlation'].append(corr_matrix_N_def)
        model_summary_dict_N_def['AIC'].append(aic_N_def)

        model_id[1] += 1

        os.makedirs(os.path.dirname(csv_path_normal), exist_ok=True)
        os.makedirs(os.path.dirname(csv_path_N_def), exist_ok=True)
        
        df_normal = pd.DataFrame([row_data_normal], columns=column_names)
        df_N_def = pd.DataFrame([row_data_N_def], columns=column_names)
        df_normal.to_csv(csv_path_normal, mode="a", header=False, index=False, sep=",", decimal=".")
        df_N_def.to_csv(csv_path_N_def, mode="a", header=False, index=False, sep=",", decimal=".")

    if run_calibration:
        print(f">>> Model {model_id[0]} | comparison of iterations (summary plots)", flush=True)
        
        
        plot_sensitivity_summary(fig_outdir=fig_outdir_normal,
                                model_summary_dict=model_summary_dict_normal)
        plot_corr_summary(system_data=system_data,
                            fig_outdir=fig_outdir_normal,
                            model_summary_dict=model_summary_dict_normal)
        plot_residuals_summary(fig_outdir=fig_outdir_normal,
                                model_summary_dict=model_summary_dict_normal)
        solution_comparison(system_data=system_data,
                            fig_outdir=fig_outdir_normal,
                            model_summary_dict=model_summary_dict_normal, 
                            ext='_normal')



        plot_sensitivity_summary(fig_outdir=fig_outdir_N_def,
                                        model_summary_dict=model_summary_dict_N_def)
        plot_corr_summary(system_data=system_data,
                            fig_outdir=fig_outdir_N_def,
                            model_summary_dict=model_summary_dict_N_def)
        plot_residuals_summary(fig_outdir=fig_outdir_N_def,
                                model_summary_dict=model_summary_dict_N_def)
        solution_comparison(system_data=system_data,
                            fig_outdir=fig_outdir_N_def,
                            model_summary_dict=model_summary_dict_N_def, 
                            ext='_N_def')

    
        
        return pd.read_csv(csv_path_normal), pd.read_csv(csv_path_N_def)




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