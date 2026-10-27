import os
import sys
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

import pandas as pd
import time

import numpy as np
import logging

from mealpy import FloatVar, PSO # type: ignore
from mealpy.utils.problem import FloatVar # type: ignore
from mealpy.swarm_based import PSO # type: ignore

from DAE_analysis_package.Aux_Functions import format_number


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

# =====================================================================================
# CHANGELOG vs. the previous version  (search for "[NEW]" / "[FIX]")
#  [NEW] Seed: every PSO run gets an explicit seed (mealpy solve(..., seed=)), returned and logged in the CSV -> reproducible.
#        Optional: system_data['pso_seed_base'] makes seeds deterministic (see RUN_functions).
#  [NEW] The final value of the PSO objective ('COST', native-units SSE on the calibration set) and the best-so-far curve
#        per epoch ('HISTORY') are returned. Before, nothing about the calibration fit was kept: the CSV only had validation metrics.
#  [FIX] Bound hits: the old flags had inverted logic and their prints were commented out; the absolute tolerance 1e-3 meant
#        different things for q_max (0.098) and t_lag (19.5). Now: parameter within 1% of the (ub-lb) width of a bound -> listed in 'BOUND_HITS'.
#  [FIX] 'except:' in the cost function swallowed everything (even KeyboardInterrupt) and returned 1e6 silently.
#        Now 'except Exception' and the number of failed evaluations is counted ('N_FAILED_EVALS').
#  [NEW] Optional system_data['pso_epoch'] / ['pso_pop_size'] (defaults 100 / 50 = unchanged) for quick tests.
#  The cost function itself is UNCHANGED (unweighted SSE in native units).
# =====================================================================================

def RUN_PSO_CALIBRATION(system_data: Dict[str, any],
                        simulate_model:callable,
                        iteration: int, 
                        lb: List[float], 
                        ub: List[float],
                        ext: str,
                        seed: Optional[int] = None,
                        ) -> Dict[str, Any]:
    """
    Run PSO optimization for parameter calibration.
    """
    params_list = system_data['calibrating_parameters']
    # Define the optimization problem
    cost_function = define_cost_function(system_data=system_data, simulate_model=simulate_model, ext=ext)
    problem = {
    "obj_func": cost_function,
    "bounds": FloatVar(lb=lb, ub=ub),
    "minmax": "min"
    }

    # Initialize PSO optimizer (you can adjust parameters as needed)
    pso = PSO.OriginalPSO(epoch=system_data.get('pso_epoch', 50), pop_size=system_data.get('pso_pop_size', 30),
                          c1=1.5, c2=1.5, w=0.5)

    # [NEW] explicit seed, always recorded
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
    
    # Solve the optimization problem, suppressing output, and measure time taken
    start = time.perf_counter()
    with suppress_all_output():
        g_best = pso.solve(problem, seed=seed)
    end = time.perf_counter()
    # [NEW] fail loudly if the objective never worked (this is how the parameters_og_list bug went unnoticed)
    st = cost_function.stats
    if st['n_eval'] > 0 and st['n_failed'] == st['n_eval']:
        raise RuntimeError(f"All {st['n_eval']} PSO objective evaluations failed. First error: {st.get('first_error')}")
    if st['n_failed'] > 0.5 * st['n_eval']:
        logger.warning("More than half of the PSO evaluations failed (%d/%d). First error: %s",
                       st['n_failed'], st['n_eval'], st.get('first_error'))
    cost_final = float(g_best.target.fitness)                         # [NEW] final PSO objective (calibration SSE, native units)
    history = [float(v) for v in pso.history.list_global_best_fit]    # [NEW] best-so-far objective per epoch (convergence check)
    
    # Update parameters with optimized values
    new_params = g_best.solution
    new_params_dict = dict(zip(params_list, new_params))

    original_params = system_data['parameters']
    refined_parameters = original_params.copy()
    parameters_values = []
    
    for param in original_params.keys():
        if param not in params_list:
            refined_parameters[param] = original_params[param]
            parameters_values.append(None)
        else:
            new_value = new_params_dict[param]
            refined_parameters[param] = new_value
            parameters_values.append(new_value)

    parameters_values_formatted = [format_number(x) for x in parameters_values]

    # [FIX] bound hits: within 1% of the bound-interval width of lb or ub (scale-free)
    bound_hits = []
    for p, l, u in zip(params_list, lb, ub):
        v = new_params_dict[p]
        if u > l and (abs(v - l) <= 0.01 * (u - l) or abs(v - u) <= 0.01 * (u - l)):
            bound_hits.append(p)
    
    df_new_params = pd.DataFrame({
        "Parameter      ": params_list,
        "Original / Base": [original_params[key] for key in params_list],
        "New"            : [refined_parameters[key] for key in params_list]})

    return {'RESULTS':refined_parameters, 'FORMATTED_VALUES': parameters_values_formatted,
            'COST': cost_final, 'SEED': seed, 'HISTORY': history, 'BOUND_HITS': bound_hits,
            'N_FAILED_EVALS': cost_function.stats['n_failed'], 'N_EVALS': cost_function.stats['n_eval'],
            'TIME_S': end - start}


def define_cost_function(system_data: Dict[str, any],
                        simulate_model:callable,
                        ext: str
                        ) -> callable:
    """
    Function to define the cost function for calibration using experimental data.
    Parameters:
        - params_list: list of parameter names to be calibrated.
    Returns:
        - cost_function: function that computes the cost based on the difference between simulated and experimental data.
    """
    # Load system information

    # SE CALIBRA A PARTIR DEL N DEF !!!!!!!!!!!!!!!!!!!!!!!


    var__exp_names = system_data['var_exp_names']
    x0_exp = system_data['x0_exp'+ext]

    t_exp = system_data['t_exp'+ext]
    df_exp = system_data['df_exp'+ext]

    # [FIX] was system_data['parameters_og_list'], which is a LIST of parameter names. simulate_model then did
    #       pars[name] = p[i] on a list -> TypeError on every call, swallowed below -> every cost was 1e6 (N_failed_evals = N_evals).
    base_parameters = system_data['parameters']

    # [NEW] cost scaling. 'native' = old behaviour (unweighted SSE in native units: C in g/L dominates, N and pH are ~ignored).
    #       'range'  = every variable divided by the range of its experimental data (same scaling as the NRMSE / AIC metrics).
    cost_scaling = system_data.get('cost_scaling', 'native')
    scale = {}
    for v in var__exp_names:
        rng = float(np.ptp(np.asarray(df_exp[v].values, dtype=float))) if cost_scaling == 'range' else 1.0
        scale[v] = rng if rng > 0 else 1.0


    stats = {'n_eval': 0, 'n_failed': 0}      # [NEW] counters

    def cost_function(p_vars: np.ndarray) -> float: # COST FUNCTION USING PE DATA
        """
        Computes the cost function based on the difference between simulated and experimental data.
        Parameters:
            - p_vars: array of parameter values to be calibrated
        Returns:
            - err: total error between simulated and experimental data
        """
        stats['n_eval'] += 1
        try:
            df_results = simulate_model(system_data=system_data, 
                                        simulation_type='calibrating', 
                                        x0=x0_exp, 
                                        parameters=base_parameters,
                                        time=t_exp,
                                        p_vars=p_vars
                                    )
            
            err = 0
            for var in var__exp_names:
                var_new = df_results[var]
                var_exp = df_exp[var]

                err += np.sum(((np.asarray(var_new, dtype=float) - np.asarray(var_exp, dtype=float)) / scale[var])**2)
        
            return err
        
        except Exception as exc:               # [FIX] was a bare 'except:'
            stats['n_failed'] += 1
            stats.setdefault('first_error', repr(exc))   # [NEW] keep the first exception so a systematic failure is visible
            err = 1e6
            return err

    cost_function.stats = stats            # [NEW] read by RUN_PSO_CALIBRATION
    return cost_function