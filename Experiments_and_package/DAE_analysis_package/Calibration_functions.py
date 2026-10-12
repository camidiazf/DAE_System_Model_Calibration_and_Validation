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

def RUN_PSO_CALIBRATION(system_data: Dict[str, any],
                        simulate_model:callable,
                        iteration: int, 
                        lb: List[float], 
                        ub: List[float], 
                        ) -> Dict[str, Any]:
    """
    Run PSO optimization for parameter calibration.
    """
    params_list = system_data['calibrating_parameters']
    print(f">>> Running PSO calibration iteration {iteration + 1}...")
    # Define the optimization problem
    problem = {
    "obj_func": define_cost_function(system_data=system_data, simulate_model=simulate_model),
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
    
    print("Optimization Results:")
    print("     Best Solutions: ", g_best.solution)
    print("     Minimum Error: ", g_best.target.fitness)
    print(f"     Optimization Time: {end - start:.2f} s")

    # Update parameters with optimized values
    new_params = g_best.solution
    new_params_dict = dict(zip(params_list, new_params))

    original_params = system_data['parameters']
    refined_parameters = original_params.copy()
    parameters_values = []
    
    i_param = 0
    for param in original_params.keys():
        if param not in params_list:
            refined_parameters[param] = original_params[param]
            parameters_values.append(None)
        else:
            new_value = new_params_dict[param]
            refined_parameters[param] = new_value
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
    
    print("\nNew Parameters after PSO Calibration:")
    df_new_params = pd.DataFrame({
        "Parameter      ": params_list,
        "Original / Base": [original_params[key] for key in params_list],
        "New"            : [refined_parameters[key] for key in params_list]})
    print(df_new_params)
    print(" ")

    return {'RESULTS':refined_parameters, 'FORMATTED_VALUES': parameters_values_formatted}


def define_cost_function(system_data: Dict[str, any],
                        simulate_model:callable,
                        ) -> callable:
    """
    Function to define the cost function for calibration using experimental data.
    Parameters:
        - params_list: list of parameter names to be calibrated.
    Returns:
        - cost_function: function that computes the cost based on the difference between simulated and experimental data.
    """
    # Load system information
    var__exp_names = system_data['var_exp_names']
    x0_exp = system_data['x0_exp']

    t_exp = system_data['t_exp']
    df_exp = system_data['df_exp']

    base_parameters = system_data['parameters_og_list']


    def cost_function(p_vars: np.ndarray) -> float: # COST FUNCTION USING PE DATA
        """
        Computes the cost function based on the difference between simulated and experimental data.
        Parameters:
            - p_vars: array of parameter values to be calibrated
        Returns:
            - err: total error between simulated and experimental data
        """
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

                err += np.sum((var_new - var_exp)**2)
        
            return err
        
        except:
            err = 1e6
            return err
        
    return cost_function