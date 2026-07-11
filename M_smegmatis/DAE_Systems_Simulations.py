import os
from typing import Dict, List, Optional

import casadi as ca # type: ignore
import numpy as np
import pandas as pd
from mealpy.swarm_based import PSO # type: ignore

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

# Growth dynamics function
def DAE_system(system_data: Dict[str, any], t: float, x: np.ndarray, z: np.ndarray, params: Dict[str, float]) -> ca.MX:
    """
    Function to define the DAE system for the growth dynamics.
    Parameters:
        - t: time variable.
        - x: state variables.
        - z: algebraic variables.
        - params: dictionary of parameters.
    Returns:
        - dXdt: vector of differential equations.
    """

    constants = system_data['constants']
    
    # State variables
    X, C, N, CO2 = x[0], x[1], x[2], x[3]

    # Algebraic variable
    pH = z[0] 

    # Explicit algebraic equations

    # pH inhibition factor
    Iph = ca.exp((params['I_val'] * ((pH - params['pH_UL']) / (params['pH_UL'] - params['pH_LL']))) ** 2)

    # Specific growth rate
    mu = (params['mu_max'] 
          * (1 - ca.exp(-t / params['t_lag'])) 
          * (C / (C + params['k_C'])) 
          * (N / (N + params['k_N'])) 
          * (1 - (X / (params['Xmax']))) 
          * Iph)
    
    ka7 = 10 ** (-constants['pka7'])

    # Differential equations
    dXdt   = (mu- params['k_d']) * X                                                    # Biomass
    dCdt   = - (mu / params['YX_C']) * X                                                # Glycerol
    dNdt   = - (mu / params['YX_N']) * X                                                # Ammonia
    dCO2dt = ((mu / params['YX_CO2']) * X) - ka7 * (CO2 / (((10 ** -pH) / ka7) + 1))    # CO2

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt)

def DAE_system_calibrating(system_data: Dict[str, any], t: float, x: np.ndarray, z: np.ndarray, p: np.ndarray, parameters: Dict[str, float]) -> ca.MX:
    """
    Function to define the DAE system for calibration.
    """
    
    # Extract system constants
    constants = system_data['constants']
    param_list = system_data['calibrating_parameters']
    
    # Parameters to calibrate as variables
    pars = parameters.copy()
    for i, name in enumerate(param_list):
        pars[name] = p[i]

    # State variables and algebraic variable
    X, C, N, CO2= x[0], x[1], x[2], x[3]
    pH = z[0] 

    # Explicit algebraic equations

    # pH inhibition factor
    Iph = ca.exp((pars['I_val'] * ((pH - pars['pH_UL']) / (pars['pH_UL'] - pars['pH_LL']))) ** 2)  

    # Specific growth rate
    mu = (pars['mu_max'] 
        * (1 - ca.exp(-t / pars['t_lag']))   
        * (C / (C + pars['k_C']))   
        * (N / (N + pars['k_N']))   
        * (1 - (X / (pars['Xmax']))) 
        * Iph)

    ka7 = 10 ** (-constants['pka7'])

    # Differential equations
    dXdt   = (mu - pars['k_d']) * X                                                    # Biomass
    dCdt   = - (mu / pars['YX_C']) * X                                                 # Glycerol
    dNdt   = - (mu / pars['YX_N']) * X                                                 # Ammonia
    dCO2dt = ((mu / pars['YX_CO2']) * X) - ka7 * (CO2 / (((10 ** -pH) / ka7) + 1))     # CO2

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt)

def simulate_model(system_data: Dict[str, any], simulation_type: str, x0: np.ndarray, parameters: Dict[str, float], 
                    time: np.ndarray, p_vars: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
    """
    Function to simulate the DAE system.
    Simulation can be of type 'calibrating' or 'normal'.
    If 'calibrating', p_vars and param_list must be provided.
    Returns a DataFrame with the simulation results.
    """
    # Extract system constants
    constants = system_data['constants']

    # Symbolic variables
    t = ca.MX.sym('t')                      # time
    x = ca.MX.sym('x', 4)      # State variables [X, C, N, CO2]
    z = ca.MX.sym('z')                      # Algebraic variable [pH]

    # Systems's differential equations
    if simulation_type == 'calibrating':
        param_list = system_data['calibrating_parameters']
        p = ca.MX.sym('p', len(param_list))
        dxdt = DAE_system_calibrating(system_data, t, x, z, p, parameters)
    else:
        p = ca.MX.sym('p', 0) 
        dxdt = DAE_system(system_data, t, x, z, parameters)

    # Algebraic equations

    ka1 = 10 ** (-constants['pka1'])  # KH2PO4
    ka2 = 10 ** (-constants['pka2'])  # C6H8O7
    ka3 = 10 ** (-constants['pka3'])  # (C6H7O7)-
    ka4 = 10 ** (-constants['pka4'])  # (C6H6O7)2-
    ka7 = 10 ** (-constants['pka7'])  # CO2
    ka9 = 10 ** (-constants['pka9'])  # H2O

    H = 10 ** (-z) 

    KHPO4 = constants['KH2PO4'] / ((H / ka1) + 1)
    C6H5O7 = constants['C6H8O7'] / ((H ** 3 / (ka2 * ka3 * ka4)) + (H ** 2 / (ka3 * ka4)) + (H / ka4) + 1)
    C6H6O7 = (H / ka4) * C6H5O7
    C6H7O7 = (H / ka3) * C6H6O7
    HCO3 = x[3] / ((H / ka7) + 1)
    OH = ka9 / H

    f_z = OH + HCO3 + KHPO4 + (3 * C6H5O7) + (2 * C6H6O7) + C6H7O7 - constants['pH_alk'] - H

    # CasADi function
    f = ca.Function('f', [t, x, z], [dxdt])
    
    # ODE system
    dae = {'t': t, 'x': x, 'z': z, 'p': p, 'ode': dxdt, 'alg': f_z}
    integrator = ca.integrator('F', 'idas', dae, {'grid': time, 'output_t0': True})

    # Solve
    try:
        if simulation_type == 'calibrating':
            sol = integrator(x0=x0[:-1], z0=x0[-1], p=p_vars)   
        else:
            sol = integrator(x0=x0[:-1], z0=x0[-1])
    except RuntimeError as e:
        print("Integration failed:", e)
        return None

    dict_results = {}

    # Extract results
    t = time
    x = sol['xf'].full().T
    z = sol['zf'].full().T

    X = x[:, 0]
    C = x[:, 1]
    N = x[:, 2]
    CO2 = x[:, 3]
    pH = z[:, 0]
    H = 10 ** (-pH)
    # Compute specific growth rate over time
    mu_values = np.zeros_like(C)
    for i in range(len(C)):
        mu_values[i] = (parameters['mu_max'] *
            (1 - np.exp(-t[i] / parameters['t_lag'])) *
            (C[i] / (C[i] + parameters['k_C'])) *
            (N[i] / (N[i] + parameters['k_N'])) *
            np.exp((parameters['I_val'] *
                    (pH[i] - parameters['pH_UL']) /
                (parameters['pH_UL'] - parameters['pH_LL'])) ** 2) *
            (1 - X[i] / parameters['Xmax']))
        
    dict_results['t'] = t
    dict_results['X'] = X
    dict_results['C'] = C
    dict_results['N'] = N
    dict_results['CO2'] = CO2
    dict_results['pH'] = pH
    dict_results['H'] = H
    dict_results['mu_values'] = mu_values

    df_results = pd.DataFrame(dict_results)
    return df_results