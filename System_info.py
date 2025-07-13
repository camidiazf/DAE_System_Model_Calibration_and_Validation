import numpy as np
import pandas as pd

""" 
System information for DAE system model calibration and validation.
This module contains the system parameters, initial conditions, experimental data,
and other constants required for the DAE system simulations.
"""


# Variable names
var_names = ['X', 'C', 'N', 'pH']

colors = {
    'X': '#66C2A6',
    'C': '#FD8D62',
    'N': '#8DA0CB',
    'CO2': '#FED92F',
    'O2': '#A7D854',
    'pH': '#E78AC3',
    'mu': '#B3B3B3'
    } # Not necessary for the simulation, but useful for plotting

# Simulation time parameters
tf = 50
n_steps = 500
time_stamps_sim = np.linspace(0, tf, n_steps + 1)

# Original parameters
parameters = {
    'k_C': 0.08, 
    'k_N': 0.01, 
    'k_O' : 0.001, 
    'k_d': 0.001, 
    'YX_CO2': 0.35, 
    'YX_O2' : 1.352, 
    'YX_C': 0.484,
    'YX_N': 21.575,
    'Xmax': 1.4462,  
    'mu_max': 0.15,
    'pH_LL': 4.6, 
    'pH_UL': 7.4,
    'I_val': 3, 
    'O2_sat' : 0.007267,
    } 

parameters_og_list = list(parameters.keys()) 

# Constants for the system
constants = {'pka1': 6.86, # pKa of KH2PO4
            'pka2': 3.13,  # pka of C6H8O7
            'pka3': 4.76,  # pka of (C6H7O7)-
            'pka4': 6.40,  # pka of (C6H6O7)2-
            'pka5': 9.25,  # pka of NH3
            'pka6': 14.15, # pka of C3H8O3
            'pka7': 6.35,  # pka of CO2
            'pka8': 10.33, # pka of (HCO3)-
            'pka9': 14,    # pka of H2O
            'KH2PO4': 2.18,
            'C6H8O7': 2,
            'pH_alk': 7.2,
            't_lag': 7,
            'k_La' : 86.26}

# Experimental data
df_exp = pd.read_excel('Experimental_data.xlsx', sheet_name='PE_Normal')
df_val = pd.read_excel('Experimental_data.xlsx', sheet_name='V_Normal')

# Initial conditions simulation PE (PARAMETER ESTIMATION)
X0 = 0.229
c0 = 5.389
n0 = 0.951
co20 = 0.439 / 1000
# o20 = parameters['O2_sat']
o20 = 0.0001
z0 =7.2
x0_sim = np.array([X0, c0, n0, co20, o20, z0])

# Initial conditions simulation VAL (PARAMETER VALIDATION
X0 = 0.223
c0 = 5.992
n0 = 1.027
co20 = 0.439 / 1000
# o20 = parameters['O2_sat']
o20 = 0.0001
z0 = 7.2
x0_sim_v = np.array([X0, c0, n0, co20, o20, z0])

# Initial points experimental PE

X0 = df_exp['X'][0]  
c0 = df_exp['C'][0] 
n0 = df_exp['N'][0] 
co20 = df_exp['CO2'][0]  # CO2 concentration in g/L
o20 = parameters['O2_sat']  # O2 saturation in g/L
z0 = df_exp['pH'][0]  
x0_exp = np.array([X0, c0, n0, co20, o20, z0]) 

# Experimental time parameters PE
t_exp = df_exp['Time (Hours)']

# Initial points experimental VAL
X0_v = df_val['X'][0]  
c0_v = df_val['C'][0] 
n0_v = df_val['N'][0] 
co20_v = 0.439 / 1000
# o20_v = parameters['O2_sat']  # O2 saturation in g/L
o20_v = 0.0001
z0_v = df_val['pH'][0]  
x0_exp_v = np.array([X0_v, c0_v, n0_v, co20_v, o20_v, z0_v]) 


# Weights for experimental data
weights_exp_stack = np.vstack([
                    df_exp.iloc[:, 2],
                    df_exp.iloc[:, 5],
                    df_exp.iloc[:, 8],
                    df_exp.iloc[:, 11]
                ]).T


# Experimental time parameters PE
t_exp_v = df_exp['Time (Hours)']


# Calibration DATA

delta = 1e-4
correlation_threshold = 0.95
perturbation = 0.10


# System information dictionary
system_info = {
    'var_names': var_names,
    'parameters': parameters,
    'parameters_og_list': parameters_og_list,
    'colors': colors,
    'constants': constants,
    'x0_sim': x0_sim,
    'x0_exp': x0_exp,
    'x0_sim_v': x0_sim_v,
    'x0_exp_v': x0_exp_v,
    'time_stamps_sim': time_stamps_sim,
    'df_exp': df_exp,
    'df_val': df_val,
    't_exp': t_exp,
    't_exp_v': t_exp_v,
    'weights_exp_stack': weights_exp_stack,
    'delta': delta,
    'correlation_threshold': correlation_threshold,
    'perturbation': perturbation

}
