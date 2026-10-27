import os
import numpy as np
import pandas as pd

""" 
System information for DAE system model calibration and validation.
This module contains the system parameters, initial conditions, experimental data,
and other constants required for the DAE system simulations.
It also includes settings for parameter estimation and validation.
As well as configurations for saving results and plotting.

Regarding 'Experimental_data.xlsx', ensure it is placed in the same directory as this script.
Recommended to use this Excel format for ease of data management according to this script.
If using a different format, modify the data import section accordingly.

"""

# Folder name of the Model you will be working with
normal_model_folder_name = 'M_smegmatis_extended_replete'
N_def_model_folder_name = 'M_smegmatis_extended_N_def'

# Variable names for which we have experimental data (in this case there is no CO2 or O2 data)

var_names = ['X', 'C', 'N', 'pH', 'O', 'CO2', 'q']  #All variables

var_exp_names = ['X', 'C', 'N', 'pH']  # Variables for which we have experimental data

# Original parameters (17 parameters) for the system model
parameters = {
    'k_C': 0.08, 
    'k_O' : 0.001, 
    'k_d': 0.001, 
    'YX_CO2': 0.35, 
    'YX_O2' : 1.352, 
    'YX_C': 0.4840,
    'Xmax': 1.1537,  
    'mu_max': 0.1335, ## 0.3
    't_lag': 7,
    'm_C':     0.008,   # glycerol maintenance coeff [g gly / gX / h]  ← the decoupler
    'f_Nrec':  0.5,     # fraction of decay-N recycled to medium (ammonification)
    'pH_opt':  7,     # pH optimum for growth  (was implicitly pH_UL)
    'pH_w':    7,     # pH inhibition width; ≈ (pH_UL − pH_LL)/2
    'Y_CO2_m': 0.05,     # CO2 produced per unit maintenance glycerol (stoich.)
    'Y_O2_m':  1.0,     # O2 consumed per unit maintenance glycerol (stoich.),
    'rho_N_max': 0.5,   # max specific N UPTAKE rate          [g N / gX / h]
    'k_Nup':    0.0001,   # half-saturation for N uptake         [g N / L]
    'q_min':    0.0001,   # subsistence N quota (growth floor)   [g N / gX],
    'q_max':     0.11,   # max N quota (storage ceiling)        [g N / gX]
    'n_cap':     1,     # Empirical exponent constant that shapes the sharpness or curvature of the inhibition decline.
    'm_cap':     1     # Empirical exponent constant that shapes the sharpness or curvature of the maintenance decline.
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
            'O2_sat' : 7.2673/1000,
            'k_La' : 86.26,
            'M_N': 14.007,  # g/mol, for the NH4+ term in the pH balance
            }

# Initial conditions simulation PE (PARAMETER ESTIMATION)
X0_normal = 0.229
c0_normal = 5.389
n0_normal = 0.951
co20_normal = 5.931692e-02
# o20 = parameters['O2_sat']
o20_normal = 0.0001
q0_normal = 0.096
z0_normal =7.2
x0_sim_normal = np.array([X0_normal, c0_normal, n0_normal, co20_normal, o20_normal, q0_normal, z0_normal])

# Initial conditions simulation VAL (PARAMETER VALIDATION
X0_normal = 0.223
c0_normal = 5.992
n0_normal = 1.027
co20_normal = 5.319168e-02
# o20 = parameters['O2_sat']
o20_normal = 0.0001
q0_normal = 0.096
z0_normal = 7.2
x0_sim_v_normal = np.array([X0_normal, c0_normal, n0_normal, co20_normal, o20_normal, q0_normal, z0_normal])


# Initial conditions simulation PE (PARAMETER ESTIMATION)
X0_N_def = 0.224115
c0_N_def = 6.05676666666667
n0_N_def = 0.0362565333333333
co20_N_def = 0.7004758
# o20 = parameters['O2_sat']
o20_N_def = 0.0001
q0_N_def = 0.096
z0_N_def = 6.9
x0_sim_N_def = np.array([X0_N_def, c0_N_def, n0_N_def, co20_N_def, o20_N_def, q0_N_def, z0_N_def])

# Initial conditions simulation VAL (PARAMETER VALIDATION
X0_N_def = 0.221913
c0_N_def = 5.51136666666667
n0_N_def = 0.0316922666666667
co20_N_def = 0.1860496
# o20 = parameters['O2_sat']
o20_N_def = 0.0001
q0_N_def = 0.096
z0_N_def = 7.09666666666667
x0_sim_v_N_def = np.array([X0_N_def, c0_N_def, n0_N_def, co20_N_def, o20_N_def, q0_N_def, z0_N_def])




# Simulation time parameters
tf = 100
n_steps = 500
time_stamps_sim = np.linspace(0, tf, n_steps + 1)

# Experimental data import
exp_data_path = os.path.join(os.getcwd(), 'Experimental_data.xlsx')

df_exp_normal = pd.read_excel(exp_data_path, sheet_name='PE_Normal') # PE = PARAMETER ESTIMATION
df_val_normal = pd.read_excel(exp_data_path, sheet_name='V_Normal') # V or VAL = PARAMETER VALIDATION


# Initial points experimental PE
X0_normal = df_exp_normal['X'][0]  
c0_normal = df_exp_normal['C'][0] 
n0_normal = df_exp_normal['N'][0] 
co20_normal = 1.377279e-01
o20_normal = constants['O2_sat']  # O2 saturation in g/L
q0_normal = 0.096
z0_normal = df_exp_normal['pH'][0]  
x0_exp_normal = np.array([X0_normal, c0_normal, n0_normal, co20_normal, o20_normal, q0_normal, z0_normal]) 

# Experimental time parameters PE
t_exp_normal = df_exp_normal['Time (Hours)']

# Initial points experimental VAL
X0_v_normal = df_val_normal['X'][0]  
c0_v_normal = df_val_normal['C'][0] 
n0_v_normal = df_val_normal['N'][0] 
co20_v_normal = 9.112441e-02
# o20_v = parameters['O2_sat']  # O2 saturation in g/L
o20_v_normal = 0.0001
q0_v_normal = 0.096
z0_v_normal = df_val_normal['pH'][0]  
x0_exp_v_normal = np.array([X0_v_normal, c0_v_normal, n0_v_normal, co20_v_normal, o20_v_normal, q0_v_normal, z0_v_normal]) 

# Weights for experimental data
weights_exp_stack_normal = np.vstack([
                    df_exp_normal.iloc[:, 2],
                    df_exp_normal.iloc[:, 5],
                    df_exp_normal.iloc[:, 8],
                    df_exp_normal.iloc[:, 11]
                ]).T

# Experimental time parameters VAL
t_exp_v_normal = df_val_normal['Time (Hours)']




df_exp_N_def = pd.read_excel(exp_data_path, sheet_name='PE_Amm_def') # PE = PARAMETER ESTIMATION
df_val_N_def = pd.read_excel(exp_data_path, sheet_name='V_Amm_def_rep') # V or VAL = PARAMETER VALIDATION

# Initial points experimental PE
X0_N_def = df_exp_N_def['X'][0]  
c0_N_def = df_exp_N_def['C'][0] 
n0_N_def = df_exp_N_def['N'][0] 
co20_N_def = 0.7004758
o20_N_def = constants['O2_sat']  # O2 saturation in g/L
q0_N_def = 0.096
z0_N_def = df_exp_N_def['pH'][0]  
x0_exp_N_def = np.array([X0_N_def, c0_N_def, n0_N_def, co20_N_def, o20_N_def, q0_N_def, z0_N_def]) 

# Experimental time parameters PE
t_exp_N_def = df_exp_N_def['Time (Hours)']

# Initial points experimental VAL
X0_v_N_def = df_val_N_def['X'][0]  
c0_v_N_def = df_val_N_def['C'][0] 
n0_v_N_def = df_val_N_def['N'][0] 
co20_v_N_def = 0.1860496
# o20_v = parameters['O2_sat']  # O2 saturation in g/L
o20_v_N_def = 0.0001
q0_v_N_def = 0.096
z0_v_N_def = df_val_N_def['pH'][0]  
x0_exp_v_N_def = np.array([X0_v_N_def, c0_v_N_def, n0_v_N_def, co20_v_N_def, o20_v_N_def, q0_v_N_def, z0_v_N_def]) 

# Weights for experimental data
weights_exp_stack_N_def = np.vstack([
                    df_exp_N_def.iloc[:, 2],
                    df_exp_N_def.iloc[:, 5],
                    df_exp_N_def.iloc[:, 8],
                    df_exp_N_def.iloc[:, 11]
                ]).T

# Experimental time parameters VAL
t_exp_v_N_def = df_val_N_def['Time (Hours)']










# Calibration settings
delta = 1e-4                  # absolute FIM step (only used if delta_rel is not defined)
delta_rel = 1e-3              # [CHANGED] relative FIM step: h = delta_rel*|p| (scale-free; used by Param_analysis_functions if present)
correlation_threshold = 0.95  # [CHANGED] was 0.80. Advisor slides: highly correlated if |kappa| > 0.95
perturbation = 0.05
fim_mode = 'residual'         # [NEW] 'residual': Q from calibration residuals (slides); 'replicate': 1/(3SE^2) weights applied once (cross-check)
sens_window = 'exp'           # [NEW] sensitivities at experimental times ('exp') or on the 0-100 h grid ('sim')

# Columns for data export
column_names = []
columns_values = []
columns_t_values = []
for key, value in parameters.items():
    columns_values.append(key)
    columns_t_values.append('t_value_' + key)

columns_var_validation = []
for var in var_exp_names:
    columns_var_validation.append('RMSE_' + var)
    columns_var_validation.append('NRMSE_' + var)
    columns_var_validation.append('MAPE_' + var)

column_names.extend(['Model', 'Param Combo', 'lb', 'ub'])
column_names.extend(columns_values)
column_names.extend(columns_t_values)
column_names.extend(columns_var_validation)
column_names.extend(['RMSE', 'NRMSE', 'MAPE', 'AIC', 'BIC'])

# System information dictionary
system_data = {
    'var_names': var_names,
    'var_exp_names': var_exp_names,
    'parameters': parameters,
    'parameters_og_list': parameters_og_list,
    'constants': constants,
    'x0_sim_normal': x0_sim_normal,
    'x0_exp_normal': x0_exp_normal,
    'x0_sim_v_normal': x0_sim_v_normal,
    'x0_exp_v_normal': x0_exp_v_normal,
    'x0_sim_N_def': x0_sim_N_def,
    'x0_exp_N_def': x0_exp_N_def,
    'x0_sim_v_N_def': x0_sim_v_N_def,
    'x0_exp_v_N_def': x0_exp_v_N_def,
    'time_stamps_sim': time_stamps_sim,
    'df_exp_normal': df_exp_normal,
    'df_val_normal': df_val_normal,
    't_exp_normal': t_exp_normal,
    't_exp_v_normal': t_exp_v_normal,
    'weights_exp_stack_normal': weights_exp_stack_normal,
    'df_exp_N_def': df_exp_N_def,
    'df_val_N_def': df_val_N_def,
    't_exp_N_def': t_exp_N_def,
    't_exp_v_N_def': t_exp_v_N_def,
    'weights_exp_stack_N_def': weights_exp_stack_N_def,
    'delta': delta,
    'delta_rel': delta_rel,
    'fim_mode': fim_mode,
    'sens_window': sens_window,
    'correlation_threshold': correlation_threshold,
    'perturbation': perturbation,
    'column_names': column_names
}



# display names only (figure titles); the keys must stay 'normal' / 'N_def'
system_data['condition_labels'] = {
    'normal': 'Nutrient-replete',
    'N_def':  'Ammonium-deficient',      # or 'Ammonium_deficient', whatever you prefer to see
}

# optional: names of the four experiments, used in the column titles of the single-solution plots
system_data['dataset_names'] = {
    'normal': {'cal': 'PE_Normal',  'val': 'V_Normal'},
    'N_def':  {'cal': 'PE_Amm_def', 'val': 'V_Amm_def_rep'},
}