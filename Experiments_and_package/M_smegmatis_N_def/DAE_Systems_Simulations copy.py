import os
from typing import Dict, List, Optional

import casadi as ca  # type: ignore
import numpy as np
import pandas as pd

import sys
import contextlib

# =============================================================================
#  REVISED DAE MODEL v2 — adds a Droop nitrogen-quota state
#
#  Carries forward every change from the previous revision (maintenance [1],
#  corrected pH inhibition [3], NH4+ in the charge balance [5], fixed mu
#  recomputation [6]) and adds:
#
#   [7] Internal nitrogen quota q [g N / g DCW] as a NEW STATE (6th state).
#       External N no longer appears directly in the growth-rate expression;
#       growth depends on q via the Droop equation. External N still drives
#       UPTAKE (rho), which fills the quota. This decouples "how fast N
#       disappears from the broth" from "how long growth continues" —
#       resolving the X/N whack-a-mole seen in the fitting plots.
#   [8] Decay-driven ammonification now releases the ACTUAL quota content of
#       the dying cells (q * k_d * X), not a flat fraction of biomass —
#       more mechanistically grounded than the previous version.
#
#  STATE VECTOR CHANGE: x = [X, C, N, CO2, O, q]   (was 5 states, now 6)
#  x0 arrays in System_data.py must gain ONE more entry (q0) inserted
#  right before the trailing pH guess (z0). See bottom of this file.
# =============================================================================


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


def _kinetics(constants, t, X, C, N, O, q, pH, p):
    """Shared kinetics -- used identically by the DAE and by mu post-processing."""
    Iph = ca.exp(-((pH - p['pH_opt']) / p['pH_w']) ** 2)      # [3] bell, <=1

    f_lag = 1 - ca.exp(-t / p['t_lag'])
    f_C   = C / (C + p['k_C'])
    f_O   = O / (O + p['k_O'])
    f_cap = 1 - X / p['Xmax']                                  # provisional cap

    # [7] Droop quota term REPLACES the external-N Monod term in growth.
    f_Nq = ca.fmax(1 - p['q_min'] / q, 0)

    mu = p['mu_max'] * f_lag * f_C * f_O * f_cap * Iph * f_Nq

    # [7] Nitrogen UPTAKE -- separate from growth, fills the quota.
    f_fill = ca.fmax((p['q_max'] - q) / (p['q_max'] - p['q_min']), 0)
    rho = p['rho_N_max'] * (N / (N + p['k_Nup'])) * f_fill

    # maintenance-gated glycerol uptake (from previous revision)
    maint = p['m_C'] * f_C
    q_C = mu / p['YX_C'] + maint

    return mu, rho, q_C, maint


def DAE_system(system_data: Dict[str, any], t: float, x: np.ndarray,
               z: np.ndarray, params: Dict[str, float]) -> ca.MX:
    constants = system_data['constants']
    X, C, N, CO2, O, q = x[0], x[1], x[2], x[3], x[4], x[5]      # [7] q added
    pH = z[0]

    mu, rho, q_C, maint = _kinetics(constants, t, X, C, N, O, q, pH, params)

    ka7 = 10 ** (-constants['pka7'])
    H = 10 ** (-pH)

    dXdt = (mu - params['k_d']) * X
    dCdt = - q_C * X
    # [7][8] N driven by UPTAKE, not growth yield; ammonification releases
    #        the dying cells' own quota content (q * k_d * X).
    dNdt = - rho * X + params['f_Nrec'] * params['k_d'] * X * q
    dqdt = rho - mu * q                                          # [7] NEW ODE

    dCO2dt = ((mu / params['YX_CO2']) * X + maint * params['Y_CO2_m'] * X)
    dCO2dt = dCO2dt - ka7 * (CO2 / ((H / ka7) + 1))
    dOdt = (constants['k_La'] * (constants['O2_sat'] - O)
            - (mu / params['YX_O2']) * X
            - maint * params['Y_O2_m'] * X)

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt, dqdt)       # [7] 6 outputs


def DAE_system_calibrating(system_data: Dict[str, any], t: float, x: np.ndarray,
                           z: np.ndarray, p: np.ndarray,
                           parameters: Dict[str, float]) -> ca.MX:
    constants = system_data['constants']
    param_list = system_data['calibrating_parameters']

    pars = parameters.copy()
    for i, name in enumerate(param_list):
        pars[name] = p[i]

    X, C, N, CO2, O, q = x[0], x[1], x[2], x[3], x[4], x[5]
    pH = z[0]

    mu, rho, q_C, maint = _kinetics(constants, t, X, C, N, O, q, pH, pars)

    ka7 = 10 ** (-constants['pka7'])
    H = 10 ** (-pH)

    dXdt = (mu - pars['k_d']) * X
    dCdt = - q_C * X
    dNdt = - rho * X + pars['f_Nrec'] * pars['k_d'] * X * q
    dqdt = rho - mu * q

    dCO2dt = ((mu / pars['YX_CO2']) * X + maint * pars['Y_CO2_m'] * X)
    dCO2dt = dCO2dt - ka7 * (CO2 / ((H / ka7) + 1))
    dOdt = (constants['k_La'] * (constants['O2_sat'] - O)
            - (mu / pars['YX_O2']) * X
            - maint * pars['Y_O2_m'] * X)

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt, dqdt)


def simulate_model(system_data: Dict[str, any], simulation_type: str,
                   x0: np.ndarray, parameters: Dict[str, float],
                   time: np.ndarray, p_vars: Optional[np.ndarray] = None
                   ) -> Optional[pd.DataFrame]:
    constants = system_data['constants']

    t = ca.MX.sym('t')
    x = ca.MX.sym('x', 6)                                          # [7] was 5
    z = ca.MX.sym('z')

    if simulation_type == 'calibrating':
        param_list = system_data['calibrating_parameters']
        p = ca.MX.sym('p', len(param_list))
        dxdt = DAE_system_calibrating(system_data, t, x, z, p, parameters)
    else:
        p = ca.MX.sym('p', 0)
        dxdt = DAE_system(system_data, t, x, z, parameters)

    # ---- algebraic pH charge balance (unchanged from previous revision) ----
    ka1 = 10 ** (-constants['pka1'])
    ka2 = 10 ** (-constants['pka2'])
    ka3 = 10 ** (-constants['pka3'])
    ka4 = 10 ** (-constants['pka4'])
    ka5 = 10 ** (-constants['pka5'])
    ka7 = 10 ** (-constants['pka7'])
    ka9 = 10 ** (-constants['pka9'])

    H = 10 ** (-z)

    KHPO4 = constants['KH2PO4'] / ((H / ka1) + 1)
    C6H5O7 = constants['C6H8O7'] / ((H ** 3 / (ka2 * ka3 * ka4))
                                    + (H ** 2 / (ka3 * ka4)) + (H / ka4) + 1)
    C6H6O7 = (H / ka4) * C6H5O7
    C6H7O7 = (H / ka3) * C6H6O7
    HCO3 = x[3] / ((H / ka7) + 1)
    OH = ka9 / H
    NH4 = (x[2] / constants['M_N']) * (H / (H + ka5))               # external N only

    f_z = (OH + HCO3 + KHPO4 + (3 * C6H5O7) + (2 * C6H6O7) + C6H7O7
           - constants['pH_alk'] - H - NH4)

    dae = {'t': t, 'x': x, 'z': z, 'p': p, 'ode': dxdt, 'alg': f_z}
    integrator = ca.integrator('F', 'idas', dae, {'grid': time, 'output_t0': True})

    try:
        if simulation_type == 'calibrating':
            sol = integrator(x0=x0[:-1], z0=x0[-1], p=p_vars)
        else:
            sol = integrator(x0=x0[:-1], z0=x0[-1])
    except RuntimeError as e:
        print("Integration failed:", e)
        return None

    t = time
    xf = sol['xf'].full().T
    zf = sol['zf'].full().T
    X, C, N, CO2, O, q = xf[:, 0], xf[:, 1], xf[:, 2], xf[:, 3], xf[:, 4], xf[:, 5]
    pH = zf[:, 0]
    H = 10 ** (-pH)

    # [7] mu recomputed with the quota factor, matching the DAE exactly
    Iph = np.exp(-((pH - parameters['pH_opt']) / parameters['pH_w']) ** 2)
    f_Nq = np.maximum(1 - parameters['q_min'] / q, 0)
    mu_values = (parameters['mu_max'] * (1 - np.exp(-t / parameters['t_lag']))
                 * (C / (C + parameters['k_C']))
                 * (O / (O + parameters['k_O']))
                 * (1 - X / parameters['Xmax'])
                 * Iph * f_Nq)

    return pd.DataFrame({'t': t, 'X': X, 'C': C, 'N': N, 'CO2': CO2,
                         'O': O, 'q': q, 'pH': pH, 'H': H,
                         'mu_values': mu_values})
