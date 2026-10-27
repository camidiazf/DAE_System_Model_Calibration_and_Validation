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
#   [9] NON-NEGATIVITY GUARDS: all biological state variables (X, C, N, O, q)
#       are clamped to zero inside rate expressions via ca.fmax so that:
#         (a) rates shut off naturally when a substrate is depleted, and
#         (b) numerical overshoot cannot propagate into negative concentrations.
#       Additionally, rho now includes the missing N Monod term
#       N_g/(N_g+k_Nup), which is the primary reason N was going negative
#       (uptake had no dependence on external N, so it continued draining N
#       past zero regardless of how low N fell).
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
    """Shared kinetics -- used identically by the DAE and by mu post-processing.

    [9] All biological state variables are guarded with ca.fmax before use in
    rate expressions.  This ensures:
      - Monod terms (f_C, f_O) → 0 when substrate is exhausted.
      - rho → 0 when external N is exhausted (primary fix for N going negative).
      - mu → 0 when X, C, or O hit zero.
      - No negative concentration can feed a positive production rate.
    """
    # [9] Guard all biological states at their physical lower bound (zero).
    #     These guarded aliases are used EVERYWHERE below; the raw symbols are
    #     not used in any arithmetic inside this function.
    X_g = ca.fmax(X, 0)
    C_g = ca.fmax(C, 0)
    N_g = ca.fmax(N, 0)
    O_g = ca.fmax(O, 0)
    q_g = ca.fmax(q, 0)

    Iph = ca.exp(-((pH - p['pH_opt']) / p['pH_w']) ** 2)      # [3] bell, <=1

    f_lag = 1 - ca.exp(-t / p['t_lag'])
    f_C   = C_g / (C_g + p['k_C'])                            # [9] uses C_g
    f_O   = O_g / (O_g + p['k_O'])                            # [9] uses O_g
    f_cap = ca.fmax(1 - X_g / p['Xmax'], 0)**p['n_cap']                     # Han-Levenspiel-type cell self-inhibition; =1-X/Xmax at n_cap=1

    # [7] Droop quota term REPLACES the external-N Monod term in growth.
    # [9] Guard against division by zero when q_g → 0.
    q_safe = ca.fmax(q_g, 1e-12)
    f_Nq = ca.fmax(1 - p['q_min'] / q_safe, 0)

    mu = p['mu_max'] * f_lag * f_C * f_O * f_cap * Iph * f_Nq

    # [7] Nitrogen UPTAKE -- separate from growth, fills the quota.
    # [9] Restored the missing N Monod term N_g/(N_g+k_Nup).
    #     Without it, rho was independent of external N, so uptake continued
    #     even when N < 0, which was the primary cause of N going negative.
    f_fill = ca.fmax((p['q_max'] - q_g) / (p['q_max'] - p['q_min']), 0)
    rho = p['rho_N_max'] * (N_g / (N_g + p['k_Nup'])) * f_fill  # [9] N_g added

    # biomass-dependent maintenance (Pirt-type, decays with X; no substrate gate); =m_C*(1-X/Xmax) at m_cap=1
    maint = p['m_C'] * ca.fmax(1 - X_g / p['Xmax'], 0)**p['m_cap']
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

    # [9] Use ca.fmax(X, 0) and ca.fmax(q, 0) in mass balances so that a
    #     transient numerical undershoot below zero does not flip the sign of
    #     the production/consumption terms and destabilise the integration.
    X_g = ca.fmax(X, 0)
    q_g = ca.fmax(q, 0)
    O_g = ca.fmax(O, 0)

    dXdt   = (mu - params['k_d']) * X_g
    dCdt   = -q_C * X_g
    # [7][8] N driven by UPTAKE, not growth yield; ammonification releases
    #        the dying cells' own quota content (q * k_d * X).
    # [9]    rho already → 0 when N ≤ 0 (via N_g in _kinetics).
    dNdt   = -rho * X_g + params['f_Nrec'] * params['k_d'] * X_g * q_g
    dqdt   = rho - mu * q_g                                   # [7] NEW ODE

    dCO2dt = ((mu / params['YX_CO2']) * X_g + maint * params['Y_CO2_m'] * X_g)
    dCO2dt = dCO2dt - ka7 * (CO2 / ((H / ka7) + 1))
    dOdt   = (constants['k_La'] * (constants['O2_sat'] - O_g)
              - (mu / params['YX_O2']) * X_g
              - maint * params['Y_O2_m'] * X_g)

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

    # [9] Same guards as DAE_system.
    X_g = ca.fmax(X, 0)
    q_g = ca.fmax(q, 0)
    O_g = ca.fmax(O, 0)

    dXdt   = (mu - pars['k_d']) * X_g
    dCdt   = -q_C * X_g
    dNdt   = -rho * X_g + pars['f_Nrec'] * pars['k_d'] * X_g * q_g
    dqdt   = rho - mu * q_g

    dCO2dt = ((mu / pars['YX_CO2']) * X_g + maint * pars['Y_CO2_m'] * X_g)
    dCO2dt = dCO2dt - ka7 * (CO2 / ((H / ka7) + 1))
    dOdt   = (constants['k_La'] * (constants['O2_sat'] - O_g)
              - (mu / pars['YX_O2']) * X_g
              - maint * pars['Y_O2_m'] * X_g)

    return ca.vertcat(dXdt, dCdt, dNdt, dCO2dt, dOdt, dqdt)


def simulate_model(system_data: Dict[str, any], simulation_type: str,
                   x0: np.ndarray, parameters: Dict[str, float],
                   time: np.ndarray, p_vars: Optional[np.ndarray] = None, ext: str = '_normal'
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

    # [7] mu recomputed with the quota factor, matching the DAE exactly.
    # [9] Guard states before recomputing mu so the post-processing value is
    #     consistent with what the solver actually used inside _kinetics.
    X_g = np.maximum(X, 0)
    C_g = np.maximum(C, 0)
    O_g = np.maximum(O, 0)
    q_g = np.maximum(q, 0)
    q_safe = np.maximum(q_g, 1e-12)

    Iph  = np.exp(-((pH - parameters['pH_opt']) / parameters['pH_w']) ** 2)
    f_Nq = np.maximum(1 - parameters['q_min'] / q_safe, 0)
    mu_values = (parameters['mu_max']
                 * (1 - np.exp(-t / parameters['t_lag']))
                 * (C_g / (C_g + parameters['k_C']))
                 * (O_g / (O_g + parameters['k_O']))
                 * np.maximum(1 - X_g / parameters['Xmax'], 0) ** parameters['n_cap']
                 * Iph * f_Nq)

    return pd.DataFrame({'t': t, 'X': X, 'C': C, 'N': N, 'CO2': CO2,
                         'O': O, 'q': q, 'pH': pH, 'H': H,
                         'mu_values': mu_values})