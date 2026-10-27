"""
End-to-end plumbing test of the new results store with a STUB simulator (a small logistic/Monod model, NOT the DAE system).
mealpy / statsmodels are replaced by tiny stand-ins only if they cannot be imported, so the test also runs where they are not
installed. It checks storage, resume, figures and comparison - not the PSO or the DAE model.

Run:  MPLBACKEND=Agg python3 test_campaign.py
"""
import os
import sys
import shutil
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- optional stand-ins
try:
    import mealpy  # noqa: F401
    from mealpy import FloatVar, PSO  # noqa: F401
    REAL_MEALPY = True
except Exception:
    REAL_MEALPY = False
    n_calls = {'solve': 0}

    class _FloatVar:
        def __init__(self, lb, ub):
            self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)

    class _Target:
        def __init__(self, f):
            self.fitness = f

    class _Agent:
        def __init__(self, x, f):
            self.solution, self.target = x, _Target(f)

    class _History:
        list_global_best_fit: list = []

    class _OriginalPSO:
        """Random search with the interface used by RUN_PSO_CALIBRATION."""
        def __init__(self, epoch, pop_size, c1=1.5, c2=1.5, w=0.5):
            self.epoch, self.pop_size = epoch, pop_size
            self.history = _History()

        def solve(self, problem, seed=None):
            n_calls['solve'] += 1
            rng = np.random.default_rng(seed)
            lb, ub = problem['bounds'].lb, problem['bounds'].ub
            best_x, best_f, hist = None, np.inf, []
            for _ in range(self.epoch):
                for _ in range(self.pop_size):
                    x = lb + rng.random(len(lb)) * (ub - lb)
                    f = problem['obj_func'](x)
                    if f < best_f:
                        best_x, best_f = x, f
                hist.append(best_f)
            self.history = _History()
            self.history.list_global_best_fit = hist
            return _Agent(best_x, best_f)

    mealpy = types.ModuleType('mealpy')
    mealpy.FloatVar = _FloatVar
    mealpy.PSO = types.SimpleNamespace(OriginalPSO=_OriginalPSO)
    mealpy_utils = types.ModuleType('mealpy.utils'); mealpy_problem = types.ModuleType('mealpy.utils.problem')
    mealpy_problem.FloatVar = _FloatVar
    mealpy_swarm = types.ModuleType('mealpy.swarm_based'); mealpy_swarm.PSO = mealpy.PSO
    sys.modules.update({'mealpy': mealpy, 'mealpy.utils': mealpy_utils, 'mealpy.utils.problem': mealpy_problem,
                        'mealpy.swarm_based': mealpy_swarm})

try:
    import statsmodels.stats.stattools  # noqa: F401
except Exception:
    sm = types.ModuleType('statsmodels'); sms = types.ModuleType('statsmodels.stats')
    smt = types.ModuleType('statsmodels.stats.stattools')
    smt.durbin_watson = lambda r: float(np.sum(np.diff(r) ** 2) / np.sum(np.asarray(r) ** 2))
    sys.modules.update({'statsmodels': sm, 'statsmodels.stats': sms, 'statsmodels.stats.stattools': smt})

# package is imported as DAE_analysis_package.* ; this script sits next to that folder
from scipy.integrate import solve_ivp  # noqa: E402

from DAE_analysis_package.RUN_functions import RUN_COMBOS  # noqa: E402
from DAE_analysis_package.Results_store import ResultsStore  # noqa: E402
from DAE_analysis_package.Comparison_functions import RUN_MODEL_COMPARISON  # noqa: E402

# ---------------------------------------------------------------- stub model + synthetic data
VAR_NAMES = ['X', 'C', 'N', 'pH']
BASE = {'mu_max': 0.25, 'k_C': 0.5, 'Xmax': 6.0, 'YX_C': 0.5, 'YX_N': 8.0, 't_lag': 3.0, 'k_d': 0.002}
TRUE = dict(BASE, mu_max=0.31, YX_C=0.43, t_lag=5.0)


def simulate_model(system_data, simulation_type, x0, parameters, time, p_vars=None):
    pars = dict(parameters)
    if simulation_type == 'calibrating':
        for name, v in zip(system_data['calibrating_parameters'], p_vars):
            pars[name] = float(v)

    def rhs(t, y):
        X, C, N, pH = y
        C, N = max(C, 0.0), max(N, 0.0)
        mu = pars['mu_max'] * (1 - np.exp(-t / pars['t_lag'])) * C / (C + pars['k_C']) * (1 - X / pars['Xmax'])
        mu *= N / (N + 0.01)
        dX = (mu - pars['k_d']) * X
        return [dX, -mu * X / pars['YX_C'], -mu * X / pars['YX_N'], -0.05 * mu * X]

    t = np.asarray(time, dtype=float)
    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(x0, float), t_eval=t, method='LSODA', rtol=1e-7, atol=1e-9)
    if not sol.success:
        return None
    return pd.DataFrame(sol.y.T, columns=VAR_NAMES)


def build_system_data():
    rng = np.random.default_rng(0)
    sd = {'parameters': dict(BASE), 'parameters_og_list': list(BASE), 'var_names': VAR_NAMES, 'var_exp_names': VAR_NAMES,
          'time_stamps_sim': np.arange(0, 100.0001, 0.5), 'perturbation': 0.01, 'delta': 1e-4, 'delta_rel': 1e-3,
          'correlation_threshold': 0.95, 'fim_mode': 'residual', 'cost_scaling': 'range', 'pso_seed_base': 1234,
          'pso_epoch': 6, 'pso_pop_size': 8,
          'dataset_names': {'normal': {'cal': 'PE_Normal', 'val': 'V_Normal'},
                            'N_def': {'cal': 'PE_Amm_def', 'val': 'V_Amm_def_rep'}}}
    x0 = {'normal': [0.1, 10.0, 1.0, 7.0], 'N_def': [0.1, 10.0, 0.2, 7.0]}
    x0v = {'normal': [0.15, 8.0, 1.0, 7.0], 'N_def': [0.15, 8.0, 0.25, 7.0]}
    for cond in ('normal', 'N_def'):
        ext = '_' + cond
        for kind, tt, x in (('cal', np.arange(0, 48.1, 4.0), x0[cond]), ('val', np.arange(0, 96.1, 8.0), x0v[cond])):
            df = simulate_model(sd, 'normal', x, TRUE, tt)
            df = df * (1 + 0.03 * rng.standard_normal(df.shape))
            if kind == 'cal':
                sd['t_exp' + ext], sd['df_exp' + ext], sd['x0_exp' + ext] = tt, df, np.array(x)
                sd['weights_exp_stack' + ext] = np.ones(len(tt) * len(VAR_NAMES))
                sd['x0_sim' + ext] = np.array(x)
            else:
                sd['t_exp_v' + ext], sd['df_val' + ext], sd['x0_exp_v' + ext] = tt, df, np.array(x)
    return sd


# ---------------------------------------------------------------- tests
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    work = tempfile.mkdtemp(prefix='campaign_test_')
    os.chdir(work)
    print("working dir:", work, "| real mealpy:", REAL_MEALPY)
    sd = build_system_data()
    pb = {'mu_max': [0.05, 0.6], 'YX_C': [0.2, 0.9], 't_lag': [0.5, 10.0]}

    print("\n[1] campaign: 3 params, k=2..3, 2 iterations -> 4 combos x 2 runs + original")
    store = RUN_COMBOS(sd, simulate_model, 'Stub', 'camp1', 'calibrate', iterations=2, parameters_and_bounds=pb, min_k=2, max_k=3)
    runs = store.read('runs')
    check(len(runs) == 9 and (runs['status'] == 'done').all(), f"runs.csv: 9 rows all done (got {len(runs)}, {runs['status'].unique().tolist()})")
    check(runs['run_id'].is_unique, "run ids unique")
    check(set(store.read('parameters')['run_id']) == set(runs['run_id']), "parameters table covers every run")
    met = store.read('metrics')
    check(set(met['dataset']) == {'cal', 'val'} and set(met['condition']) == {'normal', 'N_def'}, "metrics: both datasets, both conditions")
    check(met[(met['variable'] == 'global')].groupby('run_id').size().eq(4).all(), "metrics: 4 global rows per run (2 cond x 2 ds)")
    fitted = met[met['fitted'].astype(str).str.lower() == 'true']
    check(set(fitted['condition']) == {'N_def'} and set(fitted['dataset']) == {'cal'}, "fitted flag only on N_def calibration data")
    ide = store.read('identifiability')
    check(len(ide) > 0 and set(ide[ide['fitted_on'].astype(str).str.lower() == 'false']['condition']) == {'normal'},
          "identifiability: normal condition flagged fitted_on=False")
    first = runs[runs['combo_id'] != 'original']['run_id'].iloc[0]
    for cond in ('normal', 'N_def'):
        d = os.path.join(store.run_dir(first), 'figures', cond)
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        check({'solution.png', 'residuals_cal.png', 'residuals_val.png', 'sensitivity.png'} <= set(files),
              f"figures/{cond}: {files}")
    sol_files = sorted(os.listdir(os.path.join(store.run_dir(first), 'solutions')))
    check(sol_files == ['N_def_cal.csv', 'N_def_val.csv', 'normal_cal.csv', 'normal_val.csv'], f"solutions: {sol_files}")
    cdirs = os.listdir(store.combos_dir)
    check(any(len(os.listdir(os.path.join(store.combos_dir, c))) > 0 for c in cdirs), f"combo summary plots written ({len(cdirs)} combo folders)")
    comp = sorted(os.listdir(store.comparison_dir))
    for need in ('All_Calibration_Solutions.png', 'All_Validation_Solutions.png', 'All_Calibration_Metrics.png',
                 'All_Validation_Metrics.png', 'All_Calibration_Solutions_Band.png', 'All_models_ranking.csv', 'excluded_runs.csv'):
        check(need in comp, f"comparison/{need}")
    rank = pd.read_csv(os.path.join(store.comparison_dir, 'All_models_ranking.csv'))
    check(len(rank) == 8, f"ranking has ALL 8 runs (got {len(rank)})")
    check(rank['rank_score_NRMSE'].is_monotonic_increasing, "ranking sorted by calibration NRMSE")
    check(os.path.exists(os.path.join(store.tables_dir, 'summary.xlsx')), "summary.xlsx written")
    wide = pd.read_csv(os.path.join(store.tables_dir, 'summary_wide.csv'))
    check(len(wide) == 9, f"summary_wide: 1 row per run ({len(wide)})")

    print("\n[2] guards")
    try:
        RUN_COMBOS(sd, simulate_model, 'Stub', 'camp1', 'calibrate', iterations=2, parameters_and_bounds=pb, min_k=2, max_k=3)
        check(False, "existing campaign without resume must raise")
    except FileExistsError:
        check(True, "existing campaign without resume=True raises FileExistsError")
    pb2 = dict(pb, mu_max=[0.05, 0.7])
    try:
        RUN_COMBOS(sd, simulate_model, 'Stub', 'camp1', 'calibrate', iterations=2, parameters_and_bounds=pb2, min_k=2, max_k=3, resume=True)
        check(False, "changed bounds on resume must raise")
    except ValueError:
        check(True, "changed bounds on resume raises ValueError")

    print("\n[3] resume skips finished runs; more iterations only run the new ones")
    n_before = None if REAL_MEALPY else n_calls['solve']
    store2 = RUN_COMBOS(sd, simulate_model, 'Stub', 'camp1', 'calibrate', iterations=2, parameters_and_bounds=pb, min_k=2, max_k=3, resume=True)
    if n_before is not None:
        check(n_calls['solve'] == n_before, "resume with same settings: 0 new PSO runs")
    store3 = RUN_COMBOS(sd, simulate_model, 'Stub', 'camp1', 'calibrate', iterations=3, parameters_and_bounds=pb, min_k=2, max_k=3, resume=True)
    runs3 = store3.read('runs')
    check(len(runs3) == 1 + 4 * 3 and (runs3['status'] == 'done').all(), f"iterations 2 -> 3: {len(runs3)} runs (expected 13)")
    check(runs3['run_id'].is_unique, "no duplicated rows after resume")
    seeds_ok = runs3.groupby('combo_id')['seed'].nunique().drop('original', errors='ignore').eq(3).all()
    check(bool(seeds_ok), "seeds differ between iterations of a combo")

    print("\n[4] top-fraction comparison on its own + failed-run path")
    out = RUN_MODEL_COMPARISON(sd, ResultsStore.open(store.dir), top_frac=0.25)
    check(len(out['ranking']) == 3 and 'Top25pct_Validation_Solutions.png' in os.listdir(store.comparison_dir),
          f"top 25 % of 12 runs = 3 ({len(out['ranking'])}), prefix Top25pct")

    def failing(system_data, simulation_type, x0, parameters, time, p_vars=None):
        raise ValueError("boom")
    try:
        RUN_COMBOS(sd, failing, 'Stub', 'camp_fail', 'calibrate', iterations=1, parameters_and_bounds={'mu_max': [0.1, 0.5]}, min_k=1, max_k=1)
        check(False, "campaign where every run fails must raise")
    except RuntimeError as e:
        check('failed' in str(e), "campaign where every run fails raises RuntimeError")

    print(f"\n{'ALL OK' if check.failed == 0 else str(check.failed) + ' FAILED'}  | results kept in {work}")
    return work


if __name__ == '__main__':
    main()
