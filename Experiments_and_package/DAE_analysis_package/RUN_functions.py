import os
import sys
import time
import traceback
from contextlib import contextmanager
from itertools import combinations
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from DAE_analysis_package.Param_analysis_functions import PARAMETER_ANALYSIS
from DAE_analysis_package.Validation_functions import RUN_VALIDATION
from DAE_analysis_package.Calibration_functions import RUN_PSO_CALIBRATION, get_pso_settings
from DAE_analysis_package.Plotting_functions import plotting_single_solution, plot_combo_summary
from DAE_analysis_package.Results_store import ResultsStore, CONDITIONS, DATASETS, now_iso
from DAE_analysis_package.Comparison_functions import RUN_MODEL_COMPARISON

"""
[STORE] RUN_COMBOS / RUN_COMBINATION now write to a ResultsStore (see Results_store.py for the folder layout and tables).

  * one campaign folder per RUN_COMBOS call:  <cwd>/<folder_model>/<folder_model>_Results/<folder_results>/
  * run ids are stable (<combo_id>__rNN, combo_id = hash of parameters + bounds), not 'Model_<order>_<iteration>'
  * both conditions are handled by ONE loop (before: two copy-pasted blocks), and the calibration experiment is plotted
    next to the validation experiment in every single-solution figure
  * resume=True continues a campaign: finished runs (runs/<run_id>/run.json) are skipped; resume=False refuses to touch an
    existing campaign folder (before, a re-run appended to the old CSV and mixed old and new rows)
  * a failed run is recorded (status 'failed' + error) and the campaign goes on; if EVERY run fails an error is raised
  * RUN_COMBOS returns the ResultsStore (before: the two wide DataFrames). store.read('metrics'), store.read('runs'), ...
    give DataFrames; tables/summary_wide.csv / summary.xlsx is the one-row-per-run view.
"""


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


def _fmt_time(seconds: float) -> str:
    s = int(round(seconds)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec:02d}s"


def _progress_text(progress) -> str:
    if progress is None or progress['done'] == 0:
        return " | ETA: calculating..."
    el = time.time() - progress['t0']
    eta = el / progress['done'] * max(progress['total'] - progress['done'], 0)
    return f" | elapsed {_fmt_time(el)} | ETA {_fmt_time(eta)}"


def _to_float(x) -> float:
    """float() that also accepts 1-element lists / arrays and returns NaN for anything else."""
    try:
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1:
            x = x[0]
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


# ----------------------------------------------------------------------------------------------
# One run = (optional) calibration + evaluation of both conditions on both datasets + parameter analysis + figures
# ----------------------------------------------------------------------------------------------
def _run_single(system_data: Dict[str, Any], simulate_model: callable, store: ResultsStore, run_id: str, combo_id: str,
                iteration: int, params_list: List[str], lb: List[float], ub: List[float],
                calibration_condition: str, seed: Optional[int], original: bool) -> bool:
    """
    Calibrates (unless `original`), evaluates, stores. Returns True if the run finished ('done'); a failure is recorded
    in the tables (status 'failed', error text, calibrated parameters if the PSO had finished) and returns False.
    """
    system_data['calibrating_parameters'] = list(params_list)
    label = 'original' if original else run_id
    info = {'run_id': run_id, 'combo_id': combo_id, 'iteration': iteration + 1, 'n_params': len(params_list),
            'calibrated_parameters': ';'.join(params_list), 'seed': None, 'cost': None, 'bound_hits': '',
            'n_failed_evals': None, 'n_evals': None, 'time_s': None, 'status': 'failed', 'error': '', 'timestamp': now_iso()}
    tables: Dict[str, List[Dict[str, Any]]] = {k: [] for k in
                                               ('parameters', 'metrics', 'identifiability', 'correlations', 'sensitivity',
                                                'residuals', 'pso_history')}
    try:
        # ---------------- calibration ----------------
        if original:
            refined = dict(system_data['parameters'])
        else:
            cal = RUN_PSO_CALIBRATION(system_data=system_data, simulate_model=simulate_model, iteration=iteration,
                                      lb=lb, ub=ub, ext='_' + calibration_condition, seed=seed)
            refined = cal['RESULTS']
            info.update({'seed': cal['SEED'], 'cost': cal['COST'], 'bound_hits': ';'.join(cal['BOUND_HITS']),
                         'n_failed_evals': cal['N_FAILED_EVALS'], 'n_evals': cal['N_EVALS'], 'time_s': cal['TIME_S']})
            tables['pso_history'] = [{'run_id': run_id, 'epoch': k + 1, 'best_cost': v} for k, v in enumerate(cal['HISTORY'])]
        for p in refined:
            is_cal = p in params_list
            k = params_list.index(p) if is_cal else None
            tables['parameters'].append({'run_id': run_id, 'parameter': p, 'value': _to_float(refined[p]),
                                         'lb': float(lb[k]) if is_cal else np.nan, 'ub': float(ub[k]) if is_cal else np.nan,
                                         'calibrated': is_cal})

        # ---------------- evaluation, per condition ----------------
        n_og = len(system_data['parameters_og_list'])
        for cond in CONDITIONS:
            ext = '_' + cond
            figdir = store.figures_dir(run_id, cond)
            sols: Dict[str, Any] = {}
            mets: Dict[str, Dict[str, float]] = {}
            for ds in DATASETS:
                res = RUN_VALIDATION(system_data=system_data, simulate_model=simulate_model, fig_outdir=figdir,
                                     parameters=refined, iteration=iteration, ext=ext, original=original, dataset=ds,
                                     label=label)
                if res is None:
                    raise RuntimeError(f"simulation failed ({cond}, {ds} dataset)")
                fitted = (not original) and cond == calibration_condition and ds == 'cal'
                for r in res['Metrics_rows']:
                    tables['metrics'].append(dict(run_id=run_id, condition=cond, fitted=fitted, **r))
                for r in res['Residuals_rows']:
                    tables['residuals'].append(dict(run_id=run_id, condition=cond, **r))
                sols[ds] = res['System_sol']
                mm = res['Metrics']
                mets[ds] = {'NRMSE': mm['NRMSE_global'], 'AIC': mm['AIC'], 'BIC': mm['BIC']}
                store.save_solution(run_id, cond, ds, res['System_sol'], system_data['var_names'],
                                    system_data['time_stamps_sim'])

            # parameter analysis (sensitivity, FIM, t-values, correlations) on this condition's calibration experiment
            pa = PARAMETER_ANALYSIS(system_data=system_data, simulate_model=simulate_model, fig_outdir=figdir,
                                    iteration=iteration, parameters=refined, ext=ext, original=original, label=label)
            fitted_on = (not original) and cond == calibration_condition      # t-values of the NOT fitted condition are not estimation statistics
            row = pa['Row_param_analysis']
            rank_ratio = row[n_og] if len(row) > n_og else None
            n_hc = row[n_og + 1] if len(row) > n_og + 1 else None
            diag = pa.get('FIM_diagnostics')
            if diag is not None:
                tv = dict(zip(system_data['parameters_og_list'], row[:n_og]))
                for k, p in enumerate(diag['parameters']):
                    tables['identifiability'].append({
                        'run_id': run_id, 'condition': cond, 'fitted_on': fitted_on, 'parameter': p,
                        'theta': float(diag['theta'][k]), 't_value': _to_float(tv.get(p)), 'SE': float(diag['SE'][k]),
                        'CI_low': float(diag['CI_low'][k]), 'CI_high': float(diag['CI_high'][k]),
                        'rel_error_pct': float(diag['rel_error_pct'][k]), 'cond_FIM': diag['cond_FIM'],
                        'rank_ratio': _to_float(rank_ratio), 'n_highly_corr': _to_float(n_hc)})
            cm, names = pa.get('Correlation_matrix'), pa.get('Correlation_names')
            if cm is not None and names is not None:
                thr = pa.get('Correlation_threshold')
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        kappa = float(cm[a, b])
                        tables['correlations'].append({
                            'run_id': run_id, 'condition': cond, 'param_1': names[a], 'param_2': names[b], 'kappa': kappa,
                            'high': bool(thr is not None and np.isfinite(kappa) and abs(kappa) > thr)})
            sdf = pa.get('Sensitivity_results')
            if sdf is not None:
                for p in sdf.index:
                    for v in sdf.columns:
                        tables['sensitivity'].append({'run_id': run_id, 'condition': cond, 'parameter': p, 'variable': v,
                                                      'sensitivity': float(sdf.loc[p, v])})

            # single-solution figure: calibration | validation (a figure problem must not lose a finished calibration)
            try:
                originals = sols if original else {ds: store.load_solution('original', cond, ds,
                                                                         t_grid=system_data['time_stamps_sim'])
                                                   for ds in DATASETS}
                plotting_single_solution(system_data=system_data, fig_outdir=figdir, condition=cond, label=label,
                                         parameters_updated=refined, sols=sols, originals=originals, metrics=mets,
                                         original=original, calibration_condition=calibration_condition)
            except Exception as exc:
                print(f">>> WARNING: single-solution figure failed for {run_id} ({cond}): {exc!r}", flush=True)

        info['status'] = 'done'
        store.save_run(run_id, info, {**{k: v for k, v in tables.items()}, 'runs': [info]}, done=True)
        return True

    except Exception as exc:
        info['status'] = 'failed'
        info['error'] = f"{type(exc).__name__}: {exc}"[:400]
        print(f">>> RUN FAILED: {run_id} | {info['error']}", flush=True)
        traceback.print_exc()
        # keep what exists (calibrated parameters!) but do NOT write run.json -> resume retries this run
        keep = {k: v for k, v in tables.items() if k in ('parameters', 'pso_history')}
        store.save_run(run_id, info, {**keep, 'runs': [info]}, done=False)
        return False


# ----------------------------------------------------------------------------------------------
# One parameter combination (all its iterations) / the original model
# ----------------------------------------------------------------------------------------------
def RUN_ORIGINAL(system_data: Dict[str, Any], simulate_model: callable, store: ResultsStore,
                 calibration_condition: str = 'N_def') -> bool:
    """The model with its starting parameters (no calibration): run_id 'original'. Skipped if already done."""
    if store.is_done('original'):
        print(">>> Original model: already in the campaign, skipped", flush=True)
        return True
    print(">>> Original model (no calibration)", flush=True)
    store.register_combo('original', [], [], [])
    return _run_single(system_data, simulate_model, store, 'original', 'original', 0, [], [], [],
                       calibration_condition, None, original=True)


def RUN_COMBINATION(system_data: Dict[str, Any],
                    simulate_model: callable,
                    store: ResultsStore,
                    params_list: List[str],
                    lb: List[float],
                    ub: List[float],
                    iterations: int,
                    calibration_condition: str = 'N_def',
                    model_index: Optional[int] = None,
                    n_combos: Optional[int] = None,
                    progress: Optional[dict] = None) -> Dict[str, int]:
    """
    Calibrate one combination of parameters `iterations` times (independent PSO runs) and store every run.
    Finished runs (resume) are skipped. Returns {'combo_id', 'done', 'failed', 'skipped'}.
    """
    combo_id = store.combo_id(params_list, lb, ub)
    store.register_combo(combo_id, params_list, lb, ub)
    seed_base = system_data.get('pso_seed_base', None)
    n_done = n_failed = n_skipped = 0
    tot = f"/{n_combos}" if n_combos else ""

    for iteration in range(iterations):
        run_id = store.run_id(combo_id, iteration)
        if store.is_done(run_id):
            n_skipped += 1
            if progress is not None:
                progress['total'] -= 1          # skipped runs must not distort the ETA
            continue
        print(f">>> Model {model_index}{tot} {combo_id} {params_list} | iteration {iteration + 1}/{iterations}"
              + _progress_text(progress), flush=True)
        # deterministic seeds that do NOT depend on the order of the combinations
        seed = None if seed_base is None else int(
            (int(seed_base) + (store.combo_hash_int(combo_id) % 1_000_000) * 1000 + iteration) % (2 ** 31 - 1))
        ok = _run_single(system_data, simulate_model, store, run_id, combo_id, iteration, list(params_list), list(lb),
                         list(ub), calibration_condition, seed, original=False)
        n_done += int(ok)
        n_failed += int(not ok)
        if progress is not None:
            progress['done'] += 1
        store.refresh_exports(excel=False)

    if n_done + n_skipped >= 2:
        print(f">>> Model {model_index}{tot} {combo_id} | summary plots over the iterations", flush=True)
        try:
            plot_combo_summary(system_data, store, combo_id, calibration_condition)
        except Exception as exc:
            print(f">>> WARNING: combination summary plots failed for {combo_id}: {exc!r}", flush=True)
    return {'combo_id': combo_id, 'done': n_done, 'failed': n_failed, 'skipped': n_skipped}


# ----------------------------------------------------------------------------------------------
# The campaign
# ----------------------------------------------------------------------------------------------
def RUN_COMBOS(system_data: Dict[str, Any],
               simulate_model: callable,
               folder_model: str,
               folder_results: str,
               solve_calibrate: str,
               iterations: int = None,
               parameters_and_bounds: Dict[str, List[float]] = None,
               min_k: int = 1,
               max_k: Optional[int] = None,
               baseline: Optional[Dict[str, Any]] = None,
               resume: bool = False,
               calibration_condition: str = 'N_def',
               rank_on: str = 'N_def',
               rank_dataset: str = 'cal',
               top_frac: float = 1.0,
               allow_config_change: bool = False) -> ResultsStore:
    """
    Run all combinations (of size min_k..max_k) of the parameters in `parameters_and_bounds`, `iterations` PSO runs each,
    and store everything in <cwd>/<folder_model>/<folder_model>_Results/<folder_results>/ (see Results_store.py).

        solve_calibrate       : 'solve' = only the original model, anything else = calibrate.
        resume                : False -> the campaign folder must not exist yet; True -> continue it (finished runs skipped;
                                the settings in campaign.json must be unchanged unless allow_config_change=True).
        calibration_condition : condition the PSO is fitted to ('N_def'); the other condition is only evaluated.
        rank_on, rank_dataset, top_frac : final comparison. Default: ALL runs (top_frac=1.0), ranked by the global NRMSE on
                                the calibration data of N_def. Use RUN_MODEL_COMPARISON(system_data, store, ...) to redo it.
    Returns the ResultsStore.
    """
    campaign_dir = os.path.join(os.getcwd(), folder_model, folder_model + "_Results", folder_results)
    store = ResultsStore(campaign_dir)

    calibrate = solve_calibrate != 'solve'
    if calibrate and (iterations is None or parameters_and_bounds is None):
        raise ValueError("Calibration needs `iterations` and `parameters_and_bounds` (use solve_calibrate='solve' for the original model only).")

    config = {
        'folder_model': folder_model, 'campaign': folder_results, 'solve_calibrate': solve_calibrate,
        'iterations': iterations, 'min_k': min_k, 'max_k': max_k,
        'parameters_and_bounds': parameters_and_bounds,
        'base_parameters': system_data['parameters'],
        'pso': get_pso_settings(system_data), 'pso_seed_base': system_data.get('pso_seed_base'),
        'cost_scaling': system_data.get('cost_scaling', 'native'),
        'calibration_condition': calibration_condition,
        'var_names': system_data.get('var_names'), 'var_exp_names': system_data.get('var_exp_names'),
        'parameters_og_list': system_data.get('parameters_og_list'),
        'analysis': {k: system_data.get(k) for k in ('delta', 'delta_rel', 'perturbation', 'fim_mode',
                                                     'correlation_threshold', 'sens_window')},
        'baseline_calibrated_parameters': system_data.get('baseline_calibrated_parameters'),
        'dataset_names': system_data.get('dataset_names'), 'condition_labels': system_data.get('condition_labels'),
        'xmax': system_data.get('xmax'),
        'baseline': None if baseline is None else baseline.get('label', 'Baseline'),
        'comparison': {'rank_on': rank_on, 'rank_dataset': rank_dataset, 'top_frac': top_frac},
    }
    store.init_campaign(config, resume=resume, allow_config_change=allow_config_change)
    print(f">>> Campaign folder: {store.dir}", flush=True)

    # original model first (stored like any other run; its trajectories are the reference line of every figure)
    RUN_ORIGINAL(system_data, simulate_model, store, calibration_condition)

    totals = {'done': 0, 'failed': 0, 'skipped': 0}
    if calibrate:
        all_params = list(parameters_and_bounds.keys())
        if max_k is None:
            max_k = len(all_params)
        combos = []
        for k in range(min_k, max_k + 1):
            for combo in combinations(all_params, k):
                combos.append(list(combo))
        combos.reverse()                         # larger combinations first

        progress = {'t0': time.time(), 'done': 0, 'total': len(combos) * iterations}
        for idx, combo in enumerate(combos, start=1):
            lb = [float(parameters_and_bounds[p][0]) for p in combo]
            ub = [float(parameters_and_bounds[p][1]) for p in combo]
            res = RUN_COMBINATION(system_data=system_data, simulate_model=simulate_model, store=store, params_list=combo,
                                  lb=lb, ub=ub, iterations=iterations, calibration_condition=calibration_condition,
                                  model_index=idx, n_combos=len(combos), progress=progress)
            for k in totals:
                totals[k] += res[k]
            if idx % 5 == 0 or idx == len(combos):
                store.refresh_exports(excel=True)

    store.refresh_exports(excel=True)

    if calibrate:
        print(f">>> Campaign finished | runs done: {totals['done']}, skipped (resume): {totals['skipped']}, failed: {totals['failed']}",
              flush=True)
        if totals['done'] + totals['skipped'] == 0 and totals['failed'] > 0:
            raise RuntimeError("Every run of the campaign failed - see the 'error' column of tables/runs.csv "
                               "and the tracebacks above.")
        try:
            RUN_MODEL_COMPARISON(system_data=system_data, store=store, rank_on=rank_on, rank_dataset=rank_dataset,
                                 top_frac=top_frac, baseline=baseline)
        except Exception as exc:
            traceback.print_exc()
            print(f">>> Comparison failed ({exc!r}); the runs are safe. Redo it with "
                  f"RUN_MODEL_COMPARISON(system_data, ResultsStore.open(r'{store.dir}'))", flush=True)
    return store