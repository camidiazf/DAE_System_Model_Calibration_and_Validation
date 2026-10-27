"""
Results_store.py
================
One folder per calibration CAMPAIGN (= one RUN_COMBOS call). Tidy (long-format) CSV tables are the source of truth;
everything else (wide summary, Excel, figures) is derived from them.

    <folder_model>/<folder_model>_Results/<campaign>/
    |-- campaign.json            what was run: base parameters, bounds, PSO settings, cost scaling, seed base, versions
    |-- README.md                layout + table dictionary (written once)
    |-- tables/
    |     runs.csv               1 row per run  (status done|failed, seed, cost, bound hits, time, error)
    |     combos.csv             1 row per parameter combination (parameters, lb, ub)
    |     parameters.csv         long: run_id, parameter, value, lb, ub, calibrated
    |     metrics.csv            long: run_id, condition, dataset(cal|val), variable|global, RMSE, NRMSE, MAPE, DW, AD, AIC, BIC, AICc
    |     identifiability.csv    long: run_id, condition, parameter, theta, t_value, SE, CI, rel.error, cond(FIM), fitted_on
    |     correlations.csv       long: run_id, condition, param_1, param_2, kappa
    |     sensitivity.csv        long: run_id, condition, parameter, variable, general relative sensitivity
    |     residuals.csv          long: run_id, condition, dataset, variable, t, range-scaled residual
    |     pso_history.csv        long: run_id, epoch, best-so-far cost
    |     summary_wide.csv/.xlsx derived: 1 row per run (what you paste in a paper table)
    |-- runs/<run_id>/           run_id = <combo_id>__rNN  (stable: does not depend on the order of the combinations)
    |     run.json               written LAST = "done" marker (resume skips runs that have it)
    |     solutions/<cond>_<cal|val>.csv     simulated trajectories on time_stamps_sim
    |     figures/<cond>/        solution.png (calibration | validation), residuals_cal.png, residuals_val.png,
    |                            sensitivity.png, corr_matrix.png
    |-- combos/<combo_id>/       mean +- sd over the iterations of one combination (only when all iterations ran in one session)
    |-- comparison/              all-models figures and tables (Comparison_functions.py)

run_id 'original' = the model with the starting parameters (no calibration); run_id 'baseline' = the reference model
(different structure), both stored like any other run so every figure/table reads them the same way.

NOTE (not implemented, may be needed at some point): importing results produced by the OLD RUN_COMBOS (two wide CSVs
'normal<model>_<campaign>_data.csv' / 'N_def<model>_<campaign>_data.csv', one row per run, parameters in columns named
after the parameter, 't_value_<param>', 'NRMSE_<var>', 'Cal_NRMSE', 'Cal_cost', 'PSO_seed', ...). A converter
`import_legacy_csv(store, csv_normal, csv_N_def)` would have to (1) rebuild run_ids from the 'Model' column (they depend
on the order of the combinations, so keep them as legacy_<Model>), (2) melt the parameter / t-value / metric columns into
the long tables above, (3) leave solutions/figures empty (trajectories were never stored; they would have to be
re-simulated from the parameters), (4) swap the old 'A-D' and 'D-W' columns if the files are older than the header fix.
"""
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CONDITIONS = ('normal', 'N_def')                       # internal keys (ext = '_' + key in System_data)
DATASETS = ('cal', 'val')                              # calibration (parameter estimation) / validation experiment
COND_LABEL = {'normal': 'Nutrient replete', 'N_def': 'Ammonium deficient'}
DATASET_LABEL = {'cal': 'Calibration', 'val': 'Validation'}

TABLE_SCHEMAS: Dict[str, List[str]] = {
    'runs': ['run_id', 'combo_id', 'iteration', 'n_params', 'calibrated_parameters', 'seed', 'cost', 'bound_hits',
             'n_failed_evals', 'n_evals', 'time_s', 'status', 'error', 'timestamp'],
    'combos': ['combo_id', 'n_params', 'parameters', 'lb', 'ub'],
    'parameters': ['run_id', 'parameter', 'value', 'lb', 'ub', 'calibrated'],
    'metrics': ['run_id', 'condition', 'dataset', 'fitted', 'variable', 'n', 'm', 'RMSE', 'NRMSE', 'MAPE', 'DW', 'AD',
                'AIC', 'BIC', 'AICc'],
    'identifiability': ['run_id', 'condition', 'fitted_on', 'parameter', 'theta', 't_value', 'SE', 'CI_low', 'CI_high',
                        'rel_error_pct', 'cond_FIM', 'rank_ratio', 'n_highly_corr'],
    'correlations': ['run_id', 'condition', 'param_1', 'param_2', 'kappa', 'high'],
    'sensitivity': ['run_id', 'condition', 'parameter', 'variable', 'sensitivity'],
    'residuals': ['run_id', 'condition', 'dataset', 'variable', 't', 'residual_scaled'],
    'pso_history': ['run_id', 'epoch', 'best_cost'],
}
TABLE_KEY = {'combos': 'combo_id'}                     # every other table is keyed by run_id

# campaign.json entries that must not change when a campaign is resumed (otherwise old and new runs are not comparable)
COMPAT_KEYS = ['parameters_and_bounds', 'base_parameters', 'pso', 'cost_scaling', 'calibration_condition']

README_TEXT = """# Calibration campaign

* `campaign.json`  everything needed to reproduce the runs (base parameters, bounds, PSO settings, cost scaling, seed base, versions).
* `tables/`        tidy CSV tables = source of truth (one row per run / parameter / metric ...). `summary_wide.csv` and
                   `summary.xlsx` are DERIVED (one row per run) and are rebuilt automatically.
* `runs/<run_id>/` one folder per run: `run.json` (written last = done marker), `solutions/` (trajectories), `figures/<condition>/`.
* `combos/`        mean +- sd over the iterations of one parameter combination.
* `comparison/`    all-models figures and tables.

run_id = `<combo_id>__rNN`; combo_id = `k<number of calibrated parameters>_<hash of parameters+bounds>` (see tables/combos.csv).
condition: `normal` = nutrient replete, `N_def` = ammonium deficient. dataset: `cal` = calibration (parameter estimation)
experiment, `val` = validation experiment. `fitted` = TRUE only for the calibration dataset of the condition the PSO was fitted on.
Metrics are computed on range-scaled residuals (see Validation_functions.py); NRMSE global = RMS of all scaled residuals.
"""


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def jsonable(o: Any) -> Any:
    """Recursively convert numpy / pandas / non-finite values into something json can write."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is None or isinstance(o, str):
        return o
    return str(o)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(jsonable(obj), fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def environment_info() -> Dict[str, Any]:
    """Versions of the packages that matter for reproducibility (+ git hash of the package folder if it is a repo)."""
    info: Dict[str, Any] = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for mod in ('numpy', 'pandas', 'scipy', 'mealpy', 'casadi', 'statsmodels'):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        info['git_hash'] = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        info['git_hash'] = None
    return info


def condition_label(system_data: Dict[str, Any], condition: str) -> str:
    """
    Display name of a condition (figure titles only). Optional System_data entry:
        condition_labels = {'normal': 'Nutrient-replete', 'N_def': 'Ammonium-deficient'}
    The keys 'normal' / 'N_def' themselves are fixed: they are the suffixes of the System_data entries ('_normal', '_N_def').
    Without the entry the defaults of COND_LABEL are used.
    """
    names = system_data.get('condition_labels') or {}
    return names.get(condition) or COND_LABEL[condition]


def dataset_label(system_data: Dict[str, Any], condition: str, dataset: str) -> str:
    """
    Name of an experimental dataset for titles. Optional System_data entry (3 lines):
        dataset_names = {'normal': {'cal': 'PE_Normal',  'val': 'V_Normal'},
                         'N_def':  {'cal': 'PE_Amm_def', 'val': 'V_Amm_def_rep'}}
    Without it: '<condition label> <calibration|validation> dataset'.
    """
    names = system_data.get('dataset_names') or {}
    name = (names.get(condition) or {}).get(dataset)
    return name if name else f"{condition_label(system_data, condition)} {DATASET_LABEL[dataset].lower()} dataset"


def _config_diffs(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    diffs = []
    for k in COMPAT_KEYS:
        if json.dumps(jsonable(old.get(k)), sort_keys=True) != json.dumps(jsonable(new.get(k)), sort_keys=True):
            diffs.append(k)
    return diffs


# ----------------------------------------------------------------------------------------------
# the store
# ----------------------------------------------------------------------------------------------
class ResultsStore:
    def __init__(self, campaign_dir: str):
        self.dir = os.path.abspath(campaign_dir)
        self.tables_dir = os.path.join(self.dir, 'tables')
        self.runs_dir = os.path.join(self.dir, 'runs')
        self.combos_dir = os.path.join(self.dir, 'combos')
        self.comparison_dir = os.path.join(self.dir, 'comparison')

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def open(cls, campaign_dir: str) -> 'ResultsStore':
        """Open an existing campaign (read access for comparison / exports)."""
        store = cls(campaign_dir)
        if not os.path.exists(os.path.join(store.dir, 'campaign.json')):
            raise FileNotFoundError(f"{store.dir} is not a campaign folder (no campaign.json).")
        return store

    def init_campaign(self, config: Dict[str, Any], resume: bool = False, allow_config_change: bool = False) -> None:
        """
        New campaign: creates the folders and campaign.json. Existing campaign: refuses unless resume=True, and on
        resume refuses if parameters/bounds/PSO/cost settings differ (unless allow_config_change=True).
        """
        cfg_path = os.path.join(self.dir, 'campaign.json')
        session = {'start': now_iso(), 'iterations': config.get('iterations'), 'min_k': config.get('min_k'),
                   'max_k': config.get('max_k'), 'solve_calibrate': config.get('solve_calibrate')}
        if os.path.exists(cfg_path):
            if not resume:
                raise FileExistsError(
                    f"Campaign '{self.dir}' already exists. Use resume=True to continue it (finished runs are skipped) "
                    f"or choose a new folder_results name.")
            old = _read_json(cfg_path)
            diffs = _config_diffs(old, jsonable(config))
            if diffs and not allow_config_change:
                raise ValueError(f"Cannot resume: these settings differ from campaign.json: {diffs}. "
                                 f"Use a new campaign name or allow_config_change=True (old and new runs then differ).")
            old.setdefault('sessions', []).append(session)
            if diffs:
                old.setdefault('config_changes', []).append({'time': now_iso(), 'keys': diffs})
            _write_json(cfg_path, old)
        else:
            if os.path.isdir(self.dir) and os.listdir(self.dir):
                raise FileExistsError(f"'{self.dir}' exists and is not empty but has no campaign.json.")
            for d in (self.tables_dir, self.runs_dir, self.combos_dir, self.comparison_dir):
                os.makedirs(d, exist_ok=True)
            cfg = dict(config)
            cfg['created'] = now_iso()
            cfg['environment'] = environment_info()
            cfg['sessions'] = [session]
            _write_json(cfg_path, cfg)
            with open(os.path.join(self.dir, 'README.md'), 'w', encoding='utf-8') as fh:
                fh.write(README_TEXT)
        for d in (self.tables_dir, self.runs_dir, self.combos_dir, self.comparison_dir):
            os.makedirs(d, exist_ok=True)

    def read_campaign(self) -> Dict[str, Any]:
        return _read_json(os.path.join(self.dir, 'campaign.json'))

    # ---- ids and paths ------------------------------------------------------------------------
    @staticmethod
    def combo_id(params: Sequence[str], lb: Sequence[float], ub: Sequence[float]) -> str:
        """Stable id of a parameter combination WITH its bounds: k<n>_<hash>. Independent of the order of combinations."""
        items = sorted(zip(params, lb, ub), key=lambda x: x[0])
        key = json.dumps([[p, float(l), float(u)] for p, l, u in items])
        return f"k{len(params)}_{hashlib.sha1(key.encode()).hexdigest()[:6]}"

    @staticmethod
    def run_id(combo_id: str, iteration: int) -> str:
        """iteration is 0-based (as in the loops); run ids are 1-based: <combo_id>__r01."""
        return f"{combo_id}__r{iteration + 1:02d}"

    @staticmethod
    def combo_hash_int(combo_id: str) -> int:
        """Integer derived from the combo hash (used for order-independent PSO seeds)."""
        return int(combo_id.split('_')[1], 16)

    def table_path(self, name: str) -> str:
        return os.path.join(self.tables_dir, f"{name}.csv")

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, run_id)

    def figures_dir(self, run_id: str, condition: Optional[str] = None) -> str:
        d = os.path.join(self.run_dir(run_id), 'figures', condition) if condition else os.path.join(self.run_dir(run_id), 'figures')
        os.makedirs(d, exist_ok=True)
        return d

    def combo_dir(self, combo_id: str) -> str:
        d = os.path.join(self.combos_dir, combo_id)
        os.makedirs(d, exist_ok=True)
        return d

    def comparison_path(self) -> str:
        os.makedirs(self.comparison_dir, exist_ok=True)
        return self.comparison_dir

    # ---- runs ---------------------------------------------------------------------------------
    def is_done(self, run_id: str) -> bool:
        p = os.path.join(self.run_dir(run_id), 'run.json')
        if not os.path.exists(p):
            return False
        try:
            return _read_json(p).get('status') == 'done'
        except Exception:
            return False

    def register_combo(self, combo_id: str, params: Sequence[str], lb: Sequence[float], ub: Sequence[float]) -> None:
        self._upsert('combos', combo_id, [{'combo_id': combo_id, 'n_params': len(params), 'parameters': ';'.join(params),
                                           'lb': ';'.join(f"{float(v):.10g}" for v in lb),
                                           'ub': ';'.join(f"{float(v):.10g}" for v in ub)}])

    def save_run(self, run_id: str, run_info: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]],
                 done: bool = True) -> None:
        """
        Upserts the rows of this run in every table (rows of the same run_id are replaced -> idempotent), then writes
        run.json LAST. done=False (failed run): tables are written, run.json is not, so a resume retries it.
        """
        os.makedirs(self.run_dir(run_id), exist_ok=True)
        for name, rows in tables.items():
            self._upsert(name, run_id, rows)
        if done:
            info = dict(run_info)
            info['run_id'] = run_id
            info['status'] = 'done'
            _write_json(os.path.join(self.run_dir(run_id), 'run.json'), info)

    def save_solution(self, run_id: str, condition: str, dataset: str, sol: Any, var_names: Sequence[str],
                      t_grid: Sequence[float]) -> Optional[str]:
        """Trajectory of one (condition, dataset) on the common time grid -> runs/<run_id>/solutions/<cond>_<ds>.csv."""
        if sol is None:
            return None
        df = pd.DataFrame({'t': np.asarray(t_grid, dtype=float)})
        for v in var_names:
            df[v] = np.asarray(sol[v], dtype=float)
        d = os.path.join(self.run_dir(run_id), 'solutions')
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{condition}_{dataset}.csv")
        df.to_csv(path, index=False)
        return path

    def load_solution(self, run_id: str, condition: str, dataset: str,
                      t_grid: Optional[Sequence[float]] = None) -> Optional[pd.DataFrame]:
        """Stored trajectory (DataFrame with 't' + states); interpolated onto t_grid if given and different."""
        path = os.path.join(self.run_dir(run_id), 'solutions', f"{condition}_{dataset}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if t_grid is not None:
            tg = np.asarray(t_grid, dtype=float)
            if len(tg) != len(df) or not np.allclose(tg, df['t'].values):
                out = {'t': tg}
                for c in df.columns:
                    if c != 't':
                        out[c] = np.interp(tg, df['t'].values, df[c].values)
                df = pd.DataFrame(out)
        return df

    # ---- tables -------------------------------------------------------------------------------
    def read(self, name: str) -> pd.DataFrame:
        path = self.table_path(name)
        if not os.path.exists(path):
            return pd.DataFrame(columns=TABLE_SCHEMAS[name])
        return pd.read_csv(path)

    def _upsert(self, name: str, key: str, rows: List[Dict[str, Any]]) -> None:
        cols = TABLE_SCHEMAS[name]
        key_col = TABLE_KEY.get(name, 'run_id')
        for r in rows:
            extra = set(r) - set(cols)
            if extra:
                raise KeyError(f"Table '{name}': unexpected columns {sorted(extra)}")
        new = pd.DataFrame(rows, columns=cols)
        os.makedirs(self.tables_dir, exist_ok=True)
        path = self.table_path(name)
        if os.path.exists(path):
            old = pd.read_csv(path)
            old = old[old[key_col].astype(str) != str(key)]
            if new.empty:
                df = old
            elif len(old):
                df = pd.concat([old, new], ignore_index=True)
            else:
                df = new
        else:
            df = new
        tmp = path + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    # ---- derived outputs ----------------------------------------------------------------------
    def build_summary_wide(self) -> pd.DataFrame:
        """
        One row per finished run: run info, calibrated parameter values, global/per-variable NRMSE, AIC, BIC for
        every (condition, dataset), and the identifiability summary of the condition the run was FITTED on
        (min t, max |kappa|, FIM condition number), i.e. the columns of the selection rule.
        """
        runs = self.read('runs')
        runs = runs[runs['status'] == 'done']
        if runs.empty:
            return pd.DataFrame()
        met = self.read('metrics')
        par = self.read('parameters')
        ide = self.read('identifiability')
        cor = self.read('correlations')

        base_cols = ['run_id', 'combo_id', 'iteration', 'n_params', 'calibrated_parameters', 'seed', 'cost', 'bound_hits',
                     'n_failed_evals', 'time_s']
        wide = runs[base_cols].copy().set_index('run_id')

        # calibrated parameter values (wide, union of the calibrated parameters)
        pc = par[par['calibrated'].astype(str).str.lower() == 'true']
        if not pc.empty:
            order = list(dict.fromkeys(pc['parameter']))
            pv = pc.pivot_table(index='run_id', columns='parameter', values='value', aggfunc='first')
            wide = wide.join(pv[order].add_prefix('p_'))

        # metrics: global + per variable, every (condition, dataset)
        if not met.empty:
            m = met.copy()
            m['col'] = m['condition'] + '_' + m['dataset']
            g = m[m['variable'] == 'global']
            for stat in ('NRMSE', 'AIC', 'BIC'):
                pv = g.pivot_table(index='run_id', columns='col', values=stat, aggfunc='first')
                wide = wide.join(pv.add_suffix(f'_{stat}'))
            v = m[m['variable'] != 'global'].copy()
            if not v.empty:
                v['col'] = v['col'] + '_NRMSE_' + v['variable']
                pv = v.pivot_table(index='run_id', columns='col', values='NRMSE', aggfunc='first')
                wide = wide.join(pv)

        # identifiability of the fitted condition
        fitted_ide = ide[ide['fitted_on'].astype(str).str.lower() == 'true']
        if not fitted_ide.empty:
            agg = fitted_ide.groupby('run_id').agg(min_t=('t_value', 'min'), cond_FIM=('cond_FIM', 'first'))
            wide = wide.join(agg)
            fitted_cor = cor[cor['condition'].isin(fitted_ide['condition'].unique())]
            if not fitted_cor.empty:
                k = fitted_cor.assign(abs_kappa=fitted_cor['kappa'].abs()).groupby('run_id')['abs_kappa'].max()
                wide = wide.join(k.rename('max_abs_kappa'))
        return wide.reset_index()

    def refresh_exports(self, excel: bool = True) -> None:
        """Rebuild tables/summary_wide.csv and tables/summary.xlsx from the long tables (cheap; safe to call any time)."""
        try:
            wide = self.build_summary_wide()
            if not wide.empty:
                wide.to_csv(os.path.join(self.tables_dir, 'summary_wide.csv'), index=False)
            if excel and not wide.empty:
                path = os.path.join(self.tables_dir, 'summary.xlsx')
                tmp = path + '.tmp.xlsx'
                with pd.ExcelWriter(tmp) as xw:
                    pd.DataFrame({'README': README_TEXT.splitlines()}).to_excel(xw, sheet_name='README', index=False)
                    wide.to_excel(xw, sheet_name='summary_wide', index=False)
                    for name in ('runs', 'combos', 'parameters', 'metrics', 'identifiability', 'correlations'):
                        self.read(name).to_excel(xw, sheet_name=name, index=False)
                os.replace(tmp, path)
        except ImportError as e:
            print(f">>> Excel export skipped ({e}); CSV tables are complete.")
        except Exception as e:                        # an export problem must never kill a long campaign
            print(f">>> Export of summary tables failed: {e!r}")