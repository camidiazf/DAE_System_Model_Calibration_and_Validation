import os
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DAE_analysis_package.Plotting_functions import _ensure_dir
from DAE_analysis_package.Validation_functions import residuals_error_equations, model_error_equations

"""
Best-model analysis (top X % of the calibrated runs).

Reads the two CSVs written by RUN_COMBINATION (nutrient replete = 'normal', nutrient deficient = 'N_def'),
selects the best top_frac of the runs by OVERALL (global) NRMSE -- one ranking for the calibration data and one,
separately, for the validation data -- re-simulates them from the parameters stored in the CSV, and produces:
    1) solution comparison figure, one line per model      (rows = states, columns = replete | deficient)
    2) solution comparison figure, shaded band             (min-max range of the top models + rank-1 line)
    3) metric bar figure                                   (global NRMSE, per-variable NRMSE, AIC/BIC)
    4) CSV tables: ranking + parameters, and the reference-model metrics
Optionally a BASELINE model of a different structure (e.g. the Monod model) is added to all the figures: its metrics
are recomputed with the same functions/scaling as the rest (see the 'baseline' argument of RUN_BEST_MODELS).

Works only from the CSVs, so it can also be called on its own after the calibration has finished.
"""

CONDITIONS = ('normal', 'N_def')
COND_LABEL = {'normal': 'Nutrient replete', 'N_def': 'Nutrient deficient'}
DATASET_LABEL = {'cal': 'Calibration', 'val': 'Validation'}
ORIGINAL_PREFIX = 'Model_0_'          # key of the un-calibrated (original) model row

# x-axis upper limit (h) per dataset: calibration experiment 0-50 h, validation experiment 0-100 h
XMAX = {'cal': 50, 'val': 100}
BAND_COLOR = "#467dd6"      # shaded range
LINE_COLOR = "#00348d"      # best-model line (band figure)
BASE_COLOR = "#f37859"      # baseline model (not in the tab10 palette used for the top models)


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _num(x):
    """Numeric coercion (the CSV holds ' ' placeholders and None for undefined metrics)."""
    return pd.to_numeric(x, errors='coerce')


def _get(row, cond: str, col: str) -> float:
    """Scalar from a merged row; NaN if the row/column does not exist."""
    if row is None or (cond, col) not in row.index:
        return np.nan
    return float(_num(row[(cond, col)]))


def _model_labels(index) -> List[str]:
    """('Model_12_3', 0) -> 'M12_3'; repeated names (CSV appended by several sessions) get '#occurrence'."""
    return [k.replace('Model_', 'M') + ('' if occ == 0 else f'#{occ}') for k, occ in index]


def _model_colors(n: int) -> list:
    """Distinct colours for <=10 models; grey ramp (dark = best rank) otherwise."""
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    return [plt.cm.Greys(v) for v in np.linspace(0.9, 0.35, n)]


def _original_mask(merged: pd.DataFrame) -> np.ndarray:
    return np.asarray(merged.index.get_level_values('key').str.startswith(ORIGINAL_PREFIX))


def _shared_legend(fig, axes) -> None:
    handles = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc='lower center',
               ncol=min(len(handles), 6), frameon=False, fontsize=10)


# ----------------------------------------------------------------------------------------------
# Loading, ranking, selection
# ----------------------------------------------------------------------------------------------
def load_runs(csv_path_normal: str, csv_path_N_def: str) -> pd.DataFrame:
    """
    Merge the replete ('normal') and deficient ('N_def') CSVs row by row (same model = same run).
    Result has MultiIndex columns (condition, column) and MultiIndex rows (key, occurrence).
    """
    frames = {}
    for cond, path in (('normal', csv_path_normal), ('N_def', csv_path_N_def)):
        df = pd.read_csv(path)
        df['key'] = df['Model'].astype(str).str.replace(f'^{cond}_', '', regex=True)
        df['occ'] = df.groupby('key').cumcount()          # robust to duplicated names if the CSV was appended to
        frames[cond] = df.set_index(['key', 'occ'])
    common = frames['normal'].index.intersection(frames['N_def'].index)
    merged = pd.concat({c: frames[c].loc[common] for c in CONDITIONS}, axis=1).sort_index()

    needed = ['NRMSE', 'Cal_NRMSE']
    for cond in CONDITIONS:
        missing = [c for c in needed if (cond, c) not in merged.columns]
        if missing:
            raise KeyError(f"CSV of '{cond}' lacks columns {missing}: it was written by an older version of the code. "
                           f"Re-run with a new folder_results name.")
    return merged


def _score(pool: pd.DataFrame, dataset: str, rank_on: str) -> pd.Series:
    """Overall NRMSE used for ranking. rank_on: 'both' (mean of replete and deficient), 'normal' or 'N_def'."""
    if rank_on not in ('both', 'normal', 'N_def'):
        raise ValueError(f"rank_on must be 'both', 'normal' or 'N_def' (got {rank_on!r})")
    col = 'Cal_NRMSE' if dataset == 'cal' else 'NRMSE'
    s_n = _num(pool[('normal', col)])
    s_d = _num(pool[('N_def', col)])
    if rank_on == 'normal':
        return s_n
    if rank_on == 'N_def':
        return s_d
    return (s_n + s_d) / 2.0


def select_best(merged: pd.DataFrame, dataset: str, rank_on: str = 'both', top_frac: float = 0.10
                ) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Top top_frac (at least 1 model) of the calibrated runs (original model excluded), lowest NRMSE first."""
    pool = merged[~_original_mask(merged)]
    score = _score(pool, dataset, rank_on)
    score = score[np.isfinite(score)].sort_values()
    n_total = len(score)
    n_top = max(1, int(np.ceil(top_frac * n_total))) if n_total else 0
    score = score.iloc[:n_top]
    return pool.loc[score.index], score, n_total


def _rebuild_parameters(row: pd.Series, system_data: Dict[str, any]) -> Dict[str, float]:
    """Original parameters, overridden by the calibrated values stored in the CSV (empty cell = not calibrated)."""
    params = dict(system_data['parameters'])
    for p in system_data['parameters_og_list']:
        if ('N_def', p) in row.index:
            v = _num(row[('N_def', p)])
            if np.isfinite(v):
                params[p] = float(v)
    return params


# ----------------------------------------------------------------------------------------------
# Simulation of the selected models
# ----------------------------------------------------------------------------------------------
def _simulate_set(system_data: Dict[str, any], sims: Dict[str, Callable], params_list: List[Dict[str, float]],
                  dataset: str) -> Dict[str, list]:
    """Smooth solutions (time_stamps_sim) in both conditions, from the x0 of the calibration or validation experiment."""
    t = system_data['time_stamps_sim']
    out = {}
    for cond in CONDITIONS:
        ext = '_' + cond
        x0 = system_data['x0_exp' + ext] if dataset == 'cal' else system_data['x0_exp_v' + ext]
        out[cond] = []
        for p in params_list:
            try:
                sol = sims[cond](system_data=system_data, simulation_type='normal', x0=x0, parameters=p, time=t)
            except Exception as e:
                print(f"!!!!!!!!!!!!! Simulation failed ({cond}, {dataset}): {e!r}")
                sol = None
            out[cond].append(sol)
    return out


def _exp_data(system_data: Dict[str, any], cond: str, dataset: str):
    ext = '_' + cond
    if dataset == 'cal':
        return system_data['t_exp' + ext], system_data['df_exp' + ext]
    return system_data['t_exp_v' + ext], system_data['df_val' + ext]


# ----------------------------------------------------------------------------------------------
# Baseline model (different structure, e.g. the Monod model)
# ----------------------------------------------------------------------------------------------
def _baseline_sd(baseline: Dict[str, any], system_data: Dict[str, any]) -> Dict[str, any]:
    """System data of the baseline; defaults to the main one (same experiments)."""
    return baseline.get('system_data', system_data)


def _baseline_params(baseline: Dict[str, any], cond: str) -> Dict[str, float]:
    """Either one parameter dict for both conditions, or {'normal': {...}, 'N_def': {...}}."""
    p = baseline['parameters']
    return dict(p[cond]) if cond in p else dict(p)


def _baseline_sims(baseline: Dict[str, any], system_data: Dict[str, any], dataset: str) -> Dict[str, any]:
    """Smooth baseline solutions for the plots (same time grid and x0 as the other models)."""
    if 'solutions' in baseline:                              # pre-computed trajectories (e.g. read from Excel)
        return _baseline_sims_precomputed(baseline, system_data, dataset)
    sd = _baseline_sd(baseline, system_data)
    t = system_data['time_stamps_sim']
    out = {}
    for cond in CONDITIONS:
        ext = '_' + cond
        x0 = sd['x0_exp' + ext] if dataset == 'cal' else sd['x0_exp_v' + ext]
        try:
            out[cond] = baseline['simulate_model'](system_data=sd, simulation_type='normal', x0=x0,
                                                           parameters=_baseline_params(baseline, cond), time=t)
        except Exception as e:
            print(f"!!!!!!!!!!!!! Baseline simulation failed ({cond}, {dataset}): {e!r}")
            out[cond] = None
    return out


def _baseline_metrics(baseline: Dict[str, any], system_data: Dict[str, any], dataset: str) -> Dict[str, Dict[str, float]]:
    """
    Baseline metrics on the SAME experimental points and with the SAME formulas as the rest of the package
    (range-scaled residuals, AIC/BIC with K = n_calibrated + 1). AIC/BIC stay NaN if n_calibrated is not given.
    """
    if 'metrics' in baseline:                                # pre-computed metrics (e.g. read from Excel)
        return _baseline_metrics_precomputed(baseline, system_data, dataset)
    sd = _baseline_sd(baseline, system_data)
    var_exp = system_data['var_exp_names']
    m = int(baseline.get('n_calibrated', 0))
    out = {}
    for cond in CONDITIONS:
        ext = '_' + cond
        if dataset == 'cal':
            x0, t, df = sd['x0_exp' + ext], sd['t_exp' + ext], sd['df_exp' + ext]
        else:
            x0, t, df = sd['x0_exp_v' + ext], sd['t_exp_v' + ext], sd['df_val' + ext]
        d = {'NRMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}
        d.update({'NRMSE_' + v: np.nan for v in var_exp})
        try:
            sol = baseline['simulate_model'](system_data=sd, simulation_type='normal', x0=x0,
                                                     parameters=_baseline_params(baseline, cond), time=t)
            y_val = [np.asarray(df[v].values, float) for v in var_exp]
            y_sim = [np.asarray(sol[v].values, float) for v in var_exp]
            for v, yv, ys in zip(var_exp, y_val, y_sim):
                d['NRMSE_' + v] = residuals_error_equations(yv, ys)['NRMSE']
            mod = model_error_equations(y_val, y_sim, m)
            d['NRMSE'] = mod['NRMSE_global']
            d['AIC'] = np.nan if mod['AIC'] is None else mod['AIC']
            d['BIC'] = np.nan if mod['BIC'] is None else mod['BIC']
        except Exception as e:
            print(f"!!!!!!!!!!!!! Baseline metrics failed ({cond}, {dataset}): {e!r}")
        out[cond] = d
    return out

def _baseline_sims_precomputed(baseline: Dict[str, any], system_data: Dict[str, any], dataset: str) -> Dict[str, any]:
    """
    Stored trajectories {dataset: {condition: DataFrame with column 't'}} interpolated onto time_stamps_sim.
    A dataset without stored trajectories gives NO line (a trajectory from another x0 is never drawn instead).
    """
    t = np.asarray(system_data['time_stamps_sim'], float)
    stored = baseline['solutions'].get(dataset) or {}
    out = {}
    for cond in CONDITIONS:
        df = stored.get(cond)
        if df is None:
            out[cond] = None
            continue
        tb = np.asarray(df['t'], float)
        inside = (t >= tb.min()) & (t <= tb.max())
        out[cond] = pd.DataFrame({c: np.where(inside, np.interp(t, tb, np.asarray(df[c], float)), np.nan)
                                  for c in df.columns if c != 't'})
    return out


def _baseline_metrics_precomputed(baseline: Dict[str, any], system_data: Dict[str, any], dataset: str
                                  ) -> Dict[str, Dict[str, float]]:
    """Stored metrics {dataset: {condition: {'NRMSE','AIC','BIC','NRMSE_<var>'}}}; missing entries stay NaN."""
    var_exp = system_data['var_exp_names']
    stored = baseline['metrics'].get(dataset) or {}
    out = {}
    for cond in CONDITIONS:
        d = {'NRMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}
        d.update({'NRMSE_' + v: np.nan for v in var_exp})
        d.update({k: v for k, v in (stored.get(cond) or {}).items() if k in d})
        out[cond] = d
    return out


def load_baseline_xlsx(path: str, label: str = 'Baseline (Monod)', variant: str = 'calibrated',
                       metrics_sheet: str = 'monod', solutions: Optional[Dict[str, str]] = None,
                       cond_names: Optional[Dict[str, str]] = None) -> Dict[str, any]:
    """
    Build the 'baseline' dict for RUN_BEST_MODELS from an Excel workbook that holds
      * a metrics sheet (same columns as the CSVs of RUN_COMBINATION), one row per '<cond>_Model_<variant>'
            variant = 'calibrated'   -> rows normal_Model_calibrated / N_def_Model_calibrated      (default)
                      'pre_calibrated' -> rows normal_Model_pre_calibrated / N_def_Model_pre_calibrated
      * one solution sheet per condition (column 't' + one column per state), named by the patterns in `solutions`
            solutions = {'val': 'Monod_{tag}_{cond}'}   tag = 'cal' (calibrated) or 'base' (pre-calibrated),
                                                         cond = cond_names[...]
            -> the sheets are the trajectories of ONE experiment, so only the datasets listed here get a line.
               Add e.g. 'cal': 'Monod_{tag}_{cond}_calx0' if you also have the calibration-experiment trajectories.
    Metrics are used as stored (they already come from the package's own functions, hence comparable).
    """
    solutions = {'val': 'Monod_{tag}_{cond}'} if solutions is None else solutions
    cond_names = {'normal': 'Nutrient replete', 'N_def': 'Ammonium deficient'} if cond_names is None else cond_names
    if variant not in ('calibrated', 'pre_calibrated'):
        raise ValueError(f"variant must be 'calibrated' or 'pre_calibrated' (got {variant!r})")
    tag = 'cal' if variant == 'calibrated' else 'base'

    tab = pd.read_excel(path, sheet_name=metrics_sheet)
    tab = tab[tab['Model'].notna()].drop_duplicates('Model', keep='last').set_index('Model')

    metrics = {}
    for dataset, pre in (('cal', 'Cal_'), ('val', '')):
        metrics[dataset] = {}
        for cond in CONDITIONS:
            key = f'{cond}_Model_{variant}'
            if key not in tab.index:
                raise KeyError(f"Row '{key}' not found in sheet '{metrics_sheet}'. Available: {list(tab.index)}")
            row = tab.loc[key]
            d = {k: float(_num(row[pre + k])) for k in ('NRMSE', 'AIC', 'BIC') if pre + k in row.index}
            for col in tab.columns:
                if col.startswith(pre + 'NRMSE_'):
                    d['NRMSE_' + col[len(pre + 'NRMSE_'):]] = float(_num(row[col]))
            metrics[dataset][cond] = d

    sols = {}
    for dataset, pattern in solutions.items():
        sols[dataset] = {cond: pd.read_excel(path, sheet_name=pattern.format(tag=tag, cond=cond_names[cond]))
                         for cond in CONDITIONS}
    return {'label': label, 'metrics': metrics, 'solutions': sols}


def _row_metrics(row: Optional[pd.Series], pre: str, var_exp: List[str]) -> Dict[str, Dict[str, float]]:
    """Same structure as _baseline_metrics, read from a CSV row (original model)."""
    out = {}
    for cond in CONDITIONS:
        d = {'NRMSE': _get(row, cond, pre + 'NRMSE'), 'AIC': _get(row, cond, pre + 'AIC'),
             'BIC': _get(row, cond, pre + 'BIC')}
        for v in var_exp:
            d['NRMSE_' + v] = _get(row, cond, f'{pre}NRMSE_{v}')
        out[cond] = d
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _add_baseline_line(ax, t, m, base, cond, var) -> None:
    """Baseline line; silently skipped if the baseline has no such state (e.g. no quota q in the Monod model)."""
    if base is None or base['sols'].get(cond) is None:
        return
    try:
        yb = np.asarray(base['sols'][cond][var])[m]
    except Exception:
        return
    ax.plot(t[m], yb, '-', color=BASE_COLOR, lw=2.0, label=base['label'], zorder=4)


def plot_best_solutions(system_data: Dict[str, any], sols: Dict[str, list], orig: Dict[str, list],
                        labels: List[str], dataset: str, outdir: str, n_total: int, rank_on: str,
                        base: Optional[Dict[str, any]] = None) -> None:
    """Rows = states, columns = nutrient replete | nutrient deficient. One line per selected model."""
    var_names = system_data['var_names']
    var_exp = system_data['var_exp_names']
    t = np.asarray(system_data['time_stamps_sim'])
    tmax = XMAX[dataset]
    m = t <= tmax                                   # only plot (and autoscale y on) the visible window
    n = len(labels)
    colors = _model_colors(n)

    fig, axes = plt.subplots(len(var_names), 2, figsize=(14, 2.6 * len(var_names)), sharex='col', squeeze=False)
    for j, cond in enumerate(CONDITIONS):
        t_exp, df_exp = _exp_data(system_data, cond, dataset)
        t_exp = np.asarray(t_exp)
        me = t_exp <= tmax
        for i, var in enumerate(var_names):
            ax = axes[i, j]
            for k in range(n - 1, -1, -1):                       # rank 1 is drawn last (on top)
                sol = sols[cond][k]
                if sol is None:
                    continue
                if n <= 10:
                    lab = labels[k]
                else:
                    lab = f'{labels[0]} (rank 1)' if k == 0 else (f'Top {n} models (darker = better)' if k == 1 else '_nolegend_')
                ax.plot(t[m], np.asarray(sol[var])[m], '-', color=colors[k], lw=2.2 if k == 0 else 1.0,
                        alpha=1.0 if k == 0 else 0.8, label=lab, zorder=3 if k == 0 else 2)
            o = orig[cond][0]
            if o is not None:
                ax.plot(t[m], np.asarray(o[var])[m], 'k--', lw=1.5, label='Original model', zorder=4)
            _add_baseline_line(ax, t, m, base, cond, var)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], c='k', marker='o', s=22,
                           label='Experimental data', zorder=5)
            ax.set_xlim(0, tmax)
            ax.grid(True, alpha=0.4)
            if j == 0:
                ax.set_ylabel(var, fontsize=12)
            if i == 0:
                ax.set_title(COND_LABEL[cond], fontsize=14)
            if i == len(var_names) - 1:
                ax.set_xlabel('Time (h)', fontsize=12)

    _shared_legend(fig, axes)
    fig.suptitle(f"Top {n} of {n_total} models by {DATASET_LABEL[dataset].lower()} NRMSE "
                 f"(ranking: {rank_on}); data shown: {DATASET_LABEL[dataset].lower()} experiment", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(os.path.join(outdir, f"Best_{DATASET_LABEL[dataset]}_Solutions.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_best_solutions_band(system_data: Dict[str, any], sols: Dict[str, list], orig: Dict[str, list],
                             labels: List[str], dataset: str, outdir: str, n_total: int, rank_on: str,
                             base: Optional[Dict[str, any]] = None) -> None:
    """
    Same layout as plot_best_solutions, but the selected models are shown as a shaded band
    (pointwise min-max envelope across the top models) with the rank-1 model as the central line.
    NOTE: the band is the range spanned by the selected models, NOT a statistical confidence interval.
    """
    var_names = system_data['var_names']
    var_exp = system_data['var_exp_names']
    t = np.asarray(system_data['time_stamps_sim'])
    tmax = XMAX[dataset]
    m = t <= tmax
    n = len(labels)

    fig, axes = plt.subplots(len(var_names), 2, figsize=(14, 2.6 * len(var_names)), sharex='col', squeeze=False)
    for j, cond in enumerate(CONDITIONS):
        t_exp, df_exp = _exp_data(system_data, cond, dataset)
        t_exp = np.asarray(t_exp)
        me = t_exp <= tmax
        valid = [s for s in sols[cond] if s is not None]
        for i, var in enumerate(var_names):
            ax = axes[i, j]
            if n > 1 and valid:
                arr = np.vstack([np.asarray(s[var])[m] for s in valid])
                ax.fill_between(t[m], arr.min(axis=0), arr.max(axis=0), color=BAND_COLOR, alpha=0.25, lw=0,
                                label=f'Range of top {n} models', zorder=1)
            best = sols[cond][0]
            if best is not None:
                ax.plot(t[m], np.asarray(best[var])[m], '-', color=LINE_COLOR, lw=2.2,
                        label=f'Best model ({labels[0]})', zorder=3)
            o = orig[cond][0]
            if o is not None:
                ax.plot(t[m], np.asarray(o[var])[m], 'k--', lw=1.5, label='Original model', zorder=4)
            _add_baseline_line(ax, t, m, base, cond, var)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], facecolors='none', edgecolors='#003b9c',
                           marker='o', s=26, label='Experimental data', zorder=5)
            ax.set_xlim(0, tmax)
            ax.grid(True, alpha=0.4)
            if j == 0:
                ax.set_ylabel(var, fontsize=12)
            if i == 0:
                ax.set_title(COND_LABEL[cond], fontsize=14)
            if i == len(var_names) - 1:
                ax.set_xlabel('Time (h)', fontsize=12)

    _shared_legend(fig, axes)
    fig.suptitle(f"Top {n} of {n_total} models by {DATASET_LABEL[dataset].lower()} NRMSE (ranking: {rank_on}); "
                 f"shaded = min-max range of the top models; data: {DATASET_LABEL[dataset].lower()} experiment", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(os.path.join(outdir, f"Best_{DATASET_LABEL[dataset]}_Solutions_Band.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_best_metrics(system_data: Dict[str, any], sel: pd.DataFrame, refs: List[tuple],
                      labels: List[str], dataset: str, outdir: str, n_total: int, rank_on: str) -> None:
    """
    Bar graphs, columns = replete | deficient; rows = (a) global NRMSE, (b) NRMSE per measured variable,
    (c) AIC and BIC. Metrics are those of the dataset used for the selection.
    refs = [(name, metrics_by_condition, bar_style), ...] -> reference models drawn first (Original, Baseline).
    """
    pre = 'Cal_' if dataset == 'cal' else ''
    var_exp = system_data['var_exp_names']
    n = len(labels)
    r = len(refs)
    nb = r + n                                       # bars per group
    colors = _model_colors(n)
    xs = np.arange(nb)
    xt = [name for name, _, _ in refs] + labels
    rot = 90 if n > 6 else 45

    def ref_val(k, cond, key):
        return refs[k][1][cond][key]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), squeeze=False, sharey='row')
    for j, cond in enumerate(CONDITIONS):
        # (a) global NRMSE
        ax = axes[0, j]
        for k in range(r):
            ax.bar(xs[k], ref_val(k, cond, 'NRMSE'), **refs[k][2])
        ax.bar(xs[r:], _num(sel[(cond, pre + 'NRMSE')]).values, color=colors, edgecolor='black')
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=rot, ha='right', fontsize=8)
        ax.set_title(f"{COND_LABEL[cond]}: global NRMSE", fontsize=13)
        if j == 0:
            ax.set_ylabel('NRMSE (-)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)

        # (b) NRMSE per measured variable (groups = variables, bars = models)
        ax = axes[1, j]
        mv = len(var_exp)
        width = 0.8 / nb
        for k in range(nb):
            off = (k - (nb - 1) / 2) * width
            if k < r:
                vals = [ref_val(k, cond, 'NRMSE_' + v) for v in var_exp]
                kw = dict(refs[k][2], label=refs[k][0])
            else:
                vals = [float(_num(sel[(cond, f'{pre}NRMSE_{v}')].iloc[k - r])) for v in var_exp]
                kw = dict(color=colors[k - r], edgecolor='black', label=labels[k - r])
            ax.bar(np.arange(mv) + off, vals, width, **kw)
        ax.set_xticks(np.arange(mv))
        ax.set_xticklabels(var_exp, fontsize=11)
        ax.set_title(f"{COND_LABEL[cond]}: NRMSE per variable", fontsize=13)
        if j == 0:
            ax.set_ylabel('NRMSE (-)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)
        if nb <= 11 and j == 1:
            ax.legend(fontsize=8, ncol=2)

        # (c) AIC / BIC
        ax = axes[2, j]
        aic = [ref_val(k, cond, 'AIC') for k in range(r)] + list(_num(sel[(cond, pre + 'AIC')]).values)
        bic = [ref_val(k, cond, 'BIC') for k in range(r)] + list(_num(sel[(cond, pre + 'BIC')]).values)
        w = 0.4
        ax.bar(xs - w / 2, aic, w, color='lightgray', edgecolor='black', label='AIC')
        ax.bar(xs + w / 2, bic, w, color='dimgray', edgecolor='black', label='BIC')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=rot, ha='right', fontsize=8)
        ax.set_title(f"{COND_LABEL[cond]}: AIC / BIC", fontsize=13)
        if j == 0:
            ax.set_ylabel('AIC / BIC', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)
        if j == 1:
            ax.legend(fontsize=9)

    fig.suptitle(f"Metrics of the top {n} of {n_total} models by {DATASET_LABEL[dataset].lower()} NRMSE "
                 f"(ranking: {rank_on}); metrics on the {DATASET_LABEL[dataset].lower()} data", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, f"Best_{DATASET_LABEL[dataset]}_Metrics.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def _summary_table(system_data: Dict[str, any], sel: pd.DataFrame, score: pd.Series, labels: List[str]) -> pd.DataFrame:
    tab = pd.DataFrame({'rank': np.arange(1, len(labels) + 1),
                        'model': labels,
                        'param_combo': sel[('N_def', 'Param Combo')].values,
                        'rank_score_NRMSE': score.values})
    for cond in CONDITIONS:
        tab[f'Cal_NRMSE_{cond}'] = _num(sel[(cond, 'Cal_NRMSE')]).values
        tab[f'Val_NRMSE_{cond}'] = _num(sel[(cond, 'NRMSE')]).values
    for p in system_data['parameters_og_list']:               # only the parameters that were calibrated in at least one of these runs
        if ('N_def', p) in sel.columns:
            v = _num(sel[('N_def', p)])
            if v.notna().any():
                tab[p] = v.values
    return tab


def _reference_table(refs: List[tuple]) -> pd.DataFrame:
    rows = []
    for name, met, _ in refs:
        for cond in CONDITIONS:
            rows.append(dict(model=name, condition=cond, **met[cond]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------
def RUN_BEST_MODELS(system_data: Dict[str, any],
                    simulate_model: Callable,
                    csv_path_normal: str,
                    csv_path_N_def: str,
                    outdir: str,
                    top_frac: float = 0.10,
                    rank_on: str = 'both',
                    baseline: Optional[Dict[str, any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Parameters
        top_frac : fraction of calibrated runs kept (at least 1 model).
        rank_on  : 'both'  -> score = mean of the global NRMSE in replete and deficient conditions   [default]
                   'N_def' -> score = global NRMSE in the nutrient-deficient condition (the one the PSO is fitted to)
                   'normal'-> score = global NRMSE in the nutrient-replete condition
        baseline : optional model of a different structure (e.g. Monod) to add to every figure. Dict with
                     'label'                 : name in legends/axes, e.g. 'Baseline (Monod)'
                     'simulate_model' : its simulation function (same call signature as yours)
                     'parameters'            : FULL parameter dict of the calibrated baseline, or
                                               {'normal': {...}, 'N_def': {...}} if it has one set per condition
                     'n_calibrated'          : number of parameters it calibrated (for AIC/BIC, K = m + 1)
                     'system_data'           : (optional) its own System_data; default = the main system_data
    Two independent rankings are made: dataset='cal' (Cal_NRMSE) and dataset='val' (NRMSE).
    Returns {'cal': table, 'val': table}.
    """
    _ensure_dir(outdir)
    sims = {'normal': simulate_model, 'N_def': simulate_model}
    var_exp = system_data['var_exp_names']

    merged = load_runs(csv_path_normal, csv_path_N_def)
    orig_rows = merged[_original_mask(merged)]
    orig_row = orig_rows.iloc[-1] if len(orig_rows) else None
    if orig_row is None:
        print(">>> Best models: original model row not found in the CSVs; it will not be shown as reference.")

    tables = {}
    for dataset in ('cal', 'val'):
        pre = 'Cal_' if dataset == 'cal' else ''
        sel, score, n_total = select_best(merged, dataset, rank_on=rank_on, top_frac=top_frac)
        if sel.empty:
            print(f">>> Best models ({dataset}): no finite NRMSE found, skipping.")
            continue
        labels = _model_labels(sel.index)
        print(f">>> FINAL comparison | {DATASET_LABEL[dataset]}: top {len(sel)} of {n_total} runs (rank_on='{rank_on}')", flush=True)

        params_list = [_rebuild_parameters(sel.iloc[i], system_data) for i in range(len(sel))]
        sols = _simulate_set(system_data, sims, params_list, dataset)
        orig = _simulate_set(system_data, sims, [dict(system_data['parameters'])], dataset)

        refs = [('Original', _row_metrics(orig_row, pre, var_exp), dict(color='white', edgecolor='black', hatch='//'))]
        base = None
        if baseline is not None:
            blabel = baseline.get('label', 'Baseline')
            base = {'label': blabel, 'sols': _baseline_sims(baseline, system_data, dataset)}
            if all(v is None for v in base['sols'].values()):
                print(f">>> {blabel}: no stored trajectory for the {DATASET_LABEL[dataset].lower()} experiment "
                      f"-> bars only, no line in the solution figures", flush=True)
            refs.append((blabel, _baseline_metrics(baseline, system_data, dataset),
                         dict(color=BASE_COLOR, edgecolor='black')))

        plot_best_solutions(system_data, sols, orig, labels, dataset, outdir, n_total, rank_on, base)
        plot_best_solutions_band(system_data, sols, orig, labels, dataset, outdir, n_total, rank_on, base)
        plot_best_metrics(system_data, sel, refs, labels, dataset, outdir, n_total, rank_on)

        tab = _summary_table(system_data, sel, score, labels)
        tab.to_csv(os.path.join(outdir, f"Best_{DATASET_LABEL[dataset]}_models.csv"), index=False)
        _reference_table(refs).to_csv(os.path.join(outdir, f"Best_{DATASET_LABEL[dataset]}_reference_metrics.csv"), index=False)
        tables[dataset] = tab

    return tables