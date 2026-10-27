"""
Campaign_compare.py
===================
Join the results of SEVERAL campaigns (several RUN_COMBOS calls, possibly different models) into one table, keeping the
trace back to the campaign, the run and its figures.

Needs only pandas / numpy / matplotlib (it reads the CSV / JSON files of the campaign folders; the rest of the package is
not imported). It reads, for every campaign:
    campaign.json, tables/summary_wide.csv, tables/identifiability.csv, tables/runs.csv

Keys
    campaign : short label of the campaign (folder name; prefixed with the model if two campaigns share a name)
    run_id   : id inside its campaign, e.g. 'k3_ab12cd__r02'. NOT unique across campaigns (same parameters + bounds -> same
               combo id), so the global key is
    uid      : '<campaign>::<run_id>'

Typical use (Jupyter):
    from DAE_analysis_package.Campaign_compare import *
    camps = load_campaigns('M_smegmatis_extended_modified')          # a root folder (searched) or a list of campaign folders
    camps['settings']                                                # PSO / cost / condition per campaign -> are they comparable?
    summary, ident = camps['summary'], camps['identifiability']
    table = summary_with_identifiability(summary, ident)             # NRMSE + identifiability, one row per run
    table.sort_values('N_def_cal_NRMSE').head(10)
    trace(table, uid)                                                # where is this run on disk?
    show_run(table, uid, 'N_def')                                    # its single-solution figure

One call for the top-N figures (global / per-variable NRMSE per condition x dataset, t-values) saved in a folder:
    out = RUN_TOP_COMPARISON('M_smegmatis_extended_modified', outdir='Compare_2026-10-25', n=10)
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# (condition, dataset) -> summary_wide column with the global NRMSE
NRMSE_COLS = ['N_def_cal_NRMSE', 'N_def_val_NRMSE', 'normal_cal_NRMSE', 'normal_val_NRMSE']
T_SIGNIFICANT = 2.0            # t > 2 -> significant (same threshold as the analysis code)


# ----------------------------------------------------------------------------------------------
# discovery and loading
# ----------------------------------------------------------------------------------------------
def find_campaigns(root: str = '.') -> List[str]:
    """Absolute paths of all campaign folders (those containing campaign.json) under `root`."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'campaign.json' in filenames:
            found.append(os.path.abspath(dirpath))
            dirnames[:] = []                       # do not look inside a campaign
    return sorted(found)


def _read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _campaign_info(path: str) -> dict:
    cfg = _read_json(os.path.join(path, 'campaign.json'))
    pso = cfg.get('pso') or {}
    env = cfg.get('environment') or {}
    # <folder_model>/<folder_model>_Results/<campaign>
    model = cfg.get('folder_model') or os.path.basename(os.path.dirname(os.path.dirname(path)))
    return {'campaign_name': cfg.get('campaign') or os.path.basename(path), 'model': model, 'campaign_path': path,
            'created': cfg.get('created'), 'calibration_condition': cfg.get('calibration_condition'),
            'cost_scaling': cfg.get('cost_scaling'), 'pso_epoch': pso.get('epoch'), 'pso_pop_size': pso.get('pop_size'),
            'iterations': cfg.get('iterations'), 'min_k': cfg.get('min_k'), 'max_k': cfg.get('max_k'),
            'pso_seed_base': cfg.get('pso_seed_base'), 'baseline': cfg.get('baseline'), 'git_hash': env.get('git_hash'),
            'condition_labels': cfg.get('condition_labels'),
            'n_sessions': len(cfg.get('sessions') or []), 'config_changes': bool(cfg.get('config_changes'))}


def _make_labels(infos: List[dict], labels: Optional[Dict[str, str]]) -> List[str]:
    """Campaign labels: user-given > campaign name > model/campaign > numbered (always unique)."""
    labels = labels or {}
    out = []
    for inf in infos:
        given = labels.get(inf['campaign_path']) or labels.get(inf['campaign_name'])
        out.append(given or inf['campaign_name'])
    if len(set(out)) < len(out):
        out = [l if labels.get(i['campaign_path']) or labels.get(i['campaign_name']) else f"{i['model']}/{l}"
               for l, i in zip(out, infos)]
    if len(set(out)) < len(out):
        seen: Dict[str, int] = {}
        uniq = []
        for l in out:
            seen[l] = seen.get(l, 0) + 1
            uniq.append(l if out.count(l) == 1 else f"{l}#{seen[l]}")
        out = uniq
    return out


def _read_table(path: str) -> Optional[pd.DataFrame]:
    """The table, or None if the file does not exist or has no rows."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def load_campaigns(paths: Union[str, os.PathLike, Sequence[str]], labels: Optional[Dict[str, str]] = None
                   ) -> Dict[str, pd.DataFrame]:
    """
    paths  : a root folder (all campaigns below it are loaded) or a list of campaign folders.
    labels : optional {campaign folder path or campaign name: short label} for the 'campaign' column.
    Returns {'summary', 'identifiability', 'runs', 'settings'} DataFrames, each with 'campaign' and 'uid' (except settings).
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = find_campaigns(str(paths))
    paths = [os.path.abspath(str(p)) for p in paths]
    if not paths:
        raise ValueError("No campaign folder found (a campaign folder contains campaign.json).")
    infos = [_campaign_info(p) for p in paths]
    names = _make_labels(infos, labels)

    summ, ide, runs, sett = [], [], [], []
    for p, inf, name in zip(paths, infos, names):
        tdir = os.path.join(p, 'tables')
        s = _read_table(os.path.join(tdir, 'summary_wide.csv'))
        i = _read_table(os.path.join(tdir, 'identifiability.csv'))
        r = _read_table(os.path.join(tdir, 'runs.csv'))
        if s is None or r is None:
            print(f">>> Campaign '{name}' skipped: no finished runs in {tdir} (all runs failed, or it is still empty). "
                  f"If runs did finish, ResultsStore.open(<campaign folder>).refresh_exports() rebuilds summary_wide.csv.")
            continue
        if i is None:
            i = pd.DataFrame(columns=['run_id', 'condition', 'fitted_on', 'parameter', 'theta', 't_value', 'SE', 'CI_low',
                                      'CI_high', 'rel_error_pct', 'cond_FIM', 'rank_ratio', 'n_highly_corr'])
        for df in (s, i, r):
            df.insert(0, 'campaign', name)
            df.insert(1, 'uid', name + '::' + df['run_id'].astype(str))
        s['model_type'] = np.where(s['combo_id'] == 'original', 'original', 'calibrated')
        s['run_dir'] = [os.path.join(p, 'runs', rid) for rid in s['run_id']]
        # campaign-level settings next to every run (filter on them to compare like with like)
        for k in ('model', 'calibration_condition', 'cost_scaling', 'pso_epoch', 'pso_pop_size', 'created'):
            s[k] = inf[k]
        summ.append(s); ide.append(i); runs.append(r)
        sett.append(dict(campaign=name, **inf))
    if not summ:
        raise ValueError("None of the campaigns has finished runs to load.")

    return {'summary': pd.concat(summ, ignore_index=True, sort=False),
            'identifiability': pd.concat(ide, ignore_index=True, sort=False),
            'runs': pd.concat(runs, ignore_index=True, sort=False),
            'settings': pd.DataFrame(sett)}


def settings_differences(settings: pd.DataFrame,
                         cols: Sequence[str] = ('model', 'calibration_condition', 'cost_scaling', 'pso_epoch',
                                                'pso_pop_size', 'iterations', 'pso_seed_base', 'baseline', 'git_hash')
                         ) -> pd.DataFrame:
    """The settings that are NOT the same in all loaded campaigns (NRMSE/AIC are only comparable if the relevant ones are)."""
    cols = [c for c in cols if c in settings.columns]
    varying = [c for c in cols if settings[c].astype(str).nunique() > 1]
    return settings[['campaign'] + varying].set_index('campaign') if varying else pd.DataFrame(
        {'info': ['all loaded campaigns share these settings']}, index=['settings'])


# ----------------------------------------------------------------------------------------------
# identifiability
# ----------------------------------------------------------------------------------------------
def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == 'true'


def identifiability_wide(ident: pd.DataFrame, value: str = 't_value', condition: str = 'N_def',
                         fitted_only: bool = True) -> pd.DataFrame:
    """
    One row per run (index uid), one column per parameter: `value` = 't_value' | 'rel_error_pct' | 'SE' | 'theta' | ...
    fitted_only=True keeps the statistics of the condition the PSO was fitted on (the only ones that are estimation statistics).
    """
    d = ident[ident['condition'] == condition]
    if fitted_only:
        d = d[_is_true(d['fitted_on'])]
    return d.pivot_table(index='uid', columns='parameter', values=value, aggfunc='first')


def identifiability_summary(ident: pd.DataFrame) -> pd.DataFrame:
    """Per run (fitted condition only): n parameters, how many have t <= 2 (NaN counts as not significant), worst relative error."""
    fit = ident[_is_true(ident['fitted_on'])]
    if fit.empty:
        return pd.DataFrame(columns=['n_identified', 'n_t_below_2', 'max_rel_error_pct'])
    g = fit.groupby('uid')
    out = pd.DataFrame({'n_identified': g['parameter'].nunique(),
                        'n_t_below_2': g['t_value'].apply(lambda s: int(((s <= T_SIGNIFICANT) | s.isna()).sum())),
                        'max_rel_error_pct': g['rel_error_pct'].max()})
    return out


def summary_with_identifiability(summary: pd.DataFrame, ident: pd.DataFrame, include_original: bool = True) -> pd.DataFrame:
    """
    summary_wide of all campaigns + n_identified / n_t_below_2 / max_rel_error_pct (fitted condition). min_t, max_abs_kappa and
    cond_FIM are already in summary_wide. One row per run; `uid` is the key.
    """
    out = summary.merge(identifiability_summary(ident), left_on='uid', right_index=True, how='left')
    if not include_original:
        out = out[out['model_type'] != 'original']
    return out.reset_index(drop=True)


def best_per_campaign(table: pd.DataFrame, by: str = 'N_def_cal_NRMSE', n: int = 3) -> pd.DataFrame:
    """The n best calibrated runs (lowest `by`) of each campaign."""
    t = table[table['model_type'] == 'calibrated'].dropna(subset=[by])
    return t.sort_values(by).groupby('campaign', sort=False).head(n).sort_values(['campaign', by]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# tracing and viewing
# ----------------------------------------------------------------------------------------------
def trace(table: pd.DataFrame, uid: str) -> pd.Series:
    """Everything needed to find a run again: campaign, model, run_id, campaign folder, run folder, figure folders, parameters."""
    row = table.loc[table['uid'] == uid]
    if row.empty:
        raise KeyError(f"uid '{uid}' not found")
    r = row.iloc[0]
    info = {'uid': uid, 'campaign': r['campaign'], 'model': r.get('model'), 'run_id': r['run_id'],
            'combo_id': r.get('combo_id'), 'calibrated_parameters': r.get('calibrated_parameters'),
            'seed': r.get('seed'), 'run_dir': r['run_dir'],
            'figures_N_def': os.path.join(r['run_dir'], 'figures', 'N_def'),
            'figures_normal': os.path.join(r['run_dir'], 'figures', 'normal'),
            'campaign_path': os.path.dirname(os.path.dirname(r['run_dir']))}
    return pd.Series(info)


def show_run(table: pd.DataFrame, uid: str, condition: str = 'N_def', figure: str = 'solution.png', width: int = 1100):
    """Display a figure of a run in Jupyter (solution.png, residuals_cal.png, residuals_val.png, sensitivity.png, corr_matrix.png)."""
    from IPython.display import Image, display
    path = os.path.join(trace(table, uid)['run_dir'], 'figures', condition, figure)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"{uid}\n{path}")
    return display(Image(filename=path, width=width))


def plot_nrmse_vs_identifiability(table: pd.DataFrame, x: str = 'N_def_cal_NRMSE', y: str = 'min_t', logy: bool = True,
                                  annotate_best: int = 0, ax=None):
    """
    Scatter of fit quality (x) against identifiability (y; default smallest t-value among the fitted parameters), one colour
    per campaign. Dashed line: t = 2. annotate_best = n labels the n runs with the lowest x with their uid.
    """
    import matplotlib.pyplot as plt
    t = table[table['model_type'] == 'calibrated'].dropna(subset=[x, y])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for name, g in t.groupby('campaign', sort=False):
        ax.scatter(g[x], g[y], label=f"{name} (n={len(g)})", s=36, alpha=0.85, edgecolor='black', linewidth=0.5)
    if y in ('min_t',):
        ax.axhline(T_SIGNIFICANT, color='k', ls='--', lw=1)
    if logy and (t[y] > 0).all():
        ax.set_yscale('log')
    if annotate_best:
        for _, r in t.nsmallest(annotate_best, x).iterrows():
            ax.annotate(r['uid'], (r[x], r[y]), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    return ax


def save_combined(path: str, camps: Dict[str, pd.DataFrame], table: Optional[pd.DataFrame] = None) -> str:
    """One Excel workbook: settings, the joined table (NRMSE + identifiability), and the long identifiability table."""
    with pd.ExcelWriter(path) as xw:
        camps['settings'].to_excel(xw, sheet_name='settings', index=False)
        (table if table is not None else camps['summary']).to_excel(xw, sheet_name='runs_summary', index=False)
        camps['identifiability'].to_excel(xw, sheet_name='identifiability_long', index=False)
    return path


# ----------------------------------------------------------------------------------------------
# Top-N across campaigns: NRMSE (global + per variable, every condition x dataset) and t-values -> one output folder
# ----------------------------------------------------------------------------------------------
DEFAULT_CONDITION_LABELS = {'normal': 'Nutrient replete', 'N_def': 'Ammonium deficient'}
DATASET_TITLES = {'cal': 'Calibration', 'val': 'Validation'}
CONDITIONS = ('normal', 'N_def')
DATASETS = ('cal', 'val')


def select_top(table: pd.DataFrame, n: int = 10, rank_by: str = 'N_def_cal_NRMSE') -> pd.DataFrame:
    """
    The n best CALIBRATED runs over all loaded campaigns (lowest `rank_by`), with rank and a legend label.
    Default ranking: global NRMSE on the calibration data of the fitted condition, so validation data never chooses the models.
    """
    if rank_by not in table.columns:
        raise KeyError(f"'{rank_by}' is not a column of the table; available NRMSE columns: "
                       f"{[c for c in table.columns if c.endswith('NRMSE')]}")
    t = table[table['model_type'] == 'calibrated'].dropna(subset=[rank_by]).sort_values(rank_by, kind='stable').head(n).copy()
    if t.empty:
        raise ValueError(f"No calibrated run with a finite '{rank_by}'.")
    t.insert(0, 'rank', range(1, len(t) + 1))
    t['label'] = [f"#{r} {c} | {rid.replace('__r', ' r')}" for r, c, rid in zip(t['rank'], t['campaign'], t['run_id'])]
    return t.reset_index(drop=True)


def _variables(table: pd.DataFrame) -> List[str]:
    """Measured variables, read from the per-variable NRMSE columns of summary_wide."""
    import re
    pat = re.compile(r'^(?:normal|N_def)_(?:cal|val)_NRMSE_(.+)$')
    out: List[str] = []
    for c in table.columns:
        m = pat.match(c)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _original_reference(table: pd.DataFrame):
    """(row of the original model, note). The original model is the same in campaigns that use the same data."""
    o = table[table['model_type'] == 'original']
    if o.empty:
        return None, ''
    cols = [c for c in o.columns if c.endswith('NRMSE') or '_NRMSE_' in c]
    note = ''
    if len(o) > 1 and cols:
        spread = float((o[cols].max() - o[cols].min()).max())
        if np.isfinite(spread) and spread > 1e-6:
            note = (f"the original model differs between campaigns (max spread of NRMSE {spread:.3g}); "
                    f"the first campaign's original is drawn")
    return o.iloc[0], note


def _colors(n: int) -> list:
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [cmap(i % cmap.N) for i in range(n)]


def _legend_handles(top: pd.DataFrame, colors: list, with_original: bool) -> list:
    import matplotlib.patches as mp
    h = []
    if with_original:
        h.append(mp.Patch(facecolor='white', edgecolor='black', hatch='//', label='Original model'))
    h += [mp.Patch(facecolor=c, edgecolor='black', label=lab) for c, lab in zip(colors, top['label'])]
    return h


def _panel_title(cond: str, ds: str, cl: Dict[str, str], fitted_condition: str) -> str:
    tag = ' [fitted]' if (cond == fitted_condition and ds == 'cal') else ''
    return f"{cl.get(cond, cond)}: {DATASET_TITLES[ds]}{tag}"


def _save(fig, outdir: str, filename: str) -> str:
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_top_nrmse_global(top: pd.DataFrame, orig_row, outdir: str, condition_labels: Optional[Dict[str, str]] = None,
                          fitted_condition: str = 'N_def', filename: str = 'NRMSE_global.png', title: str = '') -> str:
    """Global NRMSE of the selected runs. Rows = condition, columns = dataset (calibration | validation)."""
    import matplotlib.pyplot as plt
    cl = condition_labels or DEFAULT_CONDITION_LABELS
    n = len(top)
    colors = _colors(n)
    has_o = orig_row is not None
    x0 = 1 if has_o else 0
    ticks = (['Orig.'] if has_o else []) + [f"#{r}" for r in top['rank']]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, squeeze=False)
    for i, cond in enumerate(CONDITIONS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            col = f"{cond}_{ds}_NRMSE"
            vals = top[col].astype(float).values if col in top.columns else np.full(n, np.nan)
            if has_o:
                ax.bar(0, float(orig_row.get(col, np.nan)), color='white', edgecolor='black', hatch='//')
            ax.bar(np.arange(n) + x0, vals, color=colors, edgecolor='black')
            ax.set_xticks(np.arange(len(ticks)))
            ax.set_xticklabels(ticks, fontsize=8 if n <= 10 else 7, rotation=0 if n <= 10 else 90)
            ax.set_title(_panel_title(cond, ds, cl, fitted_condition), fontsize=12)
            ax.grid(True, axis='y', alpha=0.5)
            if j == 0:
                ax.set_ylabel('Global NRMSE (-)', fontsize=11)
    fig.legend(handles=_legend_handles(top, colors, has_o), loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8,
               frameon=False)
    fig.suptitle(title or f"Global NRMSE of the {n} best runs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, outdir, filename)


def plot_top_nrmse_per_variable(top: pd.DataFrame, orig_row, variables: Sequence[str], outdir: str,
                                condition_labels: Optional[Dict[str, str]] = None, fitted_condition: str = 'N_def',
                                filename: str = 'NRMSE_per_variable.png', title: str = '') -> str:
    """NRMSE per measured variable (groups) of the selected runs (bars). Rows = condition, columns = dataset."""
    import matplotlib.pyplot as plt
    cl = condition_labels or DEFAULT_CONDITION_LABELS
    n = len(top)
    colors = _colors(n)
    has_o = orig_row is not None
    nb = n + int(has_o)
    width = 0.8 / max(nb, 1)
    xs = np.arange(len(variables))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True, squeeze=False)
    for i, cond in enumerate(CONDITIONS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for k in range(nb):
                off = (k - (nb - 1) / 2) * width
                if has_o and k == 0:
                    vals = [float(orig_row.get(f"{cond}_{ds}_NRMSE_{v}", np.nan)) for v in variables]
                    ax.bar(xs + off, vals, width, color='white', edgecolor='black', hatch='//')
                else:
                    m = k - int(has_o)
                    vals = [float(top.iloc[m].get(f"{cond}_{ds}_NRMSE_{v}", np.nan)) for v in variables]
                    ax.bar(xs + off, vals, width, color=colors[m], edgecolor='black')
            ax.set_xticks(xs)
            ax.set_xticklabels(variables, fontsize=11)
            ax.set_title(_panel_title(cond, ds, cl, fitted_condition), fontsize=12)
            ax.grid(True, axis='y', alpha=0.5)
            if j == 0:
                ax.set_ylabel('NRMSE per variable (-)', fontsize=11)
    fig.legend(handles=_legend_handles(top, colors, has_o), loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8,
               frameon=False)
    fig.suptitle(title or f"NRMSE per variable of the {n} best runs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, outdir, filename)


def top_tvalues(ident: pd.DataFrame, top: pd.DataFrame, value: str = 't_value') -> pd.DataFrame:
    """Rows = selected runs (in rank order, index = uid), columns = estimated parameters (most frequently estimated first)."""
    d = ident[ident['uid'].isin(top['uid']) & _is_true(ident['fitted_on'])]
    M = d.pivot_table(index='uid', columns='parameter', values=value, aggfunc='first').reindex(top['uid'])
    order = M.notna().sum().sort_values(ascending=False, kind='stable').index
    return M[order]


def plot_top_tvalues(ident: pd.DataFrame, top: pd.DataFrame, outdir: str, filename: str = 'tvalues_heatmap.png',
                     title: str = '') -> str:
    """
    Heatmap of the t-values (|theta / SE|, FIM) of the parameters each selected run ESTIMATED (rows = runs, columns = parameters).
    Red numbers: t <= 2 (not significant). Grey cells: parameter not estimated in that run. Colour: log scale.
    Only the condition the PSO was fitted on is used (the other condition's statistics are not estimation statistics).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    M = top_tvalues(ident, top)
    A = M.values.astype(float)
    if A.size == 0:
        raise ValueError("No t-values for the selected runs (identifiability table empty?).")
    vmax = float(np.nanmax(A)) if np.isfinite(A).any() else 10.0
    norm = LogNorm(vmin=1.0, vmax=max(vmax, 10.0))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad('#e6e6e6')
    fig, ax = plt.subplots(figsize=(max(7.0, 0.95 * A.shape[1] + 5.5), max(4.0, 0.55 * A.shape[0] + 2.5)))
    im = ax.imshow(np.ma.masked_invalid(np.clip(A, 1.0, None)), cmap=cmap, norm=norm, aspect='auto')
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            v = A[r, c]
            if not np.isfinite(v):
                continue
            red = v <= T_SIGNIFICANT
            colr = 'red' if red else ('white' if float(norm(max(v, 1.0))) > 0.6 else 'black')
            ax.text(c, r, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha='center', va='center', fontsize=9, color=colr,
                    fontweight='bold' if red else 'normal')
    ax.set_xticks(range(A.shape[1]))
    ax.set_xticklabels(M.columns, rotation=45, ha='right')
    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels(top['label'], fontsize=8)
    fig.colorbar(im, ax=ax, label='t-value (log scale)', pad=0.02)
    ax.set_title(title or f"t-values of the estimated parameters, {len(top)} best runs\n"
                          f"red = t <= {T_SIGNIFICANT:g} (not significant); grey = not estimated in that run (or t not computable)",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, outdir, filename)


def RUN_TOP_COMPARISON(paths: Union[str, os.PathLike, Sequence[str]], outdir: str, n: int = 10,
                       rank_by: str = 'N_def_cal_NRMSE', labels: Optional[Dict[str, str]] = None,
                       condition_labels: Optional[Dict[str, str]] = None, fitted_condition: Optional[str] = None,
                       overwrite: bool = False) -> Dict[str, object]:
    """
    One call: load the campaigns, take the n best calibrated runs over ALL of them, and write into `outdir`
        Top<n>_NRMSE_global.png        global NRMSE, rows = condition, columns = calibration | validation
        Top<n>_NRMSE_per_variable.png  NRMSE per measured variable, same layout
        Top<n>_tvalues_heatmap.png     t-values of the estimated parameters (fitted condition)
        Top<n>_models.csv              the selected runs with campaign, run_id, parameters, NRMSE, identifiability, run_dir
        Top<n>_tvalues.csv             t-values behind the heatmap (rows = uid)
        campaigns_used.csv, info.json  which campaigns / settings / ranking produced this folder (traceability)
    Run it again with another `outdir` (and other campaign folders) for another comparison; an existing non-empty folder is
    refused unless overwrite=True, so earlier results are never overwritten by accident.
        paths : root folder to search, or list of campaign folders (see load_campaigns).
    """
    import datetime
    if os.path.isdir(outdir) and os.listdir(outdir) and not overwrite:
        raise FileExistsError(f"Output folder '{outdir}' is not empty. Use a new folder name, or overwrite=True.")
    os.makedirs(outdir, exist_ok=True)

    camps = load_campaigns(paths, labels)
    table = summary_with_identifiability(camps['summary'], camps['identifiability'])
    top = select_top(table, n, rank_by)
    n_eff = len(top)
    if n_eff < n:
        print(f">>> Only {n_eff} calibrated runs with a finite '{rank_by}' are available (asked for {n}).")

    settings = camps['settings']
    diff = settings_differences(settings)
    if 'info' not in diff.columns:
        print(">>> WARNING: the campaigns differ in settings that can make NRMSE/AIC not comparable "
              "(data, cost scaling, calibrated condition); check before ranking them together:")
        print(diff.to_string())
    if condition_labels is None:
        found = [v for v in settings.get('condition_labels', []) if isinstance(v, dict)]
        condition_labels = dict(DEFAULT_CONDITION_LABELS, **found[0]) if found else DEFAULT_CONDITION_LABELS
    if fitted_condition is None:
        fc = settings['calibration_condition'].dropna()
        fitted_condition = str(fc.iloc[0]) if len(fc) else 'N_def'
        if fc.astype(str).nunique() > 1:
            print(">>> WARNING: the campaigns were fitted on different conditions: "
                  f"{fc.astype(str).unique().tolist()}; the t-values / [fitted] tags assume '{fitted_condition}'.")

    orig_row, note = _original_reference(table)
    if note:
        print(">>> NOTE:", note)
    variables = _variables(table)
    tag = f"Top{n_eff}"
    sub = f"ranked by {rank_by} over {len(settings)} campaign(s)"
    files = [
        plot_top_nrmse_global(top, orig_row, outdir, condition_labels, fitted_condition, f"{tag}_NRMSE_global.png",
                              title=f"Global NRMSE of the {n_eff} best runs ({sub})"),
        plot_top_nrmse_per_variable(top, orig_row, variables, outdir, condition_labels, fitted_condition,
                                    f"{tag}_NRMSE_per_variable.png",
                                    title=f"NRMSE per variable of the {n_eff} best runs ({sub})"),
        plot_top_tvalues(camps['identifiability'], top, outdir, f"{tag}_tvalues_heatmap.png"),
    ]
    tv = top_tvalues(camps['identifiability'], top)
    tv.to_csv(os.path.join(outdir, f"{tag}_tvalues.csv"))
    keep = (['rank', 'label', 'uid', 'campaign', 'run_id', 'model', 'calibrated_parameters', 'seed', 'cost', 'bound_hits']
            + [c for c in top.columns if c.endswith('NRMSE') or '_NRMSE_' in c or c.endswith('_AIC') or c.endswith('_BIC')]
            + ['min_t', 'n_t_below_2', 'max_rel_error_pct', 'max_abs_kappa', 'cond_FIM', 'run_dir'])
    top[[c for c in keep if c in top.columns]].to_csv(os.path.join(outdir, f"{tag}_models.csv"), index=False)
    settings.drop(columns=['condition_labels'], errors='ignore').to_csv(os.path.join(outdir, 'campaigns_used.csv'), index=False)
    with open(os.path.join(outdir, 'info.json'), 'w', encoding='utf-8') as fh:
        json.dump({'created': datetime.datetime.now().isoformat(timespec='seconds'), 'n_requested': n, 'n_selected': n_eff,
                   'rank_by': rank_by, 'fitted_condition': fitted_condition, 'condition_labels': condition_labels,
                   'campaigns': settings[['campaign', 'campaign_path']].to_dict('records'),
                   'selected_uids': top['uid'].tolist(), 'note': note}, fh, indent=2)
    print(f">>> Saved {len(files)} figures + tables in {os.path.abspath(outdir)}")
    return {'camps': camps, 'table': table, 'top': top, 'tvalues': tv, 'files': files}