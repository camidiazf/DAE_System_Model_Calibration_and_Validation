import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DAE_analysis_package.Results_store import ResultsStore, CONDITIONS, DATASETS, DATASET_LABEL, condition_label
from DAE_analysis_package.Plotting_functions import save_fig_file, experiment_data, dataset_xmax, _shared_legend
# baseline helpers (a model of a different structure, e.g. Monod): simulate it, or read stored trajectories / metrics
from DAE_analysis_package.Best_models_functions import _baseline_sims, _baseline_metrics

"""
Model comparison from a results campaign (replaces RUN_BEST_MODELS for new campaigns).

Differences with Best_models_functions.RUN_BEST_MODELS
  * reads the tidy tables and the STORED trajectories of the campaign (no CSV joining, no re-simulation);
  * top_frac = 1.0 by default: ALL finished runs are plotted (set 0.1 for the top 10 %);
  * ONE ranking, by default the global NRMSE on the CALIBRATION data of the fitted condition (N_def). The same run keeps the
    same label, colour and order in the calibration and the validation figures. (Ranking validation by validation NRMSE
    lets the validation data choose the model.)
  * runs that failed or have a non-finite score are listed in a table instead of silently disappearing;
  * the Original model (and an optional Baseline) are drawn / tabulated on the calibration AND the validation datasets.

Outputs in <campaign>/comparison/ (prefix 'All' or 'Top<N>pct'):
    <prefix>_Cal_Solutions.png, <prefix>_Val_Solutions.png          one line per run (rows = states, columns = replete | deficient)
    <prefix>_Cal_Solutions_Band.png, <prefix>_Val_Solutions_Band.png  min-max band + rank-1 line
    <prefix>_Cal_Metrics.png, <prefix>_Val_Metrics.png               global NRMSE, NRMSE per variable, AIC / BIC bars
    <prefix>_models_ranking.csv      rank, label, run_id, parameters, cost, NRMSE/AIC/BIC (cal & val, both conditions), identifiability
    <prefix>_reference_metrics.csv   Original (and Baseline) metrics, same columns as the bars
    excluded_runs.csv                failed runs / runs without a finite ranking score
"""

BAND_COLOR = "#467dd6"
LINE_COLOR = "#00348d"
BASE_COLOR = "#f37859"


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _short(run_id: str) -> str:
    return run_id.replace('__r', ' r')


def _model_colors(n: int) -> list:
    """Distinct colours for <= 10 runs; grey ramp (dark = best rank) otherwise."""
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    return [plt.cm.Greys(v) for v in np.linspace(0.9, 0.35, n)]


def _run_metric_table(met: pd.DataFrame, run_ids: List[str], dataset: str, var_exp: List[str]
                      ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{run_id: {condition: {'NRMSE','AIC','BIC','NRMSE_<var>'}}} for ONE dataset, from the metrics table."""
    out = {rid: {c: dict({'NRMSE': np.nan, 'AIC': np.nan, 'BIC': np.nan}, **{'NRMSE_' + v: np.nan for v in var_exp})
                 for c in CONDITIONS} for rid in run_ids}
    m = met[met['dataset'] == dataset]
    for rid, c, var, nrmse, aic, bic in zip(m['run_id'], m['condition'], m['variable'], m['NRMSE'], m['AIC'], m['BIC']):
        if rid not in out or c not in out[rid]:
            continue
        d = out[rid][c]
        if var == 'global':
            d['NRMSE'], d['AIC'], d['BIC'] = nrmse, aic, bic
        else:
            d['NRMSE_' + var] = nrmse
    return out


def rank_runs(store: ResultsStore, rank_on: str = 'N_def', rank_dataset: str = 'cal', top_frac: float = 1.0
              ) -> Dict[str, Any]:
    """
    Finished calibrated runs ordered by the global NRMSE of `rank_dataset` ('cal' or 'val').
        rank_on : 'N_def' | 'normal' (that condition's NRMSE) | 'both' (mean of the two)
    Returns {'selected': [run_id...], 'score': Series(all finite scores, sorted), 'n_total', 'excluded': DataFrame}.
    """
    if rank_on not in ('both', 'normal', 'N_def'):
        raise ValueError(f"rank_on must be 'both', 'normal' or 'N_def' (got {rank_on!r})")
    if rank_dataset not in DATASETS:
        raise ValueError(f"rank_dataset must be 'cal' or 'val' (got {rank_dataset!r})")
    runs = store.read('runs')
    runs = runs[runs['combo_id'] != 'original']
    done = runs[runs['status'] == 'done']
    met = store.read('metrics')
    g = met[(met['variable'] == 'global') & (met['dataset'] == rank_dataset) & (met['run_id'].isin(done['run_id']))]
    piv = g.pivot_table(index='run_id', columns='condition', values='NRMSE', aggfunc='first')
    for c in CONDITIONS:
        if c not in piv.columns:
            piv[c] = np.nan
    score = piv.mean(axis=1, skipna=False) if rank_on == 'both' else piv[rank_on]
    score = score.reindex(done['run_id'])
    finite = score[np.isfinite(score.astype(float))].astype(float).sort_values(kind='stable')

    excl = runs[runs['status'] != 'done'][['run_id', 'combo_id', 'iteration', 'status', 'error']].copy()
    nonfinite = [r for r in done['run_id'] if r not in finite.index]
    if nonfinite:
        extra = done[done['run_id'].isin(nonfinite)][['run_id', 'combo_id', 'iteration', 'status', 'error']].copy()
        extra['error'] = 'no finite ranking score'
        excl = pd.concat([excl, extra], ignore_index=True) if len(excl) else extra

    n_total = len(finite)
    n_top = max(1, int(np.ceil(top_frac * n_total))) if n_total else 0
    return {'selected': finite.index[:n_top].tolist(), 'score': finite, 'n_total': n_total, 'excluded': excl}


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


def _decorate(ax, i, j, n_rows, var, cond_label, xmax) -> None:
    ax.set_xlim(0, xmax)
    ax.grid(True, alpha=0.4)
    if j == 0:
        ax.set_ylabel(var, fontsize=12)
    if i == 0:
        ax.set_title(cond_label, fontsize=14)
    if i == n_rows - 1:
        ax.set_xlabel('Time (h)', fontsize=12)


def plot_solutions(system_data: Dict[str, Any], sols: Dict[str, list], orig: Dict[str, Any], labels: List[str],
                   dataset: str, title: str, outdir: str, filename: str, base: Optional[Dict[str, Any]] = None) -> None:
    """Rows = states, columns = replete | deficient. One line per run (rank 1 drawn last, on top)."""
    var_names = list(system_data['var_names'])
    var_exp = list(system_data['var_exp_names'])
    t = np.asarray(system_data['time_stamps_sim'], dtype=float)
    n = len(labels)
    colors = _model_colors(n)

    fig, axes = plt.subplots(len(var_names), 2, figsize=(14, 2.6 * len(var_names) + 1.0), sharex='col', squeeze=False)
    for j, cond in enumerate(CONDITIONS):
        t_exp, df_exp = experiment_data(system_data, cond, dataset)
        xmax = dataset_xmax(system_data, cond, dataset)
        m = t <= xmax
        me = t_exp <= xmax
        for i, var in enumerate(var_names):
            ax = axes[i, j]
            for k in range(n - 1, -1, -1):
                sol = sols[cond][k]
                if sol is None or var not in sol:
                    continue
                if n <= 10:
                    lab = labels[k]
                else:
                    lab = f'{labels[0]} (rank 1)' if k == 0 else (f'All {n} models (darker = better)' if k == 1 else '_nolegend_')
                ax.plot(t[m], np.asarray(sol[var])[m], '-', color=colors[k], lw=2.2 if k == 0 else 1.0,
                        alpha=1.0 if k == 0 else 0.8, label=lab, zorder=3 if k == 0 else 2)
            o = orig.get(cond)
            if o is not None and var in o:
                ax.plot(t[m], np.asarray(o[var])[m], 'k--', lw=1.5, label='Original model', zorder=4)
            _add_baseline_line(ax, t, m, base, cond, var)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], c='k', marker='o', s=22, label='Experimental data', zorder=5)
            _decorate(ax, i, j, len(var_names), var, condition_label(system_data, cond), xmax)
    _shared_legend(fig, axes, ncol=6, y=0.0)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    save_fig_file(fig, outdir, filename)


def plot_solutions_band(system_data: Dict[str, Any], sols: Dict[str, list], orig: Dict[str, Any], labels: List[str],
                        dataset: str, title: str, outdir: str, filename: str, base: Optional[Dict[str, Any]] = None) -> None:
    """
    Same layout, runs shown as a shaded band (pointwise min-max envelope) with the rank-1 run as central line.
    NOTE: the band is the range spanned by the runs, NOT a statistical confidence interval.
    """
    var_names = list(system_data['var_names'])
    var_exp = list(system_data['var_exp_names'])
    t = np.asarray(system_data['time_stamps_sim'], dtype=float)
    n = len(labels)

    fig, axes = plt.subplots(len(var_names), 2, figsize=(14, 2.6 * len(var_names) + 1.0), sharex='col', squeeze=False)
    for j, cond in enumerate(CONDITIONS):
        t_exp, df_exp = experiment_data(system_data, cond, dataset)
        xmax = dataset_xmax(system_data, cond, dataset)
        m = t <= xmax
        me = t_exp <= xmax
        valid = [s for s in sols[cond] if s is not None]
        for i, var in enumerate(var_names):
            ax = axes[i, j]
            arrs = [np.asarray(s[var])[m] for s in valid if var in s]
            if n > 1 and arrs:
                arr = np.vstack(arrs)
                ax.fill_between(t[m], arr.min(axis=0), arr.max(axis=0), color=BAND_COLOR, alpha=0.25, lw=0,
                                label=f'Range of {n} models', zorder=1)
            best = sols[cond][0]
            if best is not None and var in best:
                ax.plot(t[m], np.asarray(best[var])[m], '-', color=LINE_COLOR, lw=2.2, label=f'Best model ({labels[0]})', zorder=3)
            o = orig.get(cond)
            if o is not None and var in o:
                ax.plot(t[m], np.asarray(o[var])[m], 'k--', lw=1.5, label='Original model', zorder=4)
            _add_baseline_line(ax, t, m, base, cond, var)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], facecolors='none', edgecolors='#003b9c', marker='o',
                           s=26, label='Experimental data', zorder=5)
            _decorate(ax, i, j, len(var_names), var, condition_label(system_data, cond), xmax)
    _shared_legend(fig, axes, ncol=6, y=0.0)
    fig.suptitle(title + "; shaded = min-max range", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    save_fig_file(fig, outdir, filename)


def plot_metrics(system_data: Dict[str, Any], run_metrics: List[Dict[str, Dict[str, float]]], refs: List[tuple],
                 labels: List[str], dataset: str, title: str, outdir: str, filename: str) -> None:
    """
    Bars, columns = replete | deficient; rows = (a) global NRMSE, (b) NRMSE per measured variable, (c) AIC and BIC.
    run_metrics[k][cond] = {'NRMSE','AIC','BIC','NRMSE_<var>'} of run k on `dataset`;
    refs = [(name, {cond: metrics}, bar_style), ...] drawn first (Original, Baseline).
    """
    var_exp = list(system_data['var_exp_names'])
    n = len(labels)
    r = len(refs)
    nb = r + n
    colors = _model_colors(n)
    xs = np.arange(nb)
    xt = [name for name, _, _ in refs] + labels
    rot = 90 if n > 6 else 45

    def val(k, cond, key):
        d = refs[k][1][cond] if k < r else run_metrics[k - r][cond]
        v = d.get(key, np.nan)
        return float(v) if v is not None else np.nan

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), squeeze=False, sharey='row')
    for j, cond in enumerate(CONDITIONS):
        ax = axes[0, j]
        for k in range(r):
            ax.bar(xs[k], val(k, cond, 'NRMSE'), **refs[k][2])
        ax.bar(xs[r:], [val(r + k, cond, 'NRMSE') for k in range(n)], color=colors, edgecolor='black')
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=rot, ha='right', fontsize=8)
        ax.set_title(f"{condition_label(system_data, cond)}: global NRMSE", fontsize=13)
        if j == 0:
            ax.set_ylabel('NRMSE (-)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)

        ax = axes[1, j]
        mv = len(var_exp)
        width = 0.8 / max(nb, 1)
        for k in range(nb):
            off = (k - (nb - 1) / 2) * width
            vals = [val(k, cond, 'NRMSE_' + v) for v in var_exp]
            if k < r:
                kw = dict(refs[k][2], label=refs[k][0])
            else:
                kw = dict(color=colors[k - r], edgecolor='black', label=labels[k - r])
            ax.bar(np.arange(mv) + off, vals, width, **kw)
        ax.set_xticks(np.arange(mv))
        ax.set_xticklabels(var_exp, fontsize=11)
        ax.set_title(f"{condition_label(system_data, cond)}: NRMSE per variable", fontsize=13)
        if j == 0:
            ax.set_ylabel('NRMSE (-)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)
        if nb <= 11 and j == 1:
            ax.legend(fontsize=8, ncol=2)

        ax = axes[2, j]
        aic = [val(k, cond, 'AIC') for k in range(nb)]
        bic = [val(k, cond, 'BIC') for k in range(nb)]
        w = 0.4
        ax.bar(xs - w / 2, aic, w, color='lightgray', edgecolor='black', label='AIC')
        ax.bar(xs + w / 2, bic, w, color='dimgray', edgecolor='black', label='BIC')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=rot, ha='right', fontsize=8)
        ax.set_title(f"{condition_label(system_data, cond)}: AIC / BIC", fontsize=13)
        if j == 0:
            ax.set_ylabel('AIC / BIC', fontsize=12)
        ax.grid(True, axis='y', alpha=0.5)
        if j == 1:
            ax.legend(fontsize=9, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)   # above the axes: bars hang from 0
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig_file(fig, outdir, filename)


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------
def RUN_MODEL_COMPARISON(system_data: Dict[str, Any],
                         store: ResultsStore,
                         rank_on: str = 'N_def',
                         rank_dataset: str = 'cal',
                         top_frac: float = 1.0,
                         baseline: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Compare the finished runs of a campaign (can be called on its own after / during the calibration).
        top_frac     : 1.0 = ALL runs (default); 0.1 = best 10 % (at least 1).
        rank_on      : 'N_def' (the condition the PSO fits) | 'normal' | 'both' (mean).
        rank_dataset : 'cal' (default; selection never sees validation data) | 'val'.
        baseline     : optional model of another structure, same dict as RUN_BEST_MODELS
                       ('label', 'simulate_model', 'parameters', 'n_calibrated', 'system_data') or the pre-computed form
                       made by load_baseline_xlsx ('solutions' / 'metrics').
    Returns {'ranking': DataFrame, 'reference': DataFrame, 'excluded': DataFrame}.
    """
    outdir = store.comparison_path()
    var_exp = list(system_data['var_exp_names'])
    t_grid = system_data['time_stamps_sim']

    rk = rank_runs(store, rank_on=rank_on, rank_dataset=rank_dataset, top_frac=top_frac)
    rk['excluded'].to_csv(os.path.join(outdir, 'excluded_runs.csv'), index=False)
    if not rk['selected']:
        print(">>> Comparison: no finished run with a finite NRMSE; nothing to plot (see comparison/excluded_runs.csv).")
        return {'ranking': pd.DataFrame(), 'reference': pd.DataFrame(), 'excluded': rk['excluded']}

    selected = rk['selected']
    n, n_total = len(selected), rk['n_total']
    prefix = 'All' if top_frac >= 1.0 else f"Top{int(round(top_frac * 100))}pct"
    labels = [f"#{k + 1} {_short(rid)}" for k, rid in enumerate(selected)]
    what = f"All {n}" if top_frac >= 1.0 else f"Top {n} of {n_total}"
    rank_txt = f"ranked by {DATASET_LABEL[rank_dataset].lower()} global NRMSE ({rank_on})"
    print(f">>> Comparison: {what} runs, {rank_txt}", flush=True)

    met = store.read('metrics')
    all_ids = ['original'] + selected
    sols_by_ds, orig_by_ds = {}, {}
    for ds in DATASETS:
        sols_by_ds[ds] = {c: [store.load_solution(rid, c, ds, t_grid=t_grid) for rid in selected] for c in CONDITIONS}
        orig_by_ds[ds] = {c: store.load_solution('original', c, ds, t_grid=t_grid) for c in CONDITIONS}
        missing = [rid for rid, s in zip(selected, sols_by_ds[ds]['N_def']) if s is None]
        if missing:
            print(f">>> Comparison: no stored {ds} trajectory for {len(missing)} run(s), e.g. {missing[:3]}", flush=True)

    ref_rows = []
    for ds in DATASETS:
        rm = _run_metric_table(met, all_ids, ds, var_exp)
        refs = [('Original', rm['original'], dict(color='white', edgecolor='black', hatch='//'))]
        base = None
        if baseline is not None:
            blabel = baseline.get('label', 'Baseline')
            base = {'label': blabel, 'sols': _baseline_sims(baseline, system_data, ds)}
            if all(v is None for v in base['sols'].values()):
                print(f">>> {blabel}: no trajectory for the {DATASET_LABEL[ds].lower()} experiment -> bars only, no line", flush=True)
            refs.append((blabel, _baseline_metrics(baseline, system_data, ds), dict(color=BASE_COLOR, edgecolor='black')))
        for name, mdict, _ in refs:
            for c in CONDITIONS:
                ref_rows.append(dict(model=name, dataset=ds, condition=c, **mdict[c]))

        ttl = (f"{what} models, {rank_txt}; data shown: {DATASET_LABEL[ds].lower()} experiment")
        dl = DATASET_LABEL[ds]
        plot_solutions(system_data, sols_by_ds[ds], orig_by_ds[ds], labels, ds, ttl, outdir, f"{prefix}_{dl}_Solutions.png", base)
        plot_solutions_band(system_data, sols_by_ds[ds], orig_by_ds[ds], labels, ds, ttl, outdir,
                            f"{prefix}_{dl}_Solutions_Band.png", base)
        plot_metrics(system_data, [rm[rid] for rid in selected], refs, labels, ds,
                     f"Metrics of {what} models on the {DATASET_LABEL[ds].lower()} data ({rank_txt})", outdir,
                     f"{prefix}_{dl}_Metrics.png")

    # ranking table = rank info + the derived wide summary of the selected runs
    wide = store.build_summary_wide()
    rank_df = pd.DataFrame({'rank': np.arange(1, n + 1), 'label': labels, 'run_id': selected,
                            'rank_score_NRMSE': rk['score'].loc[selected].values})
    if not wide.empty:
        rank_df = rank_df.merge(wide, on='run_id', how='left')
    rank_df.to_csv(os.path.join(outdir, f"{prefix}_models_ranking.csv"), index=False)
    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(os.path.join(outdir, f"{prefix}_reference_metrics.csv"), index=False)
    return {'ranking': rank_df, 'reference': ref_df, 'excluded': rk['excluded']}
