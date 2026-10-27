import os
import pandas as pd
from typing import Dict, Optional, List, Sequence, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

from DAE_analysis_package.Results_store import CONDITIONS, DATASETS, DATASET_LABEL, dataset_label, condition_label

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

"""
Module containing plotting and saving functions for DAE system model calibration and validation.

[STORE] Figures are no longer named by 'iteration+1' (which silently overwrote when the same name was reused). The caller
(RUN_functions) gives a folder (runs/<run_id>/figures/<condition>/) and every function takes a `filename`/`title`.
plotting_single_solution now draws CALIBRATION | VALIDATION side by side (rows = states).
"""

CAL_COLOR = "#00348d"       # calibrated model
ORIG_COLOR = "black"        # original model


# ----------------------------------------------------------------------------------------------
# Saving helpers
# ----------------------------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    """Ensure the directory exists."""
    os.makedirs(path, exist_ok=True)


def save_fig_file(fig: plt.Figure, fig_outdir: str, filename: str, dpi: int = 200) -> str:
    """Save a figure as <fig_outdir>/<filename> and close it (Jupyter does not show it twice)."""
    _ensure_dir(fig_outdir)
    path = os.path.join(fig_outdir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --- legacy helpers (kept so old notebooks importing them do not break) ---
def _iter_index(iteration: Optional[int]) -> int:
    """Map None -> 0 (original), otherwise i -> i+1."""
    return 0 if iteration is None else int(iteration) + 1


def save_fig(fig: plt.Figure, fig_outdir: str, name: str, iteration: Optional[int] = None) -> None:
    """LEGACY naming (name_<iteration+1>.png). New code uses save_fig_file."""
    if iteration == 'original':
        fname = f"{name}_original.png"
    elif iteration == "summary":
        fname = f"{name}_summary.png"
    else:
        fname = f"{name}_{iteration+1}.png"
    save_fig_file(fig, fig_outdir, fname)


def figures_dir_from_csv_path(csv_path: str) -> str:
    """LEGACY: figures folder next to a CSV."""
    outdir = os.path.join(os.path.dirname(csv_path), "Figures")
    _ensure_dir(outdir)
    return outdir


def save_solution_data(solution: Dict[str, np.ndarray], time_stamps: np.ndarray, var_names: List[str],
                       outdir: str, name: str) -> None:
    """Save simulation solution data for external plotting."""
    _ensure_dir(outdir)
    solution_df = pd.DataFrame({"t": time_stamps})
    for var in var_names:
        solution_df[var] = np.asarray(solution[var])
    solution_df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)


# ----------------------------------------------------------------------------------------------
# Experiment helpers shared by the single-solution and comparison figures
# ----------------------------------------------------------------------------------------------
def experiment_data(system_data: Dict[str, Any], condition: str, dataset: str):
    """(t_exp, DataFrame) of the calibration ('cal') or validation ('val') experiment of a condition."""
    ext = '_' + condition
    if dataset == 'cal':
        return np.asarray(system_data['t_exp' + ext], dtype=float), system_data['df_exp' + ext]
    return np.asarray(system_data['t_exp_v' + ext], dtype=float), system_data['df_val' + ext]


def dataset_xmax(system_data: Dict[str, Any], condition: str, dataset: str) -> float:
    """
    Upper x limit (h) of a dataset's panels: system_data['xmax'] = {'cal': 50, 'val': 100} if given, otherwise the last
    experimental time rounded up to a multiple of 10 h (5 h below 40 h).
    """
    cfg = system_data.get('xmax')
    if isinstance(cfg, dict) and cfg.get(dataset):
        return float(cfg[dataset])
    t_exp, _ = experiment_data(system_data, condition, dataset)
    tmax = float(np.max(t_exp))
    step = 10.0 if tmax > 40 else 5.0
    return float(np.ceil(tmax / step) * step)


def _fmt(v, spec: str) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 'n/a'
    return format(v, spec) if np.isfinite(v) else 'n/a'


def _metric_line(m: Optional[Dict[str, Any]]) -> str:
    if not m:
        return ''
    return f"global NRMSE = {_fmt(m.get('NRMSE'), '.3f')} | AIC = {_fmt(m.get('AIC'), '.1f')} | BIC = {_fmt(m.get('BIC'), '.1f')}"


def _params_text(params: Dict[str, float], calibrated: Sequence[str]) -> str:
    def line(k):
        try:
            return f"{k:<10s}{float(params[k]):>11.4g}"
        except (TypeError, ValueError):
            return f"{k:<10s}{str(params[k]):>11s}"
    cal = [line(k) for k in params if k in calibrated]
    fix = [line(k) for k in params if k not in calibrated]
    out = []
    if cal:
        out += ["CALIBRATED"] + cal + [""]
    out += ["FIXED" if cal else "PARAMETERS"] + fix
    return "\n".join(out)


def _shared_legend(fig, axes, ncol: int = 6, y: float = 0.0) -> None:
    handles = {}
    ax_iter = axes if isinstance(axes, (list, tuple)) else np.ravel(axes)
    for ax in ax_iter:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    if handles:
        fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=min(len(handles), ncol),
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, y))


# ----------------------------------------------------------------------------------------------
# One run, one condition
# ----------------------------------------------------------------------------------------------
def plotting_single_solution(system_data: Dict[str, Any],
                             fig_outdir: str,
                             condition: str,
                             label: str,
                             parameters_updated: Dict[str, float],
                             sols: Dict[str, Any],
                             originals: Optional[Dict[str, Any]],
                             metrics: Optional[Dict[str, Dict[str, float]]],
                             original: bool = False,
                             calibration_condition: str = 'N_def',
                             filename: str = 'solution.png') -> Optional[str]:
    """
    One run, one condition: rows = states, columns = CALIBRATION | VALIDATION experiment.
        sols      : {'cal': DataFrame|None, 'val': DataFrame|None}  trajectories on time_stamps_sim (calibrated model)
        originals : same structure for the original (un-calibrated) model, or None
        metrics   : {'cal': {'NRMSE','AIC','BIC'}, 'val': {...}}  shown in the column titles
    The calibration column of the condition the PSO was fitted on is marked '[fitted]'; the other condition's calibration
    column is data the parameters were NOT fitted to ('[not fitted]').
    """
    var_names = list(system_data['var_names'])
    var_exp = list(system_data['var_exp_names'])
    t = np.asarray(system_data['time_stamps_sim'], dtype=float)
    calibrated = [] if original else list(system_data.get('calibrating_parameters') or [])
    n = len(var_names)

    fig = plt.figure(figsize=(17, 2.5 * n + 1.6))
    gs = gridspec.GridSpec(n, 3, width_ratios=[0.8, 2, 2], figure=fig, wspace=0.25, hspace=0.42,
                           left=0.02, right=0.99, top=0.88, bottom=0.10)
    ax_txt = fig.add_subplot(gs[:, 0])
    ax_txt.axis('off')
    ax_txt.text(0.0, 1.0, _params_text(parameters_updated, calibrated), ha='left', va='top', fontsize=8.5,
                family='monospace', transform=ax_txt.transAxes)

    axes = []
    for j, ds in enumerate(DATASETS):
        t_exp, df_exp = experiment_data(system_data, condition, ds)
        xmax = dataset_xmax(system_data, condition, ds)
        mk = t <= xmax
        me = t_exp <= xmax
        for i, var in enumerate(var_names):
            ax = fig.add_subplot(gs[i, j + 1])
            axes.append(ax)
            o = (originals or {}).get(ds)
            if o is not None and var in o:
                ax.plot(t[mk], np.asarray(o[var])[mk], 'k-' if original else 'k--', lw=1.5, label='Original model', zorder=3)
            s = (sols or {}).get(ds)
            if (not original) and s is not None and var in s:
                ax.plot(t[mk], np.asarray(s[var])[mk], '-', color=CAL_COLOR, lw=2.0, label='Calibrated model', zorder=4)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], c='k', marker='o', s=22, label='Experimental data', zorder=5)
            ax.set_xlim(0, xmax)
            ax.grid(True, alpha=0.4)
            if j == 0:
                ax.set_ylabel(var, fontsize=12)
            if i == n - 1:
                ax.set_xlabel('Time (h)', fontsize=12)
            if i == 0:
                tag = ''
                if ds == 'cal' and not original:
                    tag = ' [fitted]' if condition == calibration_condition else ' [not fitted]'
                ax.set_title(f"{DATASET_LABEL[ds]}: {dataset_label(system_data, condition, ds)}{tag}\n"
                             f"{_metric_line((metrics or {}).get(ds))}", fontsize=11)

    _shared_legend(fig, axes, ncol=3, y=0.01)
    what = 'Original model' if original else (f"Calibrated model {label}" if label else 'Calibrated model')
    if calibrated:
        what += f" (fitted: {', '.join(calibrated)})"
    fig.suptitle(f"{what} - {condition_label(system_data, condition)}", fontsize=15, y=0.97)
    return save_fig_file(fig, fig_outdir, filename)


# ----------------------------------------------------------------------------------------------
# Parameter analysis / residuals (one run, one condition)
# ----------------------------------------------------------------------------------------------
def plot_corr_matrix(fig_outdir: str, corr: np.ndarray, parameters_og_list: List[str], title: str,
                     filename: str = 'corr_matrix.png') -> None:
    """Heatmap of the parameter correlation matrix (determinability) of the analysed parameters."""
    mask = np.isnan(corr)
    side = max(6.0, 0.9 * len(parameters_og_list) + 3.0)
    fig, ax = plt.subplots(figsize=(side * 1.2, side))
    sns.heatmap(corr, mask=mask, xticklabels=parameters_og_list, yticklabels=parameters_og_list,
                annot=True, fmt=".3f", vmin=-1, vmax=1, center=0, cmap='Greys', ax=ax)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    save_fig_file(fig, fig_outdir, filename)


def plot_sensitivity_analysis(fig_outdir: str, sensitivity_df: pd.DataFrame, top5_plot_df: pd.DataFrame,
                              var_names: List[str], title: str, filename: str = 'sensitivity.png') -> None:
    """Sensitivity of every measured state to every parameter + the 5 most sensitive parameters."""
    palette = sns.color_palette("Greys", n_colors=sensitivity_df.shape[1])

    fig = plt.figure(figsize=(25, len(var_names) * 5), constrained_layout=True)
    gs = gridspec.GridSpec(len(var_names), 2, width_ratios=[2, 1], figure=fig)

    for i, state in enumerate(sensitivity_df.columns):
        ax = fig.add_subplot(gs[i, 0])
        ax.bar(sensitivity_df.index, sensitivity_df[state], color='gray', edgecolor='black', linewidth=1)
        ax.set_title(f"State {state}", pad=14, fontsize=14)
        ax.set_ylabel('General relative sensitivity (-)', fontsize=16)
        ax.axhline(0.01, color='black', linestyle='--', linewidth=1)    # non-sensitive threshold (slides)
        ax.set_xticks(range(len(sensitivity_df.index)))
        ax.set_xticklabels(sensitivity_df.index, rotation=90, ha='right')
        ax.grid(True, axis='y', linestyle='-', alpha=0.6)

    rows = slice(1, -1) if len(var_names) > 2 else slice(None)        # [FIX] gs[1:-1] is empty with <= 2 states
    ax_right = fig.add_subplot(gs[rows, 1])
    top5_plot_df.plot(kind='barh', stacked=True, color=palette, ax=ax_right)
    ax_right.invert_yaxis()
    ax_right.set_xlabel('General relative sensitivity, sum over variables (-)', fontsize=16)
    ax_right.set_title('5 most sensitive parameters', fontsize=18)
    ax_right.legend(title='State', fontsize=16)
    ax_right.grid(True, alpha=0.6)

    fig.suptitle(title, fontsize=18)
    save_fig_file(fig, fig_outdir, filename)


def plot_residuals_analysis(fig_outdir: str, residuals: np.ndarray, title: str,
                            filename: str = 'residuals.png') -> None:
    """Histogram of the (range-scaled, pooled) residuals with normal fit + normal probability plot."""
    residuals = np.asarray(residuals, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax0 = axes[0]
    ax0.hist(residuals, bins='auto', density=True, alpha=0.6, color='gray', edgecolor='black')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax0.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    ax0.plot(x, stats.norm.pdf(x, mu, std), 'k', linewidth=2)
    ax0.set_xlabel('Residuals (range-scaled)', fontsize=14)
    ax0.set_ylabel('Density distribution', fontsize=14)
    ax0.set_title('Histogram of Residuals\nand Normal Distribution', fontsize=16)
    ax0.grid(True)

    ax1 = axes[1]
    (osm, osr), (slope, intercept, _r) = stats.probplot(residuals, dist="norm")
    ax1.scatter(osm, osr, color='gray', edgecolor='black')
    ax1.plot(osm, slope * osm + intercept, color='black', linewidth=2)
    ax1.set_title('Normal Probability Plot', fontsize=16)
    ax1.set_xlabel('Theoretical Quantiles', fontsize=14)
    ax1.set_ylabel('Ordered Values', fontsize=14)
    ax1.grid(True)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    save_fig_file(fig, fig_outdir, filename)


# ----------------------------------------------------------------------------------------------
# Summaries over the iterations of ONE parameter combination (lists extracted from the stored tables)
# ----------------------------------------------------------------------------------------------
def plot_sensitivity_summary(fig_outdir: str, sensitivity_list: List[pd.DataFrame], title: str,
                             filename: str = 'sensitivity_summary.png') -> None:
    """Mean +- sd of the sensitivity matrices (rows = parameters, columns = states) of several runs."""
    dfs = [d for d in sensitivity_list if d is not None and not d.empty]
    if len(dfs) == 0:
        return
    params = dfs[0].index.tolist()
    states = dfs[0].columns.tolist()
    stacked = np.stack([d.reindex(index=params, columns=states).values.astype(float) for d in dfs], axis=0)
    mean_sens = np.nanmean(stacked, axis=0)
    std_sens = np.nanstd(stacked, axis=0)

    fig, axes = plt.subplots(len(states), 1, figsize=(10, 3 * len(states)), sharex=True, squeeze=False)
    for j, state in enumerate(states):
        ax = axes[j, 0]
        ax.bar(params, mean_sens[:, j], yerr=std_sens[:, j], capsize=4, color='gray', edgecolor='black', linewidth=1)
        ax.axhline(0.01, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"Sensitivity of '{state}' (mean +- sd, {len(dfs)} runs)")
        ax.set_ylabel("Sensitivity")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.5)
    axes[-1, 0].set_xlabel("Parameter")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_fig_file(fig, fig_outdir, filename)


def plot_residuals_summary(fig_outdir: str, residuals_list: List[np.ndarray], title: str,
                           filename: str = 'residuals_summary.png') -> None:
    """Aggregated residual histogram and Q-Q plot (mean +- sd over runs)."""
    residuals_all = [np.asarray(r, dtype=float) for r in residuals_list if r is not None and len(r) > 0]
    if len(residuals_all) == 0:
        return
    all_res = np.concatenate(residuals_all)
    mu_all, std_all = stats.norm.fit(all_res)

    bins = np.histogram_bin_edges(all_res, bins="auto")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    densities = np.vstack([np.histogram(r, bins=bins, density=True)[0] for r in residuals_all])
    mean_den = densities.mean(axis=0)
    std_den = densities.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0 = axes[0]
    ax0.bar(bin_centers, mean_den, width=np.diff(bins), alpha=0.7, yerr=std_den, capsize=4, color='gray',
            edgecolor='black', linewidth=1)
    x_norm = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    ax0.plot(x_norm, stats.norm.pdf(x_norm, mu_all, std_all), 'k-', lw=2, label="Normal fit")
    ax0.set_xlabel("Residual value (range-scaled)")
    ax0.set_ylabel("Density")
    ax0.set_title("Residuals histogram (mean +- sd) with normal fit")
    ax0.legend()
    ax0.grid(alpha=0.4)

    ax1 = axes[1]
    m = min(r.size for r in residuals_all)
    probs = (np.arange(1, m + 1) - 0.5) / m
    theo = stats.norm.ppf(probs, loc=mu_all, scale=std_all)
    sorted_all = np.vstack([np.sort(r)[:m] for r in residuals_all])
    mean_q = sorted_all.mean(axis=0)
    std_q = sorted_all.std(axis=0)
    ax1.errorbar(theo, mean_q, yerr=std_q, fmt='o', color='gray', ecolor='gray', elinewidth=1, capsize=3)
    lims = [min(theo.min(), mean_q.min()), max(theo.max(), mean_q.max())]
    ax1.plot(lims, lims, 'k--')
    ax1.set_xlabel("Theoretical quantiles")
    ax1.set_ylabel("Mean ordered residuals")
    ax1.set_title("Aggregated Q-Q plot (mean +- sd)")
    ax1.grid(alpha=0.4)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_fig_file(fig, fig_outdir, filename)


def plot_corr_summary(fig_outdir: str, corr_list: List[np.ndarray], names: List[str], title: str,
                      filename: str = 'corr_matrix_summary.png') -> None:
    """Mean +- sd of the parameter correlation matrices of several runs (same parameters)."""
    mats = [c for c in corr_list if c is not None]
    if not mats:
        return
    stack = np.stack(mats, axis=0)
    mean_corr = np.nanmean(stack, axis=0)
    std_corr = np.nanstd(stack, axis=0)
    n = mean_corr.shape[0]
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{mean_corr[i, j]:.2f}\n+-{std_corr[i, j]:.2f}"

    side = max(6.0, 0.9 * n + 3.0)
    fig, ax = plt.subplots(figsize=(side * 1.2, side))
    sns.heatmap(mean_corr, annot=annot, fmt="", xticklabels=names, yticklabels=names, cmap="coolwarm", vmin=-1, vmax=1,
                center=0, annot_kws={"fontsize": 10, 'fontweight': 'bold'}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    save_fig_file(fig, fig_outdir, filename)


def plot_combo_solutions(system_data: Dict[str, Any], fig_outdir: str, condition: str, run_ids: List[str],
                         solutions: Dict[str, Dict[str, Any]], originals: Optional[Dict[str, Any]], title: str,
                         filename: str = 'solutions_all_iterations.png') -> None:
    """
    All iterations of one combination in one figure: rows = states, columns = calibration | validation.
        solutions : {run_id: {'cal': DataFrame|None, 'val': DataFrame|None}}
    """
    var_names = list(system_data['var_names'])
    var_exp = list(system_data['var_exp_names'])
    t = np.asarray(system_data['time_stamps_sim'], dtype=float)
    n = len(var_names)
    colors = sns.color_palette("Set2", n_colors=max(len(run_ids), 3))

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n + 1.2), squeeze=False, sharex='col')
    for j, ds in enumerate(DATASETS):
        t_exp, df_exp = experiment_data(system_data, condition, ds)
        xmax = dataset_xmax(system_data, condition, ds)
        mk = t <= xmax
        me = t_exp <= xmax
        for i, var in enumerate(var_names):
            ax = axes[i, j]
            o = (originals or {}).get(ds)
            if o is not None and var in o:
                ax.plot(t[mk], np.asarray(o[var])[mk], 'k--', lw=1.5, label='Original model', zorder=4)
            for k, rid in enumerate(run_ids):
                s = solutions.get(rid, {}).get(ds)
                if s is not None and var in s:
                    ax.plot(t[mk], np.asarray(s[var])[mk], '-', color=colors[k], lw=1.6, alpha=0.9,
                            label=rid.split('__')[-1], zorder=3)
            if var in var_exp:
                ax.scatter(t_exp[me], np.asarray(df_exp[var])[me], c='k', marker='o', s=22, label='Experimental data', zorder=5)
            ax.set_xlim(0, xmax)
            ax.grid(True, alpha=0.4)
            if j == 0:
                ax.set_ylabel(var, fontsize=12)
            if i == 0:
                ax.set_title(f"{DATASET_LABEL[ds]}: {dataset_label(system_data, condition, ds)}", fontsize=12)
            if i == n - 1:
                ax.set_xlabel('Time (h)', fontsize=12)
    _shared_legend(fig, axes, ncol=8, y=0.0)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_fig_file(fig, fig_outdir, filename)


def plot_combo_summary(system_data: Dict[str, Any], store, combo_id: str, calibration_condition: str = 'N_def') -> None:
    """
    Summary figures of one parameter combination over its finished iterations (needs >= 2), built ONLY from the stored
    tables/trajectories (so it also works for a combination finished in an earlier session).
    Files: combos/<combo_id>/<condition>_{solutions_all_iterations, sensitivity_summary, corr_matrix_summary,
           residuals_summary_cal, residuals_summary_val}.png
    """
    runs = store.read('runs')
    runs = runs[(runs['combo_id'] == combo_id) & (runs['status'] == 'done')].sort_values('iteration')
    run_ids = runs['run_id'].tolist()
    if len(run_ids) < 2:
        return
    outdir = store.combo_dir(combo_id)
    calibrated = [p for p in str(runs['calibrated_parameters'].iloc[0]).split(';') if p and p != 'nan']

    sens_t = store.read('sensitivity')
    corr_t = store.read('correlations')
    res_t = store.read('residuals')
    for cond in CONDITIONS:
        sols = {rid: {ds: store.load_solution(rid, cond, ds, t_grid=system_data['time_stamps_sim']) for ds in DATASETS}
                for rid in run_ids}
        orig = {ds: store.load_solution('original', cond, ds, t_grid=system_data['time_stamps_sim']) for ds in DATASETS}
        tag = f"{combo_id} ({', '.join(calibrated)}), {len(run_ids)} runs - {condition_label(system_data, cond)}"
        plot_combo_solutions(system_data, outdir, cond, run_ids, sols, orig, title=f"All iterations: {tag}",
                             filename=f"{cond}_solutions_all_iterations.png")

        s = sens_t[(sens_t['condition'] == cond) & (sens_t['run_id'].isin(run_ids))]
        if not s.empty:
            lst = [g.pivot(index='parameter', columns='variable', values='sensitivity') for _, g in s.groupby('run_id')]
            plot_sensitivity_summary(outdir, lst, title=f"Sensitivity: {tag}", filename=f"{cond}_sensitivity_summary.png")

        c = corr_t[(corr_t['condition'] == cond) & (corr_t['run_id'].isin(run_ids))]
        if not c.empty and calibrated:
            idx = {p: i for i, p in enumerate(calibrated)}
            mats = []
            for _, g in c.groupby('run_id'):
                m = np.full((len(calibrated), len(calibrated)), np.nan)
                np.fill_diagonal(m, 1.0)
                for a, b, k in zip(g['param_1'], g['param_2'], g['kappa']):
                    if a in idx and b in idx:
                        m[idx[a], idx[b]] = m[idx[b], idx[a]] = k
                mats.append(m)
            plot_corr_summary(outdir, mats, calibrated, title=f"Parameter correlation: {tag}",
                              filename=f"{cond}_corr_matrix_summary.png")

        for ds in DATASETS:
            r = res_t[(res_t['condition'] == cond) & (res_t['dataset'] == ds) & (res_t['run_id'].isin(run_ids))]
            if not r.empty:
                lst = [g['residual_scaled'].values for _, g in r.groupby('run_id')]
                plot_residuals_summary(outdir, lst, title=f"Residuals ({DATASET_LABEL[ds].lower()}): {tag}",
                                       filename=f"{cond}_residuals_summary_{ds}.png")