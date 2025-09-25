import os
import pandas as pd
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats

from System_info import system_info as system_data
from DAE_Systems_Simulations import simulate_model

# Global functions to be used across plotting functions
def _iter_index(iteration: Optional[int]) -> int:
    """Map None -> 0 (original), otherwise i -> i+1."""

    return 0 if iteration is None else int(iteration) + 1

def _ensure_dir(path: str) -> None:
    """Ensure the directory exists."""

    os.makedirs(path, exist_ok=True)

def save_fig(fig: plt.Figure, 
            fig_outdir: str, 
            name: str, 
            iteration: Optional[int] = None
            ) -> None:
    """Save figure to the specified directory with appropriate naming."""

    _ensure_dir(fig_outdir)
    if iteration is None:
        fname = f"{name}_original.png"
    elif iteration == "summary":
        fname = f"{name}_summary.png"
    else:
        fname = f"{name}_{iteration+1}.png"

    fig.savefig(os.path.join(fig_outdir, fname), dpi=200, bbox_inches="tight")
    fig.canvas.draw_idle()
    plt.show()

def figures_dir_from_csv_path(csv_path: str) -> str:
    """
    Generate a directory path for saving figures based on the CSV file path and ensure it exists.
    """

    outdir = os.path.join(os.path.dirname(csv_path), "Figures")
    _ensure_dir(outdir)
    return outdir


# Plotting functions
def plot_corr_matrix(corr: np.ndarray, 
                    parameters_og_list: List[str], 
                    fig_outdir: str, 
                    iteration: Optional[int] = None
                    ) -> None:
    """
    Plots the correlation matrix of parameters.
    """
    mask = np.isnan(corr)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, mask=mask,
        xticklabels=parameters_og_list, yticklabels=parameters_og_list,
        annot=True, fmt=".3f",
        vmin=-1, vmax=1, center=0, cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Parameter Correlation Matrix")
    fig.tight_layout()

    save_fig(fig, fig_outdir, "Corr_Matrix", iteration)


def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame,
                                top5_plot_df: pd.DataFrame,
                                var_names: List[str], 
                                fig_outdir: str, 
                                iteration: Optional[int] = None,
                                ) -> None:
    """
    Plots the sensitivity analysis results for all states and highlights the top 5 most sensitive parameters.
    """
    palette = sns.color_palette("Set2", n_colors=sensitivity_df.shape[1])

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(len(var_names), 2, width_ratios=[2, 1], figure=fig)

    axes_left = []
    for i, state in enumerate(sensitivity_df.columns):
        ax = fig.add_subplot(gs[i, 0])
        ax.bar(sensitivity_df.index, sensitivity_df[state], color='red')
        ax.set_title(f"State {state}", pad=14)  
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_xticks(range(len(sensitivity_df.index)))
        ax.set_xticklabels(sensitivity_df.index, rotation=90, ha='right')
        ax.grid(True, axis='y', linestyle='-', alpha=0.6)
        axes_left.append(ax)
    
    ax_right = fig.add_subplot(gs[1:-1, 1]) 
    top5_plot_df.plot(kind='barh', stacked=True, color=palette, ax=ax_right)
    ax_right.invert_yaxis()
    ax_right.set_xlabel('Mean relative sensitivity', fontsize=14)
    ax_right.set_title('5 most sensitive parameters', fontsize=16)
    ax_right.legend(title='State', fontsize=12)

    fig.suptitle('Parameter Sensitivities Overview', fontsize=18)
    plt.subplots_adjust(hspace=0.6)  
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    
    save_fig(fig, fig_outdir, "Sensitivity", iteration)

def plot_residuals_analysis(residuals: np.ndarray, 
                            fig_outdir: str, 
                            iteration: Optional[int] = None
                            ) -> None:
    """
    Function to plot the residuals analysis including histogram and normal probability plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram of residuals and normal distribution fit
    ax0 = axes[0]
    n, bins, patches = ax0.hist(residuals, bins='auto', density=True, alpha=0.6, color='blue')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax0.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax0.plot(x, p, 'k', linewidth=2)
    ax0.set_xlabel('Residuals')
    ax0.set_ylabel('Density distribution')
    ax0.set_title('Histogram of Residuals\nand Normal Distribution')
    ax0.grid(True)

    # Normal Probability Plot 
    ax1 = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title('Normal Probability Plot')
    ax1.set_xlabel('Theoretical Quantiles')
    ax1.set_ylabel('Ordered Values')
    ax1.grid(True)

    plt.tight_layout()

    save_fig(fig, fig_outdir, "Residuals_Analysis", iteration)

def plotting_comparison(iteration:Optional[int], 
                        params_list: list, 
                        parameters_updated: Dict[str, float], 
                        AIC: float, 
                        fig_outdir: str, 
                        new_og: Optional[Dict[str, float]] = None 
                        ) -> None:
    """
    Plots the comparison between the original model parameters and the updated parameters against validation data.
    Parameters:
        - params_list: list of parameters that are being calibrated.
        - parameters_updated: dictionary of updated parameters for the model after calibration.
        - AIC: Akaike Information Criterion value for the updated model.
    """

    print(" ")
    print("---------------------------------------------------------------------------------")
    print("---------------------- MODEL COMPARISON TO VALIDATION DATA ----------------------")
    print("---------------------------------------------------------------------------------")
    print(" ")
        
    def darken_color(color, factor=0.6):
        rgb = mcolors.to_rgb(color)
        return tuple(factor * c for c in rgb)

    var_names = system_data['var_names']
    colors = system_data['colors']
    time_stamps_sim = system_data['time_stamps_sim']
    df_val = system_data['df_val']
    t_exp_v = system_data['t_exp_v']
    x0_exp_v = system_data['x0_exp_v']

    parameters_og = system_data['parameters']
    parameters_og_new = {}

    if new_og is None: # Original parameters case
        parameters_og_new = parameters_og.copy()
        
    else: # Calibration case
        for key, value in parameters_og.items():
            if key in list(new_og.keys()):
                parameters_og_new[key] = new_og[key]
            else:
                parameters_og_new[key] = value
        
    original_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_exp_v, 
                                    parameters=parameters_og_new, 
                                    time=time_stamps_sim)

    if original_sol_v is None:
        print("!!!!!!!!!!!!! Simulation with original parameters and validation data failed. Please check the parameters and initial conditions.")
        return None

    if iteration is not None:
        new_sol_v = simulate_model(simulation_type='normal', 
                                    x0=x0_exp_v, 
                                    parameters=parameters_updated, 
                                    time=time_stamps_sim)
        if new_sol_v is None:
            print("!!!!!!!!!!!!! Simulation with updated parameters and validation data failed. Please check the parameters and initial conditions.")
            return None
    
    fig, axes = plt.subplots(len(var_names), 1, figsize=(14, 14))
    fig.subplots_adjust(left=0.35, top=0.88, hspace=0.2)

    for i, var in enumerate(var_names):
        axes[i].scatter(t_exp_v, df_val[var], marker='o', label=f"{var} exp validation", color=colors[var])
        axes[i].plot(time_stamps_sim, original_sol_v[var], '-', label=f"{var} original", color=colors[var])
        if iteration is not None:
            axes[i].plot(time_stamps_sim, new_sol_v[var], '--', label=f"{var} new",color=darken_color(colors[var]))
        axes[i].set_xlabel("Time (h)")
        axes[i].set_ylabel(var)
        axes[i].legend()
        axes[i].grid(True)
    # Create a text box for parameters and AIC
    textstr = "\n".join([f"{k}: {v}" for k, v in parameters_updated.items()])

    fig.text(
        0.15, 0.6, textstr,
        ha='center', va='center', fontsize=14,         # slightly larger
        bbox={'boxstyle': 'square,pad=1.0',  # more padding
            'facecolor': 'white',
            'alpha': 0.8,
            'edgecolor': 'none'}
            )

    fig.text(
        0.15, 0.4, f"AIC = {AIC:.4f}",
        ha='center', va='top', fontsize=16,
        fontweight='bold'
        )

    if iteration is None:
        title = ("Initial Model vs. Validation Data")
    else:
        title = (f"New Model fitting {params_list} vs. Validation Data")

    fig.suptitle(title, fontsize=18, linespacing=2, y=0.92)

    save_fig(fig, fig_outdir, "Plotting_Comparison", iteration)


def plot_sensitivity_summary(fig_outdir: str, 
                            sensitivity_df_all: list
                            ) -> None:
    """Plot sensitivity analysis results for all iterations with mean and std deviation."""
    stacked_sens = np.stack([df.values for df in sensitivity_df_all], axis=0)
    mean_sens = stacked_sens.mean(axis=0)
    std_sens  = stacked_sens.std(axis=0)

    params = sensitivity_df_all[0].index.tolist()
    states = sensitivity_df_all[0].columns.tolist()

    fig, axes = plt.subplots(len(states), 1, figsize=(8, 2*len(states)), sharex=True)

    for j, state in enumerate(states):
        axes[j].bar(params,
                    mean_sens[:, j],
                    yerr=std_sens[:, j],
                    capsize=4,
                    color='red',
                    linewidth=1.5) 
        axes[j].set_title(f"Sensitivity of “{state}” (mean ± σ)")
        axes[j].set_ylabel("Sensitivity")
        axes[j].tick_params(axis="x", rotation=90)
        axes[j].grid(axis="y", alpha=0.5)
    axes[-1].set_xlabel("Parameter")
    plt.tight_layout()

    save_fig(fig, fig_outdir, "Sensitivity", "summary")

def plot_residuals_summary(fig_outdir: str, 
                            residuals_all: list
                            ) -> None:
    """Plot aggregated residuals analysis for all iterations with mean and std deviation."""
    all_res    = np.concatenate(residuals_all)
    mu_all, std_all = stats.norm.fit(all_res)

    bins       = np.histogram_bin_edges(all_res, bins="auto")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    densities  = np.vstack([np.histogram(r, bins=bins, density=True)[0] for r in residuals_all])

    mean_den = densities.mean(axis=0)
    std_den  = densities.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax0 = axes[0]
    ax0.bar(bin_centers, mean_den,
            width=np.diff(bins), alpha=0.7, yerr=std_den,
            capsize=4, color='blue', linewidth=1.5)
    x_norm = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    ax0.plot(x_norm, stats.norm.pdf(x_norm, mu_all, std_all), 'k-', lw=2, label="Normal fit")
    ax0.set_xlabel("Residual value")
    ax0.set_ylabel("Density")
    ax0.set_title("Residuals Histogram\n(mean ± σ) with Normal Fit")
    ax0.legend()
    ax0.grid(alpha=0.4)

    ax1 = axes[1]
    m     = residuals_all[0].size
    probs = (np.arange(1, m+1) - 0.5) / m
    theo  = stats.norm.ppf(probs, loc=mu_all, scale=std_all)
    sorted_all = np.vstack([np.sort(r) for r in residuals_all])
    mean_q     = sorted_all.mean(axis=0)
    std_q      = sorted_all.std(axis=0)

    ax1.errorbar(theo, mean_q, yerr=std_q, 
                fmt='o', ecolor='gray', elinewidth=1, capsize=3)
    lims = [min(theo.min(), mean_q.min()), max(theo.max(), mean_q.max())]
    ax1.plot(lims, lims, 'r--')
    ax1.set_xlabel("Theoretical Quantiles")
    ax1.set_ylabel("Mean Ordered Residuals")
    ax1.set_title("Aggregated Q–Q Plot\n(mean ± σ)")
    ax1.grid(alpha=0.4)

    plt.tight_layout()

    save_fig(fig, fig_outdir, "Residuals_Analysis", "summary")

def plot_corr_summary(fig_outdir: str, 
                        corr_stack: np.ndarray, 
                        mean_corr: np.ndarray, 
                        parameters_og_list: list
                        ) -> None:
    """Plot aggregated correlation matrix for all iterations with mean and std deviation."""
    
    std_corr   = corr_stack.std(axis=0)
    n      = mean_corr.shape[0]
    annot  = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{mean_corr[i,j]:.2f}\n±{std_corr[i,j]:.2f}"

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(mean_corr,
                annot=annot,
                fmt="",
                xticklabels=parameters_og_list,
                yticklabels=parameters_og_list,
                cmap="coolwarm",
                center=0,
                annot_kws={"fontsize":10, 'fontweight':'bold'},
                ax=ax)
    ax.set_title("Parameter Correlation Matrix (mean ± σ)")

    fig.tight_layout()

    save_fig(fig, fig_outdir, "Corr_Matrix", "summary")