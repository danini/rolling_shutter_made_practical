#!/usr/bin/env python3
"""
Plot benchmark results for RS relative pose solvers.

Reads the SuperANSAC benchmark CSV format:
  method,experiment,param_name,param_value,trial,R_err_deg,t_err_deg,omega_err,v_err,inlier_count,iterations,time_ms

Usage:
  python3 plot_benchmark.py results.csv [output_dir]
  python3 plot_benchmark.py file1.csv file2.csv ... [--outdir DIR]
"""
import sys
import os
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pip install pandas required")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'lines.linewidth': 2.2,
    'lines.markersize': 7,
    'figure.dpi': 150,
})

# ============================================================
# Method configuration
# ============================================================
# Order determines legend order (top to bottom)
METHOD_ORDER = [
    'ac_rs_7direct',
    # 'ac_rs',
    'ac_rs_4',
    'pc_rs',
    'gs_5pt',
    'dai_20pt',
    'dai_44pt',
    'npt_lin',
    'npt_gn',
    'npt_cv',
    'npt_iter',
    # MAGSAC+NestedRANSAC variants
    'ac_rs_7direct_magsac',
    'ac_rs_magsac',
    'ac_rs_4_magsac',
    'pc_rs_magsac',
    'gs_5pt_magsac',
]

METHOD_LABELS = {
    'ac_rs_7direct': 'Ours (7AC)',
    'ac_rs':         'AC-RS 5pt (ours)',
    'ac_rs_4':       'AC-RS 4pt (ours)',
    'pc_rs':         'PC-RS (12pt)',
    'gs_5pt':        'GS-5PC',
    'npt_lin':       'Npt+linRS',
    'npt_gn':        'Npt+linRS+GN',
    'npt_cv':        'Npt+cvRS',
    'npt_iter':      'Npt+iterRS',
    'ac_rs_6eigval': '6AC-Eigval',
    'dai_20pt':      'RS-20PC',
    'dai_44pt':      'RS-44PC',
    # MAGSAC variants
    'ac_rs_7direct_magsac': '7AC-Direct+MAG',
    'ac_rs_magsac':         'AC-RS 5pt+MAG',
    'ac_rs_4_magsac':       'AC-RS 4pt+MAG',
    'pc_rs_magsac':         'PC-RS+MAG',
    'gs_5pt_magsac':        '5pt (GS)+MAG',
}

METHOD_COLORS = {
    'ac_rs_7direct': '#d62728',   # red - highlight
    'ac_rs':         '#2ca02c',   # green
    'ac_rs_4':       '#17becf',   # cyan
    'pc_rs':         '#ff7f0e',   # orange
    'gs_5pt':        '#1f77b4',   # blue
    'npt_lin':       '#9467bd',   # purple
    'npt_gn':        '#8c564b',   # brown
    'npt_cv':        '#e377c2',   # pink
    'npt_iter':      '#bcbd22',   # olive
    'ac_rs_6eigval': '#7f7f7f',   # gray
    'dai_20pt':      '#aec7e8',   # light blue
    'dai_44pt':      '#ffbb78',   # light orange
    # MAGSAC variants - same colors, will be distinguished by linestyle
    'ac_rs_7direct_magsac': '#d62728',
    'ac_rs_magsac':         '#2ca02c',
    'ac_rs_4_magsac':       '#17becf',
    'pc_rs_magsac':         '#ff7f0e',
    'gs_5pt_magsac':        '#1f77b4',
}

METHOD_MARKERS = {
    'ac_rs_7direct': 'D',
    'ac_rs':         'o',
    'ac_rs_4':       'h',
    'pc_rs':         's',
    'gs_5pt':        '^',
    'npt_lin':       'v',
    'npt_gn':        '<',
    'npt_cv':        '>',
    'npt_iter':      'P',
    'ac_rs_6eigval': 'X',
    'dai_20pt':      'd',
    'dai_44pt':      'p',
    # MAGSAC variants - same markers
    'ac_rs_7direct_magsac': 'D',
    'ac_rs_magsac':         'o',
    'ac_rs_4_magsac':       'h',
    'pc_rs_magsac':         's',
    'gs_5pt_magsac':        '^',
}

METHOD_LINESTYLES = {
    'ac_rs_7direct': '-',
    'ac_rs':         '-',
    'ac_rs_4':       '-',
    'pc_rs':         '--',
    'gs_5pt':        '--',
    'npt_lin':       ':',
    'npt_gn':        ':',
    'npt_cv':        ':',
    'npt_iter':      ':',
    'ac_rs_6eigval': '-.',
    'dai_20pt':      '--',
    'dai_44pt':      '--',
    # MAGSAC variants - dashed to distinguish from MSAC (solid)
    'ac_rs_7direct_magsac': '-.',
    'ac_rs_magsac':         '-.',
    'ac_rs_4_magsac':       '-.',
    'pc_rs_magsac':         '-.',
    'gs_5pt_magsac':        '-.',
}

# Map experiment names to display labels
EXPERIMENT_LABELS = {
    'increasing_rs':       'RS magnitude',
    'increasing_N':        'Number of correspondences $N$',
    'increasing_pt_noise': 'Pixel noise $\\sigma$ (px)',
    'increasing_ac_noise': 'Affine noise',
    'increasing_outliers': 'Outlier ratio',
}

ERROR_LABELS = {
    'R_err_deg':  'Rotation error (deg)',
    't_err_deg':  'Translation error (deg)',
    'omega_err':  '$\\|\\Delta\\omega\\|$',
    'v_err':      '$\\|\\Delta v\\|$',
    'time_ms':    'Runtime (ms)',
}

# Which experiments to include in the main plots
MAIN_EXPERIMENTS = ['increasing_rs', 'increasing_pt_noise', 'increasing_N',
                    'increasing_ac_noise', 'increasing_outliers']

# Which methods are RS-aware (have meaningful omega_err, v_err)
RS_METHODS = [m for m in METHOD_ORDER if not m.startswith('gs_5pt')]


def load_data(csv_paths):
    """Load and concatenate one or more CSV files."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def plot_sweep(df, experiment, error_metric, ax, log_y=True,
               methods=None, show_legend=True, ylim=None):
    """Plot median for one experiment sweep."""
    if methods is None:
        methods = METHOD_ORDER

    for method in methods:
        sub = df[(df['method'] == method) & (df['experiment'] == experiment)]
        if sub.empty:
            continue

        grouped = sub.groupby('param_value')[error_metric]
        medians = grouped.median()

        x = medians.index.values
        y = medians.values

        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, 'gray')
        marker = METHOD_MARKERS.get(method, '.')
        ls = METHOD_LINESTYLES.get(method, '-')

        ax.plot(x, y, label=label, color=color, marker=marker,
                linestyle=ls, linewidth=1.8, markersize=5)

    ax.set_xlabel(EXPERIMENT_LABELS.get(experiment, experiment))
    ax.set_ylabel(ERROR_LABELS.get(error_metric, error_metric))
    if log_y:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)
    if show_legend:
        ax.legend(fontsize=12, loc='best', ncol=1)
    ax.grid(True, alpha=0.3, which='both')


def make_all_plots(csv_paths, output_dir='.'):
    df = load_data(csv_paths)
    print(f"Loaded {len(df)} rows, methods: {sorted(df['method'].unique())}")
    print(f"Experiments: {sorted(df['experiment'].unique())}")

    experiments = MAIN_EXPERIMENTS
    metrics = ['R_err_deg', 't_err_deg', 'omega_err', 'v_err']

    # ==========================================
    # Full 4x3 grid: all metrics x 3 experiments
    # ==========================================
    fig, axes = plt.subplots(len(metrics), len(experiments),
                              figsize=(22, 13), constrained_layout=True)

    for j, exp in enumerate(experiments):
        for i, metric in enumerate(metrics):
            show_leg = (j == len(experiments) - 1 and i == 0)
            methods = METHOD_ORDER if metric in ('R_err_deg', 't_err_deg', 'time_ms') else RS_METHODS
            plot_sweep(df, exp, metric, axes[i, j], show_legend=show_leg,
                       methods=methods)

    fig.savefig(os.path.join(output_dir, 'benchmark_all.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'benchmark_all.png'), dpi=150)
    plt.close(fig)
    print("  Saved benchmark_all.pdf/png")

    # ==========================================
    # Pose-only figure: 2 rows (R_err, t_err) x 3 experiments
    # ==========================================
    fig, axes = plt.subplots(2, len(experiments),
                              figsize=(22, 7), constrained_layout=True)
    for j, exp in enumerate(experiments):
        plot_sweep(df, exp, 'R_err_deg', axes[0, j],
                   show_legend=(j == len(experiments) - 1))
        plot_sweep(df, exp, 't_err_deg', axes[1, j], show_legend=False)

    fig.savefig(os.path.join(output_dir, 'benchmark_pose.pdf'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'benchmark_pose.png'), dpi=200)
    plt.close(fig)
    print("  Saved benchmark_pose.pdf/png")

    # ==========================================
    # RS parameters: omega_err and v_err for RS methods only
    # ==========================================
    fig, axes = plt.subplots(2, len(experiments),
                              figsize=(22, 7), constrained_layout=True)
    for j, exp in enumerate(experiments):
        plot_sweep(df, exp, 'omega_err', axes[0, j],
                   methods=RS_METHODS, show_legend=(j == len(experiments) - 1))
        plot_sweep(df, exp, 'v_err', axes[1, j],
                   methods=RS_METHODS, show_legend=False)

    fig.savefig(os.path.join(output_dir, 'benchmark_rs_params.pdf'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'benchmark_rs_params.png'), dpi=200)
    plt.close(fig)
    print("  Saved benchmark_rs_params.pdf/png")

    # ==========================================
    # Individual experiment figures
    # ==========================================
    for exp in experiments:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
        for i, metric in enumerate(metrics):
            methods = METHOD_ORDER if metric in ('R_err_deg', 't_err_deg') else RS_METHODS
            show_leg = (i == 0)
            plot_sweep(df, exp, metric, axes[i], show_legend=show_leg,
                       methods=methods)
        exp_short = exp.replace('increasing_', '')
        fig.savefig(os.path.join(output_dir, f'benchmark_{exp_short}.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, f'benchmark_{exp_short}.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved benchmark_{exp_short}.pdf/png")

    # ==========================================
    # Runtime comparison
    # ==========================================
    fig, axes = plt.subplots(1, len(experiments),
                              figsize=(22, 4), constrained_layout=True)
    for j, exp in enumerate(experiments):
        plot_sweep(df, exp, 'time_ms', axes[j], log_y=True,
                   show_legend=(j == len(experiments) - 1))
    fig.savefig(os.path.join(output_dir, 'benchmark_runtime.pdf'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'benchmark_runtime.png'), dpi=200)
    plt.close(fig)
    print("  Saved benchmark_runtime.pdf/png")

    # ==========================================
    # Print summary tables
    # ==========================================
    print("\n=== Median Errors (all trials) ===")
    for exp in experiments:
        print(f"\n--- {EXPERIMENT_LABELS.get(exp, exp)} ---")
        sub_exp = df[df['experiment'] == exp]
        vals = sorted(sub_exp['param_value'].unique())

        # Header
        print(f"{'Value':>8s}", end="")
        for method in METHOD_ORDER:
            if not sub_exp[sub_exp['method'] == method].empty:
                lbl = METHOD_LABELS.get(method, method)[:12]
                print(f"  {lbl:>12s}", end="")
        print()

        for val in vals:
            row = f"{val:>8.2f}"
            for method in METHOD_ORDER:
                sub = sub_exp[(sub_exp['method'] == method)
                              & (sub_exp['param_value'] == val)]
                if sub.empty:
                    continue
                r_med = sub['R_err_deg'].median()
                row += f"  {r_med:>12.2f}"
            print(row)


if __name__ == '__main__':
    # Parse arguments
    csv_paths = []
    output_dir = 'benchmark_plots'

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--outdir' and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            csv_paths.append(args[i])
            i += 1

    if not csv_paths:
        csv_paths = ['benchmark_superansac_results.csv']

    os.makedirs(output_dir, exist_ok=True)
    make_all_plots(csv_paths, output_dir)
