from textwrap import wrap
from typing import Optional, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
blues = ["#63bff0", "#a7d5ed", "#0d88e6", ]
rads = ["#ea5545", "#e1a692", "#b30000", ]
greens = ['green', "#466964", "#48b5c4", "#3c4e4b"]
light_oranges = ["#edbf33", "#ede15b", "#ef9b20", ]
oranges = ["#e14d2a", "#ffb55a", "#fd841f"]
purples = ["#9080ff", "#beb9db", "#5e569b"]
pinks = ["#df979e", "#e4bcad", '#c80064']
grays = ["#b3bfd1", "#d7e1ee", "#54504c"]
limes = ["#d0ee11", "#d0f400", "#c9e52f", ]
bright_pinks = ["#de25da", "#ff80ff", "#eb44e8", ]
turquoise = ["#599e94", "#6cd4c5", "#466964", ]
browns = ["#553939", "#a77979", "#704f4f", ]
sage = ["#7d8f69", "#a9af7e", "#557153", ]
strong_blue = ["#5170fd", "#aee1fc", "#4636fc"]
maroon = ["#900c3f", "#c70039", '#900c3f']
COLORS_7 = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']

COLORS_20 = ["#adb5bd", "#7ec4cf", '#4F6272', "#3de0fe", "#c77dff", '#B7C3F3', '#2d6a4f', '#2a9d8f', '#8EB897',
             '#9a8c98',
             '#f6bd60', '#8a5a44', "#ede7b1", "#ff9505", '#e29578', '#f5cac3', "#f7d6e0", '#DD7596', '#DD7596',
             "#8c2f39", "#d0ee11",
             ]
COLORS_16 = ["#adb5bd", '#4F6272', "#c77dff", '#B7C3F3', '#2d6a4f', '#2a9d8f', '#8EB897', '#9a8c98',
             '#f6bd60', "#ede7b1", '#e29578', '#f5cac3', "#f7d6e0", '#DD7596', '#DD7596', "#8c2f39"]
COLORS_12 = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51',
             '#2a9d8f', "#d0ee11", "#5e569b", "#553939", "#b30000", "#b3bfd1"]


def plot_nn_regression_prediction(
        results_all: pd.DataFrame,
        results_adj_only: pd.DataFrame,
        results_meta_only: pd.DataFrame,
        lr: str = '',
        task: str = '',
        out_path: Optional[str] = None,
        cut: int = 200,
):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6))
    ax1.plot(results_all['Epoch'][:cut], results_all['losses'][:cut], c=COLORS[0], label='full input', )
    ax1.plot(results_adj_only['Epoch'][:cut], results_adj_only['losses'][:cut], c=COLORS[1],
             label='adjacency matrix input', )
    ax1.plot(results_meta_only['Epoch'][:cut], results_meta_only['losses'][:cut], c=COLORS[2],
             label='structure metrics input', )
    ax2.plot(results_all['Epoch'][:cut], results_all['r2s test'][:cut], c=COLORS[0], label='full input', )
    ax2.plot(results_adj_only['Epoch'][:cut], results_adj_only['r2s test'][:cut], c=COLORS[1],
             label='adjacency matrix input', )
    ax2.plot(results_meta_only['Epoch'][:cut], results_meta_only['r2s test'][:cut], c=COLORS[2],
             label='structure metrics input', )
    ax3.plot(results_all['Epoch'][:cut], results_all['r2s train'][:cut], c=COLORS[0], label='full input', )
    ax3.plot(results_adj_only['Epoch'][:cut], results_adj_only['r2s train'][:cut], c=COLORS[1],
             label='adjacency matrix input', )
    ax3.plot(results_meta_only['Epoch'][:cut], results_meta_only['r2s train'][:cut], c=COLORS[2],
             label='structure metrics input', )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R2 - test')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R2 - train')
    ax1.set_title(f'{task} - Prediction of teachability mean performance (lr: {lr})')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def plot_prediction_zoom(
        results_all: pd.DataFrame,
        results_adj_only: pd.DataFrame,
        results_meta_only: pd.DataFrame,
        lr: str = '',
        task: str = '',
        out_path: Optional[str] = None,
        cut: Optional[int] = None,
        on_end: bool = False,
):
    if on_end:
        zommed_all_results = results_all[-cut:]
        zommed_results_adj_only = results_adj_only[-cut:]
        zommed_results_meta_only = results_meta_only[-cut:]
    else:
        zommed_all_results = results_all[:cut]
        zommed_results_adj_only = results_adj_only[:cut]
        zommed_results_meta_only = results_meta_only[:cut]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6))
    ax1.plot(zommed_all_results['Epoch'], zommed_all_results['losses'], c=COLORS[0], label='full input', )
    ax1.plot(zommed_results_adj_only['Epoch'], zommed_results_adj_only['losses'], c=COLORS[1],
             label='adjacency matrix input', )
    ax1.plot(zommed_results_meta_only['Epoch'], zommed_results_meta_only['losses'], c=COLORS[2],
             label='structure metrics input', )
    ax2.plot(zommed_all_results['Epoch'], zommed_all_results['r2s test'], c=COLORS[0], label='full input', )
    ax2.plot(zommed_results_adj_only['Epoch'], zommed_results_adj_only['r2s test'], c=COLORS[1],
             label='adjacency matrix input', )
    ax2.plot(zommed_results_meta_only['Epoch'], zommed_results_meta_only['r2s test'], c=COLORS[2],
             label='structure metrics input', )
    ax3.plot(zommed_all_results['Epoch'], zommed_all_results['r2s train'], c=COLORS[0], label='full input', )
    ax3.plot(zommed_results_adj_only['Epoch'], zommed_results_adj_only['r2s train'], c=COLORS[1],
             label='adjacency matrix input', )
    ax3.plot(zommed_results_meta_only['Epoch'], zommed_results_meta_only['r2s train'], c=COLORS[2],
             label='structure metrics input', )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R2 - test')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R2 - train')
    ax1.set_title(f'{task} - Prediction of teachability mean performance (lr: {lr})', loc='center', wrap=True)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def plot_prediction_final_r2(
        results_all: pd.DataFrame,
        results_adj_only: pd.DataFrame,
        results_meta_only: pd.DataFrame,
        lr: str = '',
        task: str = '',
        out_path: Optional[str] = None,
        cut: Optional[int] = None,
):
    data_types = [
        'full input',
        'adjacency matrix input',
        'structure metrics input',
    ]
    mean_train_r2s = [
        results_all['r2s train'][-cut:].mean(),
        results_adj_only['r2s train'][-cut:].mean(),
        results_meta_only['r2s train'][-cut:].mean(),
    ]
    mean_test_r2s = [
        results_all['r2s test'][-cut:].mean(),
        results_adj_only['r2s test'][-cut:].mean(),
        results_meta_only['r2s test'][-cut:].mean(),
    ]
    max_train_r2s = [
        results_all['r2s train'][-cut:].max(),
        results_adj_only['r2s train'][-cut:].max(),
        results_meta_only['r2s train'][-cut:].max(),
    ]
    max_test_r2s = [
        results_all['r2s test'][-cut:].max(),
        results_adj_only['r2s test'][-cut:].max(),
        results_meta_only['r2s test'][-cut:].max(),
    ]
    fig, (ax1, ax2,) = plt.subplots(2, 1, figsize=(7, 6))
    bar_size = 0.15
    padding = 0.15
    y_locs = np.arange(2 * (bar_size * 3 + padding))
    for i in range(3):
        ax1.barh(y_locs + (i * bar_size), [max_train_r2s[i], max_test_r2s[i]], height=bar_size, color=COLORS[i],
                 label=data_types[i])
        ax1.set(
            yticks=y_locs + bar_size,
            yticklabels=['Train', 'Test'],
            ylim=[0 - padding, len(y_locs) - padding],
            xlim=[max(max(max_train_r2s), max(max_test_r2s)) - 0.09, max(max(max_train_r2s), max(max_test_r2s)) + 0.03],
        )
        ax1.set_xlabel('max r2 score')

        ax2.barh(y_locs + (i * bar_size), [mean_train_r2s[i], mean_test_r2s[i]], height=bar_size, color=COLORS[i],
                 label=data_types[i])
        ax2.set(
            yticks=y_locs + bar_size,
            yticklabels=['Train', 'Test'],
            ylim=[0 - padding, len(y_locs) - padding],
            xlim=[max(max(mean_train_r2s), max(mean_test_r2s)) - 0.09,
                  max(max(mean_train_r2s), max(mean_test_r2s)) + 0.03],
        )
        ax2.set_xlabel('mean r2 score')

    ax1.set_title(f'{task} - Prediction of teachability mean performance (after convergence) (lr: {lr})', loc='center',
                  wrap=True)
    ax1.legend()
    ax2.legend()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def _get_binned_results_table(
        metric_names: List[str],
        mean_metric_name: str,
        result_df: pd.DataFrame,
        plot_error_bars: bool,
        bins_size: int,
) -> pd.DataFrame:
    vals = np.linspace(result_df[mean_metric_name].min(), result_df[mean_metric_name].max(), bins_size)
    bins = vals.tolist()
    labels = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    result_df[f'{mean_metric_name}_2'] = pd.cut(x=result_df[mean_metric_name], bins=bins, labels=labels,
                                                include_lowest=True)
    eval_metrics = ['mean']
    if plot_error_bars:
        eval_metrics.append(np.std)
    agg_dict = {
        metric: eval_metrics
        for metric in metric_names
    }
    r2 = result_df.groupby([f'{mean_metric_name}_2'], as_index=False).agg(agg_dict).dropna()
    return r2


def plot_binned_metric_vs_mean_metric(
        metric_name: str,
        mean_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        plot_error_bars: bool = True,
        bins_size: int = 500,
        task: str = '',
        color_ind: int = 0,
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
):
    r2 = _get_binned_results_table(
        metric_names=[metric_name],
        mean_metric_name=mean_metric_name,
        result_df=result_df,
        bins_size=bins_size,
        plot_error_bars=plot_error_bars,
    )
    if plot_error_bars:
        plt.scatter(r2[f'{mean_metric_name}_2'], r2[metric_name]['mean'], c=COLORS[color_ind])
        plt.fill_between(
            r2[f'{mean_metric_name}_2'],
            r2[metric_name]['mean'] + r2[metric_name]['std'],
            r2[metric_name]['mean'] - r2[metric_name]['std'],
            color="#b3bfd1",
            alpha=0.2,
        )
    else:
        plt.scatter(r2[f'{mean_metric_name}_2'], r2[metric_name], c=COLORS[color_ind])
    plt.xlabel(f"mean {mean_metric_name.replace('_', ' ')}")
    plt.ylabel(metric_name.replace('_', ' '))
    plt.title(f"Mean bined {mean_metric_name.replace('_', ' ')} of an architecture and its mean"
              f" {metric_name.replace('_', ' ')},{title_addition} {task}", loc='center', wrap=True)
    plt.savefig(
        f"{plot_path}/{time_str}_{task}{name_addition}_binned_{metric_name}_vs_{mean_metric_name}.png")
    plt.show()


def plot_metric_vs_mean_metric(
        metric_name: str,
        mean_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        task: str = '',
        name_addition: str = '',
        color_ind: int = 0,
        local_base_path: Optional[str] = None,
):
    plt.scatter(result_df[mean_metric_name], result_df[metric_name], c=COLORS[color_ind])
    plt.xlabel(f'{mean_metric_name}')
    plt.ylabel(metric_name)
    plt.title(f'Mean {mean_metric_name} of an architecture and its mean {metric_name}', loc='center', wrap=True)
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}{name_addition}_{metric_name}_vs_{mean_metric_name}.png")
    plt.show()


def plot_3d_three_metrices_vs_mean_metric(
        metric_names: List[str],
        mean_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        name_addition: str = '',
        bins_size: int = 500,
        task: str = '',
        local_base_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    r2 = _get_binned_results_table(
        metric_names=metric_names,
        mean_metric_name=mean_metric_name,
        result_df=result_df,
        bins_size=bins_size,
        plot_error_bars=False,
    )
    x, y, z = metric_names
    p3d = ax.scatter(r2[x], r2[y], r2[z], s=30,
                     c=np.linspace(r2[f'{mean_metric_name}_2'].min(), r2[f'{mean_metric_name}_2'].max(),
                                   num=r2[f'{mean_metric_name}_2'].shape[0]),
                     cmap='Purples')
    fig.colorbar(p3d, pad=0.05, shrink=0.8, location='right', label=mean_metric_name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.tight_layout()
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}{name_addition}_three_metrices_vs_{mean_metric_name}.png")
    plt.show()


def plot_two_binned_metrices_vs_third_means_metrics_colors(
        metric_names: List[str],
        mean_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        color_scheme: str,
        plot_error_bars: bool = True,
        bins_size: int = 500,
        name_addition: str = '',
        task: str = '',
        local_base_path: Optional[str] = None,
):
    r2 = _get_binned_results_table(
        metric_names=metric_names,
        mean_metric_name=mean_metric_name,
        result_df=result_df,
        bins_size=bins_size,
        plot_error_bars=plot_error_bars,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    x, y = metric_names
    if plot_error_bars:
        ax.scatter(
            r2[x]['mean'],
            r2[y]['mean'],
            c=np.linspace(
                r2[f'{mean_metric_name}_2'].min(), r2[f'{mean_metric_name}_2'].max(),
                num=r2[f'{mean_metric_name}_2'].shape[0]
            ),
            cmap=color_scheme,
        )
        ax.errorbar(
            r2[x]['mean'],
            r2[y]['mean'],
            xerr=r2[x]['std'],
            yerr=r2[y]['std'],
            markersize=0.0,
            linestyle='None',
            ecolor="#b3bfd1",
            alpha=0.2,
        )
        p3d = ax.scatter(
            r2[x]['mean'],
            r2[y]['mean'],
            c=np.linspace(
                r2[f'{mean_metric_name}_2'].min(), r2[f'{mean_metric_name}_2'].max(),
                num=r2[f'{mean_metric_name}_2'].shape[0]
            ),
            cmap=color_scheme,

        )
        ax.scatter(
            r2[x]['mean'],
            r2[y]['mean'],
            c=np.linspace(
                r2[f'{mean_metric_name}_2'].min(), r2[f'{mean_metric_name}_2'].max(),
                num=r2[f'{mean_metric_name}_2'].shape[0]
            ),
            cmap=color_scheme,
        )
    else:
        p3d = ax.scatter(
            r2[x], r2[y],
            c=np.linspace(
                r2[f'{mean_metric_name}_2'].min(), r2[f'{mean_metric_name}_2'].max(),
                num=r2[f'{mean_metric_name}_2'].shape[0]
            ),
            cmap=color_scheme
        )
    labels = sorted({f'{label:.1f}' for label in r2[f'{mean_metric_name}_2']})
    handles, _ = p3d.legend_elements(prop="sizes", num=len(labels) + 1, alpha=0.6)
    ax.set_xlabel(f'mean {x}')
    ax.set_ylabel(f'mean {y}')
    ax.set_title(f'{x} and {y} as a function of mean binned {mean_metric_name}', loc='center', wrap=True)
    fig.colorbar(p3d, pad=0.05, location='right', label=mean_metric_name)
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}_{name_addition}mean_{x}_vs_mean_{y}_by_{mean_metric_name}.png")
    plt.show()


def plot_two_metrices_vs_third_means_metrics_colors(
        metric_names: List[str],
        mean_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        color_scheme: str,
        task: str = '',
        name_addition: str = '',
        local_base_path: Optional[str] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    x, y = metric_names
    p3d = ax.scatter(
        result_df[x], result_df[y],
        c=np.linspace(
            result_df[mean_metric_name].min(), result_df[mean_metric_name].max(),
            num=result_df[mean_metric_name].shape[0]
        ),
        cmap=color_scheme
    )
    labels = sorted({f'{label:.1f}' for label in result_df[mean_metric_name]})
    handles, _ = p3d.legend_elements(prop="sizes", num=len(labels) + 1, alpha=0.6)
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(f'{x} and {y} as a function of {mean_metric_name}', loc='center', wrap=True)
    fig.colorbar(p3d, pad=0.05, location='right', label=mean_metric_name)
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}{name_addition}_{x}_vs_{y}_by_{mean_metric_name}.png")
    plt.show()


def plot_two_metrices_top_and_bottom_quentiles(
        metric_names: List[str],
        quantile_metric: str,
        result_df: pd.DataFrame,
        time_str: str,
        color_scheme: List[str],
        task: str = '',
        name_addition: str = '',
        title_addition: str = '',
        top_quantile: float = 0.9,
        bottom_quantile: float = 0.1,
        local_base_path: Optional[str] = None,
):
    top_results = result_df[result_df[quantile_metric] >= result_df[quantile_metric].quantile(top_quantile)]
    bottom_results = result_df[result_df[quantile_metric] <= result_df[quantile_metric].quantile(bottom_quantile)]
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    x, y = metric_names
    ax.scatter(
        top_results[x], top_results[y],
        c=color_scheme[0],
        label=f'top {top_quantile} quantile',
    )
    ax.scatter(
        bottom_results[x], bottom_results[y],
        c=color_scheme[-1],
        label=f'bottom {bottom_quantile} quantile',
    )
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.legend()
    ax.set_title(
        f"{x} and {y} for the top and bottom's quantiles of {quantile_metric}{title_addition}".replace('_', ' '),
        loc='center', wrap=True)
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}{name_addition}{x}_vs_{y}_top_bottom_quantiles_by_{quantile_metric}.png")
    plt.show()


def plot_two_metrices_top_and_bottom_quentiles_X4(
        metric_names: List[str],
        quantile_metric: str,
        all_results_df: pd.DataFrame,
        time_str: str,
        color_scheme: List[str],
        min_connectivity_range: float = 0.1,
        max_connectivity_range: float = 0.9,
        task: str = '',
        top_quantile: float = 0.9,
        bottom_quantile: float = 0.1,
        local_base_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 7))
    ax = []
    connectivity_range = np.linspace(min_connectivity_range, max_connectivity_range, 4, endpoint=True)
    x, y = metric_names
    x_label, y_label = ['\n'.join(wrap(l.replace('_', ' '), 20)) for l in metric_names]
    for i in range(4):
        min_connectivity = round(connectivity_range[i] - 0.1, 2)
        max_connectivity = round(connectivity_range[i] + 0.1, 2)
        result_df = all_results_df[all_results_df['connectivity_ratio'].between(min_connectivity, max_connectivity)]
        ax.append(fig.add_subplot(2, 2, i + 1))
        top_results = result_df[result_df[quantile_metric] >= result_df[quantile_metric].quantile(top_quantile)]
        bottom_results = result_df[result_df[quantile_metric] <= result_df[quantile_metric].quantile(bottom_quantile)]
        ax[i].scatter(
            top_results[x], top_results[y],
            c=color_scheme[0],
            label=f'top {top_quantile} quantile',
        )
        ax[i].scatter(
            bottom_results[x], bottom_results[y],
            c=color_scheme[-1],
            label=f'bottom {bottom_quantile} quantile',
        )
        ax[i].set_title(
            f"connectivity between {min_connectivity}-{max_connectivity}".replace('_', ' '),
            loc='center',
            wrap=True,
        )

        ax[i].set_xlabel(x_label, wrap=True)
        ax[i].set_ylabel(y_label, wrap=True)
    fig.suptitle(
        f"{x} and {y} for the top and bottom's quantiles of {quantile_metric}".replace('_', ' '),
        wrap=True,
    )
    plt.tight_layout(pad=1)
    ax[3].legend()
    plt.savefig(
        f"{local_base_path}/plots/{task}_teachbility_plots/{time_str}_{task}_4_connectivities_{x}_vs_{y}_top_bottom_quantiles_by_{quantile_metric}.png")
    plt.show()


def plot_loss_and_r2s_for_selected_feature_numbers(
        desiered_num_features: List[str],
        all_results_dict,
        cut: int,
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        lr: float,
        on_end: bool = False,
        regressor: str = 'Lightgbm',
        model: str = 'an ANN',
        sufix: str = ''
):
    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10))
    for i, num_features in enumerate(desiered_num_features):
        if on_end:
            zommed_all_results = all_results_dict[num_features][-cut:]
        else:
            zommed_all_results = all_results_dict[num_features][:cut]
        ax1.plot(zommed_all_results['Epoch'], zommed_all_results['log loss'], c=COLORS_7[i],
                 label=num_features, )
        ax2.plot(zommed_all_results['Epoch'], zommed_all_results['r2s test'], c=COLORS_7[i],
                 label=num_features, )
        ax3.plot(zommed_all_results['Epoch'], zommed_all_results['r2s train'], c=COLORS_7[i],
                 label=num_features, )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('log loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R2 - test')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R2 - train')
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0 + box1.height * 0.15,
                      box1.width, box1.height * 0.95])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0 + box2.height * 0.15,
                      box2.width, box2.height * 0.95])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0 + box3.height * 0.15,
                      box3.width, box3.height * 0.95])

    # Put a legend below current axis
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=int(np.ceil(len(desiered_num_features) / 2)), title='#features')
    ax1.set_title(
        f"Performance over time of {model} predicting the mean performance of an architecture using selected "
        f"number of features, {task}, {regressor}",
        loc='center',
        wrap=True
    )
    plt.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_top_regression_feature_selection_performance_lr_{lr}_{cut}{sufix}.png")

    plt.show()


def plot_all_num_features_models(
        x_label: str,
        y_label: str,
        all_results_dict: dict,
        cut: int,
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        lr: float,
        regressor: str = 'Lightgbm',
):
    plt.figure()
    for i, num_features in enumerate(sorted(all_results_dict)):
        plt.plot(all_results_dict[num_features][x_label][:cut], all_results_dict[num_features][y_label][:cut],
                 c=COLORS_20[i], label=num_features, )
    plt.xlabel(x_label)
    y_label = y_label.replace('r2s', 'R2')
    plt.ylabel(y_label)
    plt.legend(title='#features')
    plt.title(
        f"{y_label} performance over time of an ANN predicting the mean performance of an architecture using "
        f"selected number of features, {task}, {regressor}",
        loc='center',
        wrap=True
    )
    plt.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_{y_label}_regression_feature_selection_performance_lr_{lr}.png")
    plt.show()


def plot_num_features_vs_r2(
        res_df: pd.DataFrame,
        r2_param: str,
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        regressor: str = 'Lightgbm',
        additional_data_txt: str = '',
):
    plt.plot(res_df['num_features'], res_df[f'{r2_param}_train_r2'], label='train', c=COLORS_7[0])
    plt.plot(res_df['num_features'], res_df[f'{r2_param}_test_r2'], label='test', c=COLORS_7[1])
    plt.xlabel('num features')
    plt.ylabel(f'{r2_param} R2 score')
    title = f"{task} - predictive ANN {r2_param} R2 score for predicting architectures mean performance as a function " \
            f"of the number of features used"
    if len(additional_data_txt) > 0:
        title = f"The ANN's R2 score for predicting architectures mean performance as a function of the number of " \
                f"features used. The {r2_param} over {additional_data_txt.replace('_', ' ')}, {task}, {regressor}"
    plt.title(title, wrap=True, )
    plt.legend()
    plt.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_{r2_param}_r2_vs_num_features_ann_regression_feature_selection_final_{additional_data_txt}.png")
    plt.show()


def plot_bar_plot_of_used_features(
        sum_uses: pd.Series,
        cut: int,
        is_top: bool,
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        regressor: str = 'Lightgbm',
        jump: List[float] = [2.0, 1.0],
        is_normalized: bool = False,
):
    if is_top:
        sum_uses = sum_uses[:cut]
        top_bottom = 'top'
        c = COLORS_7[0]
        title_intro = 'most'
        jump = jump[0]
    else:
        sum_uses = sum_uses[-cut:]
        top_bottom = 'bottom'
        c = COLORS_7[1]
        title_intro = 'least'
        jump = jump[1]
    y_label = 'sum of uses'
    normalized = ''
    if is_normalized:
        y_label = 'normalized sum of uses'
        normalized = '_normalized'
    plt.figure(figsize=(15, 12))
    labels = [ind.replace('_', ' ') for ind in sum_uses.index]
    plt.bar(labels, sum_uses, color=c)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(np.arange(0, sum_uses.max() + jump / 2, jump))
    plt.title(f'{title_intro.capitalize()} common features chosen by {regressor}, {task} task')
    plt.ylabel(y_label)
    plt.xlabel('feature name')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_{top_bottom}{normalized}_used_features.png")
    plt.show()


def plot_r2_vs_num_features(
        res_df: pd.DataFrame,
        plots_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        regressor: str = 'Lightgbm',
):
    plt.plot(res_df['num_features'], res_df['train_r2'], label='train', c=COLORS[0])
    plt.plot(res_df['num_features'], res_df['test_r2'], label='test', c=COLORS[1])
    plt.xlabel('num features')
    plt.ylabel('R2 score')
    plt.title(
        f"{regressor}'s R2 score for predicting architectures mean performance as a function of the "
        f"number of features used, {task}",
        wrap=True,
    )
    plt.legend()
    plt.savefig(
        f"{local_base_path}/plots/{plots_folder}/{time_str}_{task}_r2_vs_num_features_{regressor}_regression_feature_selection.png")
    plt.show()


def plot_double_bar_plot_of_used_features_different_algos(
        sum_uses: pd.DataFrame,
        cut: int,
        is_top: bool,
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str,
        jump: List[float] = [2.0, 1.0],
        is_normalized: bool = False,
):
    fig, ax = plt.subplots(figsize=(15, 12))
    width = 0.25
    multiplier = 0
    if is_top:
        sum_uses = sum_uses[:cut]
        top_bottom = 'top'
        c = [COLORS_7[0], COLORS_7[1], COLORS_7[3]]
        title_intro = 'Most'
        jump = jump[0]
    else:
        sum_uses = sum_uses[-cut:]
        top_bottom = 'bottom'
        c = [COLORS_7[-1], COLORS_7[-2], COLORS_7[-3]]
        title_intro = 'Least'
        jump = jump[1]
    y_label = 'sum of uses'
    normalized = ''
    if is_normalized:
        y_label = 'normalized sum of uses'
        normalized = 'normalized'
    max_val = 0
    x = np.arange(len(sum_uses))
    for i, algo in enumerate(sum_uses.columns):
        if algo == 'sum':
            continue
        max_val = sum_uses[algo].max() if (sum_uses[algo].max() > max_val) else max_val
        offset = width * multiplier
        rects = ax.bar(x + offset, sum_uses[algo], width, label=algo, color=c[i])
        multiplier += 1
    ax.legend(loc='lower left')
    labels = [ind.replace('_', ' ') for ind in sum_uses.index]
    ax.set_xticks(x + width, labels, rotation=30, ha='right')
    ax.set_yticks(np.arange(0, max_val + jump / 2, jump))
    ax.set_title(f'{title_intro} common features chosen by LightGBM and XGBoost regressors, {task} task')
    ax.set_ylabel(y_label)
    ax.set_xlabel('feature name')
    plt.tight_layout()
    fig.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_{top_bottom}{normalized}_used_features_LightGBM_and_XGBoost.png")
    plt.show()


def plot_mean_performance_bars_for_num_features(
        all_results_dict: pd.DataFrame,
        desiered_num_features: List[Union[str, int]],
        fig_out_folder: str,
        local_base_path: str,
        time_str: str,
        task: str = '',
        start_ind: int = 20,
        end_ind: int = 1000,
        model: str = 'an ANN',
        sufix: str = '',
):
    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
    mean_train_r2s = []
    mean_test_r2s = []
    mean_loss = []
    mean_train_r2s_err = []
    mean_test_r2s_err = []
    mean_loss_err = []
    for i, num_features in enumerate(desiered_num_features):
        zommed_all_results = all_results_dict[num_features][start_ind:end_ind]
        mean_loss.append(zommed_all_results['log loss'].mean())
        mean_test_r2s.append(zommed_all_results['r2s test'].mean())
        mean_train_r2s.append(zommed_all_results['r2s train'].mean())
        mean_loss_err.append(zommed_all_results['log loss'].std())
        mean_test_r2s_err.append(zommed_all_results['r2s test'].std())
        mean_train_r2s_err.append(zommed_all_results['r2s train'].std())
        ax1.barh(num_features, mean_loss[i], label=num_features, color=COLORS_7[i], xerr=mean_loss_err[i],
                 ecolor='grey')
        ax2.barh(num_features, mean_test_r2s[i], label=num_features, color=COLORS_7[i],
                 xerr=mean_test_r2s_err[i], ecolor='grey')
        ax3.barh(num_features, mean_train_r2s[i], label=num_features, color=COLORS_7[i],
                 xerr=mean_train_r2s_err[i], ecolor='grey')

    ax1.set_xlabel('mean log loss')
    ax1.set_yticks([])
    ax1.set_xlim(
        [min(mean_loss) - (0.5 * max(mean_loss_err)) - 0.2, max(mean_loss) + (0.5 * max(mean_loss_err)) + 0.2],
    )
    ax2.set_xlabel('mean R2 test')
    ax2.set_yticks([])
    ax2.set_xlim(
        [min(mean_test_r2s) - (0.5 * max(mean_test_r2s_err)) - 0.005,
         max(mean_test_r2s) + (0.5 * max(mean_test_r2s_err)) + 0.005],
    )
    ax3.set_xlabel('mean R2 train')
    ax3.set_yticks([])
    ax3.set_xlim(
        [min(mean_train_r2s) - (0.5 * max(mean_train_r2s_err)) - 0.001,
         max(mean_train_r2s) + (0.5 * max(mean_train_r2s_err)) + 0.001],
    )
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0,
                      box1.width * 0.8, box1.height])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0,
                      box2.width * 0.8, box2.height])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0,
                      box3.width * 0.8, box3.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='#features')
    ax1.set_title(
        f"Mean Performance over time of {model} predicting the mean\nperformance of an architecture using selected "
        f"number of features, {task}",
        loc='center',
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(
        f"{local_base_path}/plots/{fig_out_folder}/{time_str}_{task}_top_mean_preformence_of_model_over_{start_ind}_{end_ind}{sufix}.png"
        , bbox_inches='tight'
    )
    plt.show()


def plot_performance_vs_target_performance_bar_plot(
        res_df: pd.DataFrame,
        plots_path: str,
        num_features: int,
        time_str: str,
        num_stds: Optional[int] = None,
):
    required_performance_max = float(res_df['required_performance_max'].iloc[0])
    required_performance_min = float(res_df['required_performance_min'].iloc[0])
    required_performance_diff = required_performance_max - required_performance_min

    fig = plt.figure()
    ax = fig.add_subplot(111, )
    n, bins, patches = ax.hist(
        x=res_df['mean_performance'].astype(float),
        bins=np.arange(
            required_performance_min - (7 * required_performance_diff),
            required_performance_max + (8 * required_performance_diff),
            required_performance_diff,
        ),
        color='#4F6272',
    )
    patches[7].set_facecolor('#B7C3F3')
    h = [Patch(facecolor='#B7C3F3', label='Color Patch'), patches]
    ax.legend(h, ['target bin', 'predictions', ])
    plt.xlabel('predicted mean performance')
    str_addition = ''
    if num_stds:
        str_addition = f'(distance from mean < {num_stds} stds) '
    plt.title(
        f'Actual mean performance of architectures with structural features drown from a multivariate gaussian '
        f'distribution {str_addition}of the target performance data (target {round(required_performance_min, 4)}-'
        f'{round(required_performance_max, 4)})',
        wrap=True,
    )
    plt.savefig(
        f'{plots_path}/{time_str}_predicted_mean_performance_of_arch_from_multi_gaussian_'
        f'{num_features}_features_{num_stds}_stds.png')
    plt.show()


def plot_two_metrics(
        x_metric_name: str,
        y_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        task: str = '',
        color_ind: int = 0,
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        mark_size: float = 2.0,
        gitter: bool = False
):
    x_data = result_df[x_metric_name]
    if gitter:
        x_data = x_data + np.random.normal(-0.5, 0.5, x_data.shape[0])
    plt.scatter(x_data, result_df[y_metric_name], c=COLORS[color_ind], s=mark_size)
    plt.xlabel(f"{x_metric_name.replace('_', ' ')}")
    plt.ylabel(y_metric_name.replace('_', ' '))
    plt.title(f"{x_metric_name.replace('_', ' ')} of an architecture and its"
              f" {y_metric_name.replace('_', ' ')}, {title_addition}{task}", loc='center', wrap=True)
    plt.savefig(
        f"{plot_path}/{time_str}_{task}{name_addition}_{y_metric_name}_vs_{x_metric_name}.png")
    plt.show()


def plot_num_uses_global_local(
        num_uses: List[str],
        features: List[str],
        bar_colors: List[str],
        bar_labels: List[str],
        time_str: str,
        title_sufix: str = 'all tasks and feature selection methods',
        local_base_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 10))
    plt.bar(features, num_uses, color=bar_colors)
    cmap = dict(zip(bar_labels, bar_colors))
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]
    leg = []
    for l in bar_labels:
        if l not in leg:
            leg.append(l)
    plt.legend(labels=leg, handles=patches, )
    plt.xticks(rotation=30, ha='right')
    plt.title(f'Feature frequency for 5 features reduction over {title_sufix}', wrap=True)
    if local_base_path:
        plt.savefig(
            f"{local_base_path}/plots/top_features/{time_str}_top_feature_analsis_{title_sufix}.png")


def plot_two_metrics_vs_colored_metric(
        x_metric_name: str,
        y_metric_name: str,
        colored_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        task: str = '',
        color_name: str = 'Blues',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        mark_size: float = 2.0,
):
    plt.scatter(result_df[x_metric_name], result_df[y_metric_name],
                s=mark_size,

                c=result_df[colored_metric_name],
                cmap=color_name,
                )
    plt.colorbar(pad=0.05, location='right', label=colored_metric_name.replace('_', ' '))
    plt.xlabel(x_metric_name.replace('_', ' '))
    plt.ylabel(y_metric_name.replace('_', ' '))
    plt.title(
        f"Architectures {x_metric_name.replace('_', ' ')} vs {y_metric_name.replace('_', ' ')} colored by "
        f"{colored_metric_name.replace('_', ' ')}, {title_addition}{task}",
        wrap=True)
    plt.savefig(
        f"{plot_path}/{time_str}_{task}{name_addition}_{x_metric_name}_vs_{y_metric_name}_colored_by_{colored_metric_name}.png")
    plt.show()


def plot_two_metrics_with_mean(
        x_metric_name: str,
        y_metric_name: str,
        result_df: pd.DataFrame,
        time_str: str,
        task: str = '',
        color_ind: int = 0,
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        mark_size: float = 2.0,
):
    x_data = result_df[x_metric_name]
    means_y = []
    mean_x = []
    for x in sorted(result_df[x_metric_name].unique()):
        means_y.append(result_df[y_metric_name][x_data == x].mean())
        mean_x.append(x)
    plt.scatter(x_data, result_df[y_metric_name], c=COLORS[color_ind], s=mark_size)
    plt.plot(mean_x, means_y, c=rads[0], label='mean')
    plt.xlabel(f"{x_metric_name.replace('_', ' ')}")
    plt.ylabel(y_metric_name.replace('_', ' '))
    plt.legend()
    plt.title(f"{x_metric_name.replace('_', ' ')} of an architecture and its"
              f" {y_metric_name.replace('_', ' ')}, {title_addition}{task}", loc='center', wrap=True)
    plt.savefig(
        f"{plot_path}/{time_str}_{task}{name_addition}_{y_metric_name}_vs_{x_metric_name}.png")
    plt.show()


def plot_two_metrics_with_mean_multi_ep(
        x_metric_name: str,
        y_metric_name: str,
        all_result_df: List[pd.DataFrame],
        epochs: List[int],
        dims: list[int],
        time_str: str,
        task: str = '',
        color_ind: int = 0,
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        mark_size: float = 2.0,
):
    num_columns = int(np.ceil(len(epochs) / 2))
    fig, ax = plt.subplots(num_columns, 2, figsize=(8, 12), )
    idxs = []
    for i in range(num_columns):
        idxs.append((i, 0))
        idxs.append((i, 1))
    for result_df, ep, (i, j) in zip(all_result_df, epochs, idxs):
        x_data = result_df[x_metric_name]
        means_y = []
        mean_x = []
        for x in sorted(result_df[x_metric_name].unique()):
            means_y.append(result_df[y_metric_name][x_data == x].mean())
            mean_x.append(x)
        ax[i, j].scatter(x_data, result_df[y_metric_name], c=COLORS[color_ind], s=mark_size)
        ax[i, j].plot(mean_x, means_y, c=rads[0], label='mean')
        if j == 0:
            ax[i, j].set_ylabel(y_metric_name.replace('_', ' '))
        if i == num_columns - 1:
            ax[i, j].set_xlabel(f"{x_metric_name.replace('_', ' ')}")
        ax[i, j].set_title(f'{ep} epochs')
        ax[i, j].legend()
    if num_columns * 2 > len(epochs):
        ax[num_columns - 1, 1].set_xlabel(f"{x_metric_name.replace('_', ' ')}")
    plt.suptitle(f"{x_metric_name.replace('_', ' ')} of an architecture and its"
                 f" {y_metric_name.replace('_', ' ')}, {dims} {title_addition}{task}", wrap=True)
    plt.tight_layout()
    plt.savefig(
        f"{plot_path}/{time_str}_{task}{name_addition}_{y_metric_name}_vs_{x_metric_name}.png")
    plt.show()


def plot_two_ann_metrics_begining_end(
        x_metric_name: str,
        y_metric_name: str,
        all_result_list: List[pd.DataFrame],
        epochs: List[int],
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        time_split_ind: int = 20,
        jump: int = 20,
        legend_title: str = 'db num epochs'
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    for i, (name, res) in enumerate(zip(epochs, all_result_list)):
        ax1.plot(res[x_metric_name][:time_split_ind], res[y_metric_name][:time_split_ind], label=f'{name}',
                 c=COLORS_12[i])
        ax2.plot(res[x_metric_name][time_split_ind::jump], res[y_metric_name][time_split_ind::jump], label=f'{name}',
                 c=COLORS_12[i])

    ax1.set_title(f'ANN {y_metric_name.replace("_", " ")} results - initial epohcs, {title_addition}, {dims}, {task}')
    ax2.set_title(f'ANN {y_metric_name.replace("_", " ")} - late epohcs, {title_addition}, {dims}, {task}')
    ax1.set_xlabel(x_metric_name)
    ax2.set_xlabel(x_metric_name)
    ax1.set_ylabel(y_metric_name)
    ax2.set_ylabel(y_metric_name)
    ax1.set_xticks(np.arange(time_split_ind)[::5])
    box1 = ax1.get_position()
    box2 = ax2.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.95, box1.height * 0.95, ])
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height * 0.95, ])
    # Put a legend to the right of the current axis
    ax1.legend(loc='upper center', ncol=1, fancybox=True, bbox_to_anchor=(1.1, 0.5), title=legend_title)
    plt.savefig(
        f"{plot_path}/{time_str}_{y_metric_name}_ANN{name_addition}.png")
    plt.show()


def plot_ann_metric_train_test(
        x_metric_name: str,
        y_metric_name: str,
        all_result_list: List[pd.DataFrame],
        epochs: List[int],
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        start_ind: int = 100,
        jump: int = 100,
        legend_title: str = 'db num epochs'
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    for i, (name, res) in enumerate(zip(epochs, all_result_list)):
        ax1.plot(res['Epoch'][start_ind::jump], res[f'{y_metric_name} train'][start_ind::jump], label=f'{name}',
                 c=COLORS_12[i])
        ax2.plot(res['Epoch'][start_ind::jump], res[f'{y_metric_name} test'][start_ind::jump], label=f'{name}',
                 c=COLORS_12[i])
    ax1.set_title(f'ANN R2 - train, {title_addition}, {dims}, {task}')
    ax2.set_title(f'ANN R2 - test, {title_addition}, {dims}, {task}')
    ax1.set_xlabel(f'{x_metric_name}')
    ax2.set_xlabel(f'{x_metric_name}')
    ax1.set_ylabel(f'{y_metric_name} train')
    ax2.set_ylabel(f'{y_metric_name} test')
    box1 = ax1.get_position()
    box2 = ax2.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.95, box1.height * 0.95, ])
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height * 0.95, ])

    # Put a legend to the right of the current axis
    ax1.legend(loc='upper center', ncol=1, fancybox=True, bbox_to_anchor=(1.1, 0.5), title=legend_title)
    plt.savefig(
        f"{plot_path}/{time_str}_r2_ANN{name_addition}.png")
    plt.show()


def plot_two_metrics_by_ephoc(
        x_metric_name: str,
        y_metric_name: str,
        all_result_df: List[pd.DataFrame],
        epochs: List[int],
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        legend_title: int = 'db num epochs'
):
    for i, (result_df, ep), in enumerate(zip(all_result_df, epochs)):
        x_data = result_df[x_metric_name]
        means_y = []
        mean_x = []
        for x in sorted(result_df[x_metric_name].unique()):
            means_y.append(result_df[y_metric_name][x_data == x].mean())
            mean_x.append(x)
        plt.plot(mean_x, means_y, c=COLORS_12[i], label=ep)
    plt.legend(title=legend_title)
    plt.ylabel(y_metric_name.replace('_', ' '))
    plt.xlabel(f"{x_metric_name.replace('_', ' ')}")
    plt.title(f"{x_metric_name.replace('_', ' ')} of an architecture and its"
              f" mean {y_metric_name.replace('_', ' ')}, {dims} {title_addition}{task}", wrap=True)
    plt.savefig(
        f"{plot_path}/{time_str}_{y_metric_name}_by_ephoc{name_addition}.png")
    plt.show()


def plot_mean_r2_bars(
        metric_name: str,
        all_result_list: List[pd.DataFrame],
        epochs: List[int],
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        start_ind: int = 100,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    test_means = [
        res[f'{metric_name} test'][start_ind:].mean()
        for res in all_result_list
    ]
    train_means = [
        res[f'{metric_name} train'][start_ind:].mean()
        for res in all_result_list
    ]
    bar_size = 0.35
    y_locs = np.arange(len(epochs))
    for i in range(len(epochs)):
        ax1.barh(y_locs[i], test_means[i], height=bar_size, color=COLORS_12[1],
                 label=epochs[i])

        ax2.barh(y_locs[i], train_means[i], height=bar_size, color=COLORS_12[1],
                 label=epochs[i])

    ax1.set(
        yticks=y_locs,
        yticklabels=[str(e) for e in epochs],
        xlim=[min(test_means) - 0.09, max(test_means) + 0.03],
    )
    ax1.set_xlabel('Test mean r2 score')
    ax1.set_title('Test mean r2')
    ax1.set_ylabel('db num epochs')
    ax2.set(
        yticks=y_locs,
        yticklabels=[str(e) for e in epochs],
        xlim=[min(train_means) - 0.09, max(train_means) + 0.03],
    )
    ax2.set_title('Train mean r2')
    ax2.set_xlabel('Train mean r2 score')
    ax2.set_ylabel('db num epochs')
    plt.suptitle(
        f'{task} - Prediction of teachability mean performance (after convergence) {title_addition}{dims}, {task}',
        wrap=True)
    plt.tight_layout()
    if plot_path:
        plt.savefig(
            f"{plot_path}/{time_str}_mean_{metric_name}_by_ephoc{name_addition}.png")
    plt.show()


def plot_hist_of_performances_by_ephoc(
        metric_name: str,
        all_result_df: List[pd.DataFrame],
        epochs: List[int],
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        bins: int = 20,
):
    num_columns = int(np.ceil(len(epochs) / 2))
    fig, axs = plt.subplots(num_columns, 2, figsize=(8, 12), sharey='all', )
    min_x = 1.0
    max_x = 0
    idxs = []
    r, c, = axs.shape
    for i in range(r):
        for j in range(c):
            idxs.append((i, j))
    for result_df, ep, (i, j) in zip(all_result_df, epochs, idxs):
        axs[i, j].hist(result_df[metric_name], bins=bins, alpha=0.7, color=COLORS[2], )
        axs[i, j].set_title(f'{ep} epochs')
        if min_x > result_df[metric_name].min():
            min_x = result_df[metric_name].min()
        if max_x < result_df[metric_name].max():
            max_x = result_df[metric_name].max()

    for i in range(r):
        axs[i, 0].set_ylabel('num archs')
    for j in range(c):
        axs[r - 1, j].set_xlabel(f"{metric_name.replace('_', ' ')}")
    for (i, j) in idxs:
        axs[i, j].set_xlim([min_x - 0.05, max_x + 0.05])
    plt.suptitle(f"Histogram of the {metric_name.replace('_', ' ')}s of architectures, {dims} {title_addition}{task}",
                 wrap=True)
    plt.tight_layout()
    if plot_path:
        plt.savefig(
            f"{plot_path}/{time_str}_hist_of_{metric_name}_by_ephoc{name_addition}.png")
    plt.show()


def plot_two_ann_metrics_train_test(
        x_metric_name: str,
        y_up_metric_name: str,
        y_down_metric_name: str,
        result: pd.DataFrame,
        dims: List[int],
        time_str: str,
        task: str = '',
        plot_path: Optional[str] = None,
        name_addition: str = '',
        title_addition: str = '',
        start_ind: int = 100,
        jump: int = 100,
        norm_losses: bool = True,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    for i, train_test in enumerate(['train', 'test']):
        if y_up_metric_name == 'losses' and norm_losses:
            y = result[f'{y_up_metric_name} {train_test}'][start_ind::jump] / (10 ** 6)
        else:
            y = result[f'{y_up_metric_name} {train_test}'][start_ind::jump]
        ax1.plot(result[x_metric_name][start_ind::jump], y, label=train_test,
                 c=COLORS[i])
        ax2.plot(result[x_metric_name][start_ind::jump], result[f'{y_down_metric_name} {train_test}'][start_ind::jump],
                 label=train_test,
                 c=COLORS[i])
    ax1.set_title(f'ANN - {y_up_metric_name}, {title_addition}, {dims}, {task}')
    ax2.set_title(f'ANN - {y_down_metric_name}, {title_addition}, {dims}, {task}')
    ax1.set_xlabel(f'{x_metric_name}')
    ax2.set_xlabel(f'{x_metric_name}')
    ax1.set_ylabel(f'{y_up_metric_name}')
    ax2.set_ylabel(f'{y_down_metric_name}')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(
        f"{plot_path}/{time_str}_{y_up_metric_name}_{y_down_metric_name}_train_test_ANN{name_addition}.png")
    plt.show()
