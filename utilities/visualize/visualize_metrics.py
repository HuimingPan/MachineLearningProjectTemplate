import matplotlib.pyplot as plt
import numpy as np
from utilities.statistics.tests import ttest_rmse, ttest_rel
from utilities.writer import save_unique_figure
from utilities.visualize import adjust_plotting

def plot_rmse(df, domain="f", **kwargs):
    """
    Plot the rmse
    :param df:
    :return:
    """

    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Subject",
                "ylabel": "RMSE(MVC)",
                "location": "upper center",
                "width": 0.2,
                "rotation": 0.,
                "location": "lower left",
                      }
    keywords.update(kwargs) if kwargs else keywords

    if domain not in ['nf', 'f', 'both']:
        raise ValueError("Domain should be one of ['nf', 'f', 'both']")

    if domain == 'both':
        meanX = np.array([-1.8, -0.6, 0.6, 1.8]) * keywords["width"] + df.shape[1]
    elif domain == 'nf':
        df = df.loc[["baseline-NF", "proposed-NF"]]
        meanX = meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]
    elif domain == 'f':
        df = df.loc[["baseline-F", "proposed-F"]]
        meanX = np.array([-0.6, 0.6]) * keywords["width"] + df.shape[1]


    fig, ax = plt.subplots(figsize=keywords["figsize"])
    rows = df.index
    colors = ['#2066a8', '#ae282c', '#cde1ec', '#f6d6c2', "#ededed"]
    marker = ['o', 's', 'v', '^', 'D']
    means = df.mean(axis=1)
    stds = df.std(axis=1)

    print("Means: ", means)
    print("Stds: ", stds)

    for i, row in enumerate(rows):
        ax.plot(df.loc[row], label=row.split("-")[0], color=colors[i], marker=marker[i], fillstyle='none')

    add_means_bar(ax, meanX, means, stds, colors, marker)
    p_value = ttest_rel(df.loc["baseline-F"], df.loc["proposed-F"])
    add_p_value_annotation(ax, meanX[0], meanX[1], 0.32, p_value)

    ticks = [f"S{i}" for i in range(1, df.shape[1] + 1)]
    adjust_plotting(ax, ticks + ["Mean"], **keywords)
    save_path = r"E:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\figures\Results\rmse.png"
    save_unique_figure(fig, save_path)
    return fig


def plot_rmse_bars_ws(df, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Window Size",
                "ylabel": "RMSE (MVC)",
                "width": 0.6,
                "colors": ['#2066a8', '#8ec1da', '#cde1ec', '#f6d6c2', '#d47264', '#ae282c'],
                "annotate_base": 0.080,
                "annotate_offset": 0.005,
                "legend": False,
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rows = df.index

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print([f"{level}: {mean} ± {std}" for level, mean, std in zip(rows, means, stds)])

    meanX = np.arange(len(rows))
    yerr = [np.zeros_like(stds), stds]
    error_kw = dict(lw=1, capsize=3, capthick=1, zorder=1)
    ax.bar(meanX, means, keywords["width"], color=keywords["colors"],
           yerr=yerr, error_kw=error_kw, edgecolor=keywords["colors"], zorder=2)
    ttest = ttest_rmse(df, 512)
    ys = np.array([5, 3, 2, 0, 1, 4]) * keywords["annotate_offset"] + keywords["annotate_base"]
    for i, row in enumerate(rows):
        if row != 512:
            add_p_value_annotation(ax, meanX[i], meanX[3], ys[i], ttest[row].pvalue)

    adjust_plotting(ax, rows, **keywords)
    save_path = r"E:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\figures\Results\rmse_across_ws.png"
    save_unique_figure(fig, save_path)
    return fig


def plot_rmse_bars(df, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Force Level (MVC)",
                "ylabel": "RMSE (MVC)",
                "width": 0.6,
                "colors": ['#2066a8', '#8ec1da', '#cde1ec', '#f6d6c2', '#d47264', '#ae282c'],
                "annotate_base": 0.190,
                "annotate_offset": 0.008,
                "legend": False,
                "rotation": 0,
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rows = df.index

    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print([f"{level}: {mean} ± {std}" for level, mean, std in zip(rows, means, stds)])

    meanX = np.arange(len(rows))
    yerr = [np.zeros_like(stds), stds]
    error_kw = dict(lw=1, capsize=3, capthick=1, zorder=1)
    ax.bar(meanX, means, keywords["width"], color=keywords["colors"],
           yerr=yerr, error_kw=error_kw, edgecolor=keywords["colors"], zorder=2)

    ttest1_pvalue = ttest_rel(df.loc[rows[0]], df.loc[rows[1]])
    ttest2_pvalue = ttest_rel(df.loc[rows[0]], df.loc[rows[2]])
    ttest3_pvalue = ttest_rel(df.loc[rows[1]], df.loc[rows[2]])
    ys = np.array([2, 3, 1]) * keywords["annotate_offset"] + keywords["annotate_base"]
    add_p_value_annotation(ax, meanX[0], meanX[1], ys[0], ttest1_pvalue)
    add_p_value_annotation(ax, meanX[0], meanX[2], ys[1], ttest2_pvalue)
    add_p_value_annotation(ax, meanX[1], meanX[2], ys[2], ttest3_pvalue)

    adjust_plotting(ax, rows, **keywords)
    plt.show()
    save_path = r"E:\OneDrive - sjtu.edu.cn\Papers\EMG-based fatigue force estimation\figures\Results\rmse_across_levels.png"
    save_unique_figure(fig, save_path)
    return fig


def add_p_value_annotation(ax, x1, x2, y, p_val):
    bar_height = y
    bar_line_height = bar_height * 1.02
    ax.plot([x1, x1], [bar_height, bar_line_height], c='black')
    ax.plot([x2, x2], [bar_height, bar_line_height], c='black')
    ax.plot([x1, x2], [bar_line_height, bar_line_height], color="black", linestyle='-', linewidth=1)
    if p_val < 0.01:
        significance = "**"
    elif p_val < 0.05:
        significance = "*"
    else:
        significance = "ns"
        ax.text((x1 + x2) * .5, bar_line_height * 1.0001, significance, ha='center', va='bottom', color='black')
        return
    ax.text((x1 + x2) * .5, bar_line_height, significance, ha='center', va='center', color='black')


def add_means_bar(ax, X, y, yerr, colors, markers):
    width = 0.2
    yerr = [np.zeros_like(yerr), yerr]
    error_kw = dict(lw=1, capsize=3, capthick=1, zorder=1)
    ax.bar(X, y, width, color=colors, yerr=yerr, error_kw=error_kw, edgecolor=colors, zorder=2)
    for x, marker in zip(X, markers):
        ax.plot(x, 0.015, color='w', marker=marker, fillstyle='none')
    return ax


