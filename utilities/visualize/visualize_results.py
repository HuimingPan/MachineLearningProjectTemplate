import matplotlib.pyplot as plt
import numpy as np
from utilities.writer import save_unique_figure
from utilities.statistics.tests import anova
from utilities.visualize import adjust_plotting
from utilities.statistics.metrics import rmse_and_cc


def plot_fitting_curve(pred, target, **kwargs):
    keywords = {"figsize": (5.6, 4.2),
                "xlabel": "Time(s)",
                "ylabel": "Normalized Force(MVC)",
                "width": 0.6,
                "colors": ['#d47264', '#2066a8'],
                "legend": True,
                "xticks": np.arange(0, len(target), 1000),
                "xticklabels": np.arange(0, len(target), 1000) / 200,
                "xlim": [0, len(target)],
                "ylim": [0, 1],
                "rotation": 0,
                "location": "upper left",
                }
    keywords.update(kwargs)

    fig, ax = plt.subplots(figsize=keywords["figsize"])

    ax.plot(target,  label='Actual force', color=keywords["colors"][0])
    ax.plot(np.arange(0, len(pred), 5), pred[::5], linestyle=':', label='Estimated force',
            color=keywords["colors"][1], )
    # ax.plot(pred, linestyle=':', label='Estimated force',
    #         color=keywords["colors"][1], )

    rmse, cc = rmse_and_cc(pred, target)
    print(f"RMSE: {rmse}, CC: {cc}")
    adjust_plotting(ax, **keywords)
    plt.show()
    save_path = (f"E:/OneDrive - sjtu.edu.cn/Papers/EMG-based fatigue force estimation/figures/Results/"
                 f"fitting_curve.png")
    save_unique_figure(fig, save_path)
    return fig


def plot_performance(label, baseline_estimate, adver_estimate):
    fig, ax = plt.subplots()
    ax.plot(label, linestyle='--', label='Actual force', color="#d47264")
    ax.plot(baseline_estimate, linestyle=':', label='Baseline force', color="#2066a8")
    ax.plot(adver_estimate, linestyle='-.', label='Adversarial force', color="#8ec1da")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Force/MVC")
    ax.set_xticks(np.arange(0, len(label), 1000))
    ax.set_xticklabels(np.arange(0, len(label), 1000) / 200)

    ax.legend()
    plt.show()
    return fig


def plot_error_target(error, target, **kwargs):
    """
    Plot the error-target bar plot
    :param error:
    :param target:
    :param bins:
    :return:
    """
    keywords = {"figsize": (5.6, 4.2),
                "width": 1,
                "colors": "#2066a8",
                "bins": 10,
                "xlabel": "Force Range (%MVC)",
                "ylabel": "Error (%MVC)"
                }
    keywords.update(kwargs)
    fig, ax = plt.subplots(figsize=keywords["figsize"])

    edge_bins = np.linspace(0, 1, keywords["bins"] + 1)
    indices = np.digitize(target, edge_bins)
    error_bin = [error[indices == i] for i in range(1, len(edge_bins))]
    mean = [np.mean(bin) for bin in error_bin]
    std = [np.std(bin) for bin in error_bin]
    yerr = [np.zeros_like(std), std]

    print(anova(error_bin))

    error_kw = dict(linestyle='--', label='Error', color="#d47264")
    ax.bar(np.arange(1, len(edge_bins)), mean, keywords["width"], color=keywords["colors"],
           yerr=yerr, error_kw=error_kw, edgecolor="black", zorder=2)
    ax.set_xlabel(keywords["xlabel"])
    ax.set_ylabel(keywords["ylabel"])
    plt.show()
    return fig
