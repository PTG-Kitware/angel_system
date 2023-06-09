import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve, accuracy_score
import pandas as pd
import json
import yaml
import os
import scipy
import time

from angel_system.data.common.load_data import time_from_name
from angel_system.data.common.load_data import activities_from_dive_csv

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


def accuracy_from_true_false_examples(true_example_scores,
                                      false_example_scores):
    s = np.hstack([true_example_scores, false_example_scores]).T
    y_tue = np.hstack([np.ones(len(true_example_scores), dtype=bool),
                       np.zeros(len(false_example_scores), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    return accuracy_score(y_tue, s)


def plot_precision_recall(true_example_scores, false_example_scores,
                          label=None, title_str=None):
    fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
    plt.rc('font', **{'size': 22})
    plt.rc('axes', linewidth=4)

    s = np.hstack([true_example_scores, false_example_scores]).T
    y_tue = np.hstack([np.ones(len(true_example_scores), dtype=bool),
                       np.zeros(len(false_example_scores), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    precision, recall, thresholds = precision_recall_curve(y_tue, s)
    thresholds = np.hstack([thresholds[0], thresholds])
    auc = -np.trapz(precision, recall)

    plt.plot(recall, precision, linewidth=6, label=label)

    plt.xlabel('Recall', fontsize=40)
    plt.ylabel('Precision', fontsize=40)

    if title_str is not None:
        plt.title(title_str, fontsize=40)

    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    fig.tight_layout()

    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="y",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="y",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='y', which='both')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="x",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="x",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='x', which='both')

    if label is not None:
        plt.legend(fontsize=20, loc=0)
