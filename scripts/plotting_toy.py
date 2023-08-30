import numpy as np
import os
#  Annoying hack
import sys
sys.path.append("../src/")
sys.path.append("..")
import pickle
import seaborn as sns
import datetime
from itertools import cycle

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

import re
from copy import deepcopy

def get_cumulative_cost_mean_and_std(data, t_steps, repeats):
    out = {key: [] for key in data.keys()}

    for model in data.keys():
        for t in range(t_steps):
            tmp = []
            for ex in range(repeats):
                tmp.append(data[model][ex][t])
            tmp = np.vstack(tmp)
            # Calculate the cumulative sum here
            out[model].append(np.cumsum(tmp.mean(axis=0)))
    return out

def set_plotting():
    # Set plotting
    params = {
        'axes.labelsize': 25,
        'font.size': 20,
        'legend.fontsize': 24,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'text.usetex': False,
        'figure.figsize': [20, 12],
        'axes.labelpad': 10,
        'lines.linewidth': 10,
        'legend.loc': 'upper left'
    }
    rcParams['agg.path.chunksize'] = 10000
    rcParams.update(params)

def main():
    best_objective_values = [-2.]
    ###### ###### ###### ###### ###### ###### ###### ###### ######
    # In this part you need to hardcode directory and a string that uniquely indentifies results to grab
    directory = os.getcwd() + '/results/latest_toy'
    data = []

    for entry in os.scandir(directory):
        if (entry.path.endswith(".pickle")) and entry.is_file():
            with open(entry.path, "rb") as pickle_off:
                data.append(pickle.load(pickle_off))
    ###### ###### ###### ###### ###### ###### ###### ###### ######
    # From here it's MOSTLY automated
    methods_list = list(set.union(set([method for i in range(len(data)) for method in data[i].keys()])))
    from matplotlib import cm

    # num_wrong_graphs = sum('wrong' in method for method in data)
    linestyles = cycle(['-', '--', ':', '-.', '-', '--'])

    # best_objective_table
    allmin = []
    for i in range(len(data)):
        for method in methods_list:
            if method in data[i]:
                allmin.append(np.min(data[i][method][0][1][0]))
    globlmin = np.min(allmin)

    # Table data
    gaps = []
    for i in range(len(data)):
        gap_dict = {}
        for method in methods_list:
            if method in data[i]:
                H = len(data[i][method][0][1][0])
                if np.min(data[i][method][0][1][0]) > globlmin:
                    y_best = np.min(data[i][method][0][1][0])
                    H_best = np.argmin(data[i][method][0][1][0])
                else:
                    y_best = globlmin
                    H_best = H

                y_start = data[i][method][0][1][0][0]
                if np.isclose(y_start,globlmin):
                    gap_dict[method] = 1.
                else:
                    num = (y_best - y_start) / (globlmin - y_start) + ((H - H_best) / H)
                    denom = 1 + ((H-1) / H)
                    gap_dict[method] = num / denom

        gaps.append(gap_dict)


    gaps_avg = {method: [] for method in methods_list}
    gaps_stds = {method: [] for method in methods_list}
    for i in range(len(data)):
        for method in methods_list:
            if method in data[i]:
                gaps_avg[method].append(np.asarray(gaps[i][method]))

    for method in methods_list:
        gaps_avg[method] = np.vstack(gaps_avg[method])
        gaps_avg[method], gaps_stds[method] = gaps_avg[method].mean(0), gaps_avg[method].var(0)


    costs, outs, var_outs = {method: [] for method in methods_list}, {method: [] for method in methods_list}, {
        method: [] for method in methods_list}
    for i in range(len(data)):
        for method in methods_list:
            if method in data[i]:
                costs[method].append(np.cumsum(data[i][method][0][0][0]))
                outs[method].append(data[i][method][0][1][0])
    for method in methods_list:
        costs[method], outs[method] = np.vstack(costs[method]), np.vstack(outs[method])
        costs[method], outs[method], var_outs[method] = costs[method].mean(0), outs[method].mean(0), outs[method].var(0)

    # WIll  average results across wrong graphs
    wrong_graphs = [method for method in methods_list if "wrong" in method]

    costs_wrong, outs_wrong, var_outs_wrong, gaps_avg_wrong, gaps_std_avg_wrong = [], [], [], [], []

    for wg in wrong_graphs:
        costs_wrong.append(costs[wg])
        outs_wrong.append(outs[wg])
        var_outs_wrong.append(var_outs[wg])

        gaps_avg_wrong.append(gaps_avg[wg])
        gaps_std_avg_wrong.append(gaps_stds[wg])

    costs_wrong, outs_wrong, var_outs_wrong = np.vstack(costs_wrong).mean(0), np.vstack(outs_wrong).mean(0), np.vstack(
        var_outs_wrong).mean(0)

    gaps_avg_wrong, gaps_std_avg_wrong = np.vstack(gaps_avg_wrong).mean(0), np.vstack(gaps_std_avg_wrong).mean(0)


    methods_list = ['ceo', 'cbo_on_true_ei', "cbo_wrong_avg"]

    costs["cbo_wrong_avg"] = costs_wrong
    outs["cbo_wrong_avg"] = outs_wrong
    var_outs["cbo_wrong_avg"] = var_outs_wrong
    gaps_avg["cbo_wrong_avg"] = gaps_avg_wrong
    gaps_stds["cbo_wrong_avg"] = gaps_std_avg_wrong

    for method in methods_list:
        print(method, "avg gap", gaps_avg[method], " stderr ", np.sqrt(gaps_stds[method]) /  np.sqrt(len(data)))

    # plt.rc('text', usetex=True)
    # # plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
    # plt.rc('font', family='serif')
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
    plt.rcParams['font.size'] = 35
    plt.rcParams['legend.fontsize'] = 25
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    plt.rcParams['axes.labelpad'] = 0
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['lines.linewidth'] = 10

    cols = []
    color = iter(cm.rainbow(np.linspace(0, 1, len(methods_list) + 2)))  #

    data_fig = {}
    data_fig['best_objective_values'] = best_objective_values
    for method in methods_list:
        data_fig[method] = {}

    for method in methods_list:
        data_fig[method]["costs"] = costs[method]
        data_fig[method]["outs"] = outs[method]
        data_fig[method]["stdevs_outs"] = np.sqrt(var_outs[method])

    data_fig["num_replicates"] = len(data)

    for method in methods_list:
        col = next(color)
        cols.append(col)
        if method == 'ceo':
            label = 'CEO'
        elif method == 'cbo_wrong_avg':
            label = 'CBO with wrong graphs'
        else:
            label = 'CBO with true graph'

        plt.plot(costs[method], outs[method], label=label, c=col, ls=next(linestyles), lw=5)

    for i, method in enumerate(methods_list):
        facecol = deepcopy(cols[i])
        facecol[3] = 0.2
        edgecolor = deepcopy(cols[i])
        edgecolor[3] = 0.6
        plt.fill_between(costs[method], outs[method] - np.sqrt(var_outs[method]) / np.sqrt(len(data)),
                         outs[method] + np.sqrt(var_outs[method]) / np.sqrt(len(data)), edgecolor=edgecolor,
                         facecolor=facecol, linewidth=3)

    curr_color = next(color)
    plt.plot(best_objective_values * 30, c=curr_color, label='True optimum', lw=6)
    plt.xlim((0, 25.))
    plt.fill_between(range(0,len(costs['ceo']) * 2) , best_objective_values[0] - 0.05, best_objective_values[0] + 0.05, edgecolor=curr_color, facecolor=curr_color, linewidth=3 , alpha=0.2 )
    plt.legend()
    plt.xlabel('Cumulative intervention cost')
    plt.ylabel('Optimal value')
    plt.title('Synthetic (a)')
    plt.savefig('paper_SYN.pdf', format='pdf',tight_layout=True,bbox_inches='tight')
    plt.show()

    graphposts = [[] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[i]["ceo"][0][3])):
            graphposts[i].append(data[i]["ceo"][0][3][j][0])
    graphposts = np.vstack(graphposts)
    graphposts_mean, graphposts_std = graphposts.mean(0), np.sqrt(graphposts.var(0))
    curr_color = 'r'
    plt.plot(range(graphposts_mean.shape[0]), graphposts_mean,  c=curr_color)
    stdr = np.sqrt(len(data))
    plt.xlabel('Iteration')
    plt.ylabel('Probability mass on true graph')
    plt.title('Posterior convergence')
    plt.fill_between(range(graphposts_mean.shape[0]), graphposts_mean - 2 * graphposts_std / stdr ,graphposts_mean +  2 * graphposts_std / stdr , edgecolor=curr_color, facecolor=curr_color, linewidth=3 , alpha=0.2 )
    plt.show()
    # plt.savefig('paper_toy_posterior.pdf', format='pdf',tight_layout=True,bbox_inches='tight')

    # with open(
    #         "results/data_fig_health/data_toy.pickle",
    #         "wb",
    # ) as handle:
    #     #  We have to use dill because our object contains lambda functions.
    #     pickle.dump(data_fig, handle)
    #     handle.close()

if __name__ == "__main__":

    os.chdir("..")

    from pathlib import Path

    Path("./results/toy").mkdir(parents=True, exist_ok=True)

    main()

    print('done')


