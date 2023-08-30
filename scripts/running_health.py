import sys
import os
import pickle
import time

sys.path.append("../src/")
sys.path.append("..")

from src.examples.example_setups import setup_stat_scm_ceo_toy, setup_stat_scm_ceo_healthcare
from src.utils.sequential_sampling import sequentially_sample_model
from src.utils.sem_utils.toy_sems import StationaryDependentSEM as StatSEM, HealthcareSEM
from src.utils.plotting import plot_outcome
from src.utils.sem_utils.sem_estimate import build_sem_hat

from numpy.random import seed
import numpy as np

# Models
from src.methods.cbo import CBO
from src.methods.ceo import CEO
from seaborn import set_style, set_theme

from src.utils.ceo_utils import store_results

import argparse
from src.utils.utilities import powerset, get_monte_carlo_expectation
from src.utils.sequential_intervention_functions import (
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)

import pickle
from itertools import cycle

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

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

def simulate_interventional_data_for_HealthCareSEM(
        graph,
        intervention_domain,
        size_intervention_grid,
        initial_structural_equation_model,
        structural_equation_model,
        exp_sets,
        seed=0,
        random_state=None,
):
    assert not random_state is None
    # Passed from outside, seed for the specific interv. point . Used to select different levels
    np.random.seed(seed)

    interventional_data = {}
    interventional_data_noiseless = {}

    canonical_exploration_sets = list(powerset(intervention_domain.keys()))

    # Get the interventional grids
    interventional_grids = get_interventional_grids(
        canonical_exploration_sets, intervention_domain, size_intervention_grid=size_intervention_grid
    )
    levels = {es: None for es in canonical_exploration_sets}
    for es in canonical_exploration_sets:
        idx = np.random.randint(0, interventional_grids[es].shape[0])  # Random indices
        levels[es] = interventional_grids[es][idx, :]
    T = 1

    """
    do(S_0)
    """
    interv = make_sequential_intervention_dictionary(graph, time_series_length=T)
    # Univariate intervention at time 0
    interv["S"][0] = float(levels[("S",)])

    # static_noise_model = {k: np.zeros(T) for k in ["X", "Z", "Y"]}
    # static_noise_model = {k: random_state.randn(T) * 0.5 for k in ["X", "Z", "Y"]}

    # Sample this model with one intervention
    intervention_samples = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        # epsilon=static_noise_model, #not passing = noise
    )
    interventional_data[("S",)] = get_monte_carlo_expectation(intervention_samples)
    for var in interventional_data[("S",)]:
        if len(interventional_data[("S",)][var].shape) == 1:
            interventional_data[("S",)][var] = interventional_data[("S",)][var].reshape(-1, 1)

    intervention_samples_noiseless = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon={k: np.zeros(T) for k in ["A","B","R","S","C","Y"]},
    )

    interventional_data_noiseless[("S",)] = get_monte_carlo_expectation(intervention_samples_noiseless)
    for var in interventional_data_noiseless[("S",)]:
        if len(interventional_data_noiseless[("S",)][var].shape) == 1:
            interventional_data_noiseless[("S",)][var] = interventional_data_noiseless[("S",)][var].reshape(-1, 1)
    """
    do(R_0)
    """
    interv = make_sequential_intervention_dictionary(graph, time_series_length=T)
    # Univariate intervention
    interv["R"][0] = float(levels[("R",)])
    # Sample this model with one intervention
    intervention_samples = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
    )
    interventional_data[("R",)] = get_monte_carlo_expectation(intervention_samples)
    for var in interventional_data[("R",)]:
        if len(interventional_data[("R",)][var].shape) == 1:
            interventional_data[("R",)][var] = interventional_data[("R",)][var].reshape(-1, 1)

    intervention_samples_noiseless = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon={k: np.zeros(T) for k in ["A","B","R","S","C","Y"]},
    )

    interventional_data_noiseless[("R",)] = get_monte_carlo_expectation(intervention_samples_noiseless)
    for var in interventional_data_noiseless[("R",)]:
        if len(interventional_data_noiseless[("R",)][var].shape) == 1:
            interventional_data_noiseless[("R",)][var] = interventional_data_noiseless[("R",)][var].reshape(-1, 1)

    """
    do(S_0, R_0)
    """
    interv = make_sequential_intervention_dictionary(graph, time_series_length=T)
    # # Multivariate intervention
    interv["S"][0] = float(levels[("S",)])
    interv["R"][0] = float(levels[("R",)])

    # Sample this model with one intervention
    intervention_samples = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
    )

    interventional_data[("S", "R")] = get_monte_carlo_expectation(intervention_samples)
    for var in interventional_data[("S", "R")]:
        if len(interventional_data[("S", "R")][var].shape) == 1:
            interventional_data[("S", "R")][var] = interventional_data[("S", "R")][var].reshape(-1,1)

    intervention_samples_noiseless = sequentially_sample_model(
        random_state,
        initial_structural_equation_model,
        structural_equation_model,
        total_timesteps=T,
        interventions=interv,
        sample_count=1,
        epsilon={k: np.zeros(T) for k in ["A","B","R","S","C","Y"]},
    )

    interventional_data_noiseless[("S", "R")] = get_monte_carlo_expectation(intervention_samples_noiseless)
    for var in interventional_data_noiseless[("S", "R")]:
        if len(interventional_data_noiseless[("S", "R")][var].shape) == 1:
            interventional_data_noiseless[("S", "R")][var] = interventional_data_noiseless[("S", "R")][var].reshape(-1,1)


    return interventional_data, interventional_data_noiseless



# I don't think this is used at all ?
seed(seed=0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seeds_replicate", type=int, nargs="+", help="seed for replicate: list of seeds for diff int. data"
)
parser.add_argument("--n_observational", type=int)
parser.add_argument("--n_trials", type=int)
parser.add_argument("--n_anchor_points", type=int)
parser.add_argument("--do_cdcbo", type=int)


args = parser.parse_args()

do_cdcbo = False if args.do_cdcbo > 1 else True

print('Doing CDCBO', do_cdcbo)

n_initial_int = len(args.seeds_replicate)
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_observational = args.n_observational

size_intervention_grid = 100
debug_mode = False

opt_results_for_pickle = {}

random_state = np.random.RandomState(0)

## Sample SEM to get observational samples

T = 1

init_sem, sem, true_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, ground_truth = setup_stat_scm_ceo_healthcare(
    T=T,
    random_state=random_state,
    n_anchor = n_anchor_points,
)


# True graph
G = graphs[0]

# Number of independent samples (samples here are the time-series on the horizontal) per time-index
D_O = sequentially_sample_model(
    random_state=random_state,
    static_sem=init_sem,
    dynamic_sem=sem,
    total_timesteps=T,
    sample_count=n_observational,  #  How many samples we take per node in each time-slice
    epsilon=None, #noise for observations
)  #  If we employ a noise model or not

# Multiple int data.  Dependent on passed seeds. As many points as passed seeds
dict_all_initial_data = {key: None for key in exploration_sets}
dict_all_initial_data_noiseless = {key: None for key in exploration_sets}

# Generation of interventional data
for s in seeds_int_data:
    current_int_data, current_int_data_noiseless = simulate_interventional_data_for_HealthCareSEM(
        graph=G,
        intervention_domain=intervention_domain,
        size_intervention_grid=size_intervention_grid,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        seed=s,
        exp_sets=exploration_sets,
        random_state=random_state,
    )
    for k, v in current_int_data.items():
        if dict_all_initial_data[k] is None:
            dict_all_initial_data[k] = v
        else:
            for inner_k, inner_v in v.items():
                dict_all_initial_data[k][inner_k] = np.vstack([v[inner_k], dict_all_initial_data[k][inner_k]])

    for k, v in current_int_data_noiseless.items():
        if dict_all_initial_data_noiseless[k] is None:
            dict_all_initial_data_noiseless[k] = v
        else:
            for inner_k, inner_v in v.items():
                dict_all_initial_data_noiseless[k][inner_k] = np.vstack(
                    [v[inner_k], dict_all_initial_data_noiseless[k][inner_k]])

D_I_noisy = dict_all_initial_data
D_I_noiseless = dict_all_initial_data_noiseless
# D_I_noisy = deepcopy(D_I_noiseless)


# Contains the exploration sets we will be investigating
print("Exploration sets:", exploration_sets)
# The intervention domains for the manipulative variables
print("Intervention domains:", intervention_domain)
#  The true outcome values of Y given an optimal intervention on the three time-slices
print(
    "True optimal outcome values:",
    [r"y^*_{} = {}".format(t, val.round(3)) for t, val in enumerate(true_objective_values)],
)

# # ### CEO
CEO_input_params = {
    "graphs": graphs,
    "init_posterior": [1 / len(graphs)] * len(graphs),
    "sem": HealthcareSEM,  # true sem. used to generate data, which always comes from the true graph !
    "base_target_variable": "Y",
    "make_sem_estimator": build_sem_hat,
    "ground_truth": ground_truth,
    "exploration_sets": exploration_sets,
    "observation_samples": D_O,
    "intervention_domain": intervention_domain,
    "intervention_samples": D_I_noisy,
    "intervention_samples_noiseless": D_I_noiseless,
    "number_of_trials": n_trials,
    "sample_anchor_points": True,
    "seed_anchor_points": 1,
    "num_anchor_points": n_anchor_points,
    "random_state": random_state,
    "debug_mode": debug_mode,
    "do_cdcbo": do_cdcbo
}

ceo = CEO(**CEO_input_params)
#
ceo.run()

opt_results_for_pickle = store_results(opt_results_for_pickle, "ceo", ceo)

if not do_cdcbo:
    ## Causal Bayesian Optimization

    CBO_input_params = {
        "G": G,
        "sem": HealthcareSEM,
        "base_target_variable": "Y",
        "make_sem_estimator": build_sem_hat,
        "exploration_sets": exploration_sets,
        "observation_samples": D_O,
        "debug_mode": debug_mode,
        "intervention_domain": intervention_domain,
        "ground_truth": ground_truth,
        "intervention_samples": D_I_noisy,
        "intervention_samples_noiseless": D_I_noiseless,
        "number_of_trials": n_trials,
        "sample_anchor_points": True,
        "seed_anchor_points": 1,
        "num_anchor_points": n_anchor_points,
        "random_state": random_state,
    }
    cbo = CBO(**CBO_input_params)
    cbo.run()

    opt_results_for_pickle = store_results(opt_results_for_pickle, "cbo_on_true_ei", cbo)

    # %%
    ##### CBO ON WRONG GRAPH
    post_wrong = [0.0] * len(graphs)
    wrong_graphs_cbo = {}
    for idx_wrong_g in range(len(graphs) - 1):
        post_wrong[idx_wrong_g + 1] = 1.0

        CBO_wrong_input_params = {
            "G": graphs[idx_wrong_g + 1],
            "sem": HealthcareSEM,
            "base_target_variable": "Y",
            "make_sem_estimator": build_sem_hat,
            "exploration_sets": exploration_sets,
            "observation_samples": D_O,
            "debug_mode": debug_mode,
            "intervention_domain": intervention_domain,
            "ground_truth": ground_truth,
            "intervention_samples": D_I_noisy,
            "intervention_samples_noiseless": D_I_noiseless,
            "number_of_trials": n_trials,
            "sample_anchor_points": True,
            "seed_anchor_points": 1,
            "num_anchor_points": n_anchor_points,
            "random_state": random_state,
        }

        wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)] = CBO(**CBO_wrong_input_params)
        wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)].run()

        opt_results_for_pickle = store_results(
            opt_results_for_pickle,
            "cbo_w_wrong_ei_{0}".format(idx_wrong_g),
            wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)],
        )

        # Reset
        for i in range(len(post_wrong)):
            post_wrong[i] = 0

os.chdir("..")

from pathlib import Path

Path("./results/cd_health_new").mkdir(parents=True, exist_ok=True)
Path("./results/comparison_cdhealth_new").mkdir(parents=True, exist_ok=True)

assert os.path.isdir('results/cd_health_new')
assert os.path.isdir('results/comparison_cdhealth_new')

# # # # # # # # # # # # # # # # #
# noise = "with_noise_Epidem"
if do_cdcbo:
    noise = "JUST_CDCBO"
    where = 'comparison_cdhealth_new'
else:
    noise = "with_noise_Health"
    where = 'cd_health_new'


unique_id = time.time()
with open(
        "results/"+where+"/results_epidem_seeds_{}_n_obs_{}__nanchor_{}_ntrials_{}_ID_{}_{}_leninitial_{}.pickle".format(
            seeds_int_data, n_observational, n_anchor_points, n_trials, unique_id, noise, len(seeds_int_data)
        ),
        "wb",
) as handle:
    #  We have to use dill because our object contains lambda functions.
    pickle.dump(opt_results_for_pickle, handle)
    handle.close()
# # # # # # # # # # # # # # # # # # #
#### Plotting
best_objective_values = true_objective_values
###### ###### ###### ###### ###### ###### ###### ###### ######
# os.chdir('..')
# In this part you need to hardcode directory and a string that uniquely indentifies results to grab
directory = os.getcwd() + '/results/cd_health_new'
data = []

for entry in os.scandir(directory):
    if (entry.path.endswith(".pickle")) and entry.is_file():
        with open(entry.path, "rb") as pickle_off:
            data.append(pickle.load(pickle_off))

directory = os.getcwd() + '/results/comparison_cdhealth_new'
# directory =  '/Users/nd/cloud/turing/private_CEO/results/comparison_cd'
data_cd = []

for entry in os.scandir(directory):
    if (entry.path.endswith(".pickle")) and entry.is_file():
        with open(entry.path, "rb") as pickle_off:
            data_cd.append(pickle.load(pickle_off))

###### ###### ###### ###### ###### ###### ###### ###### ######
# From here it's MOSTLY automated
methods_list = list(set.union(set([method for i in range(len(data)) for method in data[i].keys()])))
methods_list.append("cd_cbo")

for i in range(len(data)):
    d = {"cd_cbo": [[]]}
    data[i].update(d)
    data[i]["cd_cbo"][0].append(deepcopy(data_cd[i]["ceo"][0][0]))  # I costs
    data[i]["cd_cbo"][0].append(deepcopy(data_cd[i]["ceo"][0][1]))  # I values

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
            if np.isclose(y_start, globlmin):
                gap_dict[method] = 1.
            else:
                num = (y_best - y_start) / (globlmin - y_start) + ((H - H_best) / H)
                denom = 1 + ((H - 1) / H)
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

methods_list = ['ceo', 'cbo_on_true_ei', "cbo_wrong_avg", "cd_cbo"]

costs["cbo_wrong_avg"] = costs_wrong
outs["cbo_wrong_avg"] = outs_wrong
var_outs["cbo_wrong_avg"] = var_outs_wrong
gaps_avg["cbo_wrong_avg"] = gaps_avg_wrong
gaps_stds["cbo_wrong_avg"] = gaps_std_avg_wrong

# for method in methods_list:
#     print(method, "avg gap", gaps_avg[method], " stderr ", np.sqrt(gaps_stds[method]) /  np.sqrt(len(data)))

# plt.rc('text', usetex=True)
# # plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
# plt.rc('font', family='serif')
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
# plt.rcParams['font.size'] = 35
# plt.rcParams['legend.fontsize'] = 25
# plt.rcParams['xtick.labelsize'] = 25
# plt.rcParams['ytick.labelsize'] = 25
# plt.rcParams['axes.labelpad'] = 0
# plt.rcParams['figure.figsize'] = [12, 8]
# plt.rcParams['legend.loc'] = 'upper left'
# plt.rcParams['lines.linewidth'] = 10

# NEIL STUFF
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rc('font', family='serif')
plt.rc('font', size=30)
plt.rc('legend', fontsize=30)  # using a size in points

set_theme(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=2.8,
)
set_style({"xtick.direction": "in", "ytick.direction": "in"})

plot_params = {}
gr = 1.61803398875 * 0.9
h = 3.5 * 1.1
plot_params['figsize'] = (h * gr, h)
plot_params['xlabel'] = 'Cumulative intervention cost'  # r'$ \textsc{Co}\left (\mathbf{X}_I,\mathbf{x}_I \right)$'
# plot_params['ylabel'] = r'$y \left ( \mathbf{x}_{I}^{\textrm{best}} \right)$'
plot_params['ylabel'] = 'Best value achieved'
plot_params['marker_color'] = 'black'
plot_params['alpha'] = 0.25

#######################

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
fig, ax = plt.subplots(1, 1, figsize=plot_params['figsize'], facecolor="w", edgecolor="k")
for method in methods_list:
    col = next(color)
    cols.append(col)
    if method == 'ceo':
        label = 'CEO'
    elif method == 'cbo_wrong_avg':
        label = 'CBO with wrong graphs'
    elif method == 'cbo_on_true_ei':
        label = 'CBO with true graph'
    else:
        label = 'CD-CBO'

    ax.plot(costs[method], outs[method], label=label, c=col, ls=next(linestyles), lw=5, alpha=0.5)

for i, method in enumerate(methods_list):
    facecol = deepcopy(cols[i])
    facecol[3] = 0.2
    edgecolor = deepcopy(cols[i])
    edgecolor[3] = 0.6
    plt.fill_between(costs[method], outs[method] - np.sqrt(var_outs[method]) / np.sqrt(len(data)),
                     outs[method] + np.sqrt(var_outs[method]) / np.sqrt(len(data)), edgecolor=edgecolor,
                     facecolor=facecol, linewidth=3)

curr_color = next(color)
ax.plot(best_objective_values * 45, c=np.array([1., 0.70054304, 0.37841105, 1.]), label='True optimum', lw=6,
        alpha=0.5)
ax.set_xlim((0, 45.))
ax.fill_between(range(0, len(costs['ceo']) * 2), best_objective_values[0] - 0.05, best_objective_values[0] + 0.05,
                edgecolor=np.array([1., 0.70054304, 0.37841105, 1.]),
                facecolor=np.array([1., 0.70054304, 0.37841105, 1.]), linewidth=3, alpha=0.2)

# legend
legend = plt.legend(
    ncol=1,
    loc='upper left',
    prop={'size': 14},
    frameon=True
)
frame = legend.get_frame()
frame.set_color('white')

ax.set_xlabel(plot_params["xlabel"])
plt.ylabel(plot_params['ylabel'])
plt.title('Health')
plt.show()





# # I don't think this is used at all ?
# seed(seed=0)
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--seeds_replicate", type=int, nargs="+", help="seed for replicate: list of seeds for diff int. data"
# )
# parser.add_argument("--n_observational", type=int)
# parser.add_argument("--n_trials", type=int)
# parser.add_argument("--n_anchor_points", type=int)
#
# args = parser.parse_args()
#
# n_initial_int = len(args.seeds_replicate)
# seeds_int_data = args.seeds_replicate
#
# n_anchor_points = args.n_anchor_points
# n_trials = args.n_trials
# n_observational = args.n_observational
#
# size_intervention_grid = n_anchor_points
# debug_mode = False
#
# opt_results_for_pickle = {}
#
# random_state = np.random.RandomState(0)
#
# ## Sample SEM to get observational samples
#
# T = 1
#
# init_sem, sem, true_dag_view, graphs, exploration_sets, intervention_domain, true_objective_values, ground_truth = setup_stat_scm_ceo_healthcare(
#     T=T,
#     random_state=random_state,
#     n_anchor = n_anchor_points,
# )
#
#
# # True graph
# G = graphs[0]
#
# # Number of independent samples (samples here are the time-series on the horizontal) per time-index
# D_O = sequentially_sample_model(
#     random_state=random_state,
#     static_sem=init_sem,
#     dynamic_sem=sem,
#     total_timesteps=T,
#     sample_count=n_observational,  #  How many samples we take per node in each time-slice
#     epsilon=None, #noise for observations
# )  #  If we employ a noise model or not
#
# # Multiple int data.  Dependent on passed seeds. As many points as passed seeds
# dict_all_initial_data = {key: None for key in exploration_sets}
# dict_all_initial_data_noiseless = {key: None for key in exploration_sets}
#
# # Generation of interventional data
# for s in seeds_int_data:
#     current_int_data, current_int_data_noiseless = simulate_interventional_data_for_HealthCareSEM(
#         graph=G,
#         intervention_domain=intervention_domain,
#         size_intervention_grid=size_intervention_grid,
#         initial_structural_equation_model=init_sem,
#         structural_equation_model=sem,
#         seed=s,
#         exp_sets=exploration_sets,
#         random_state=random_state,
#     )
#     for k, v in current_int_data.items():
#         if dict_all_initial_data[k] is None:
#             dict_all_initial_data[k] = v
#         else:
#             for inner_k, inner_v in v.items():
#                 dict_all_initial_data[k][inner_k] = np.vstack([v[inner_k], dict_all_initial_data[k][inner_k]])
#
#     for k, v in current_int_data_noiseless.items():
#         if dict_all_initial_data_noiseless[k] is None:
#             dict_all_initial_data_noiseless[k] = v
#         else:
#             for inner_k, inner_v in v.items():
#                 dict_all_initial_data_noiseless[k][inner_k] = np.vstack(
#                     [v[inner_k], dict_all_initial_data_noiseless[k][inner_k]])
#
# D_I_noisy = dict_all_initial_data
# D_I_noiseless = dict_all_initial_data_noiseless
#
# # Contains the exploration sets we will be investigating
# print("Exploration sets:", exploration_sets)
# # The intervention domains for the manipulative variables
# print("Intervention domains:", intervention_domain)
# #  The true outcome values of Y given an optimal intervention on the three time-slices
# print(
#     "True optimal outcome values:",
#     [r"y^*_{} = {}".format(t, val.round(3)) for t, val in enumerate(true_objective_values)],
# )
#
# # # ### CEO
# CEO_input_params = {
#     "graphs": graphs,
#     "init_posterior": [1 / len(graphs)] * len(graphs),
#     "sem": HealthcareSEM,  # true sem. used to generate data, which always comes from the true graph !
#     "base_target_variable": "Y",
#     "make_sem_estimator": build_sem_hat,
#     "ground_truth": ground_truth,
#     "exploration_sets": exploration_sets,
#     "observation_samples": D_O,
#     "intervention_domain": intervention_domain,
#     "intervention_samples": D_I_noisy,
#     "intervention_samples_noiseless": D_I_noiseless,
#     "number_of_trials": n_trials,
#     "sample_anchor_points": True,
#     "seed_anchor_points": 1,
#     "num_anchor_points": n_anchor_points,
#     "random_state": random_state,
#     "debug_mode": debug_mode,
# }
#
# ceo = CEO(**CEO_input_params)
# #
# ceo.run()
#
# opt_results_for_pickle = store_results(opt_results_for_pickle, "ceo", ceo)
#
# ## Causal Bayesian Optimization
#
# CBO_input_params = {
#     "G": G,
#     "sem": HealthcareSEM,
#     "base_target_variable": "Y",
#     "make_sem_estimator": build_sem_hat,
#     "exploration_sets": exploration_sets,
#     "observation_samples": D_O,
#     "debug_mode": debug_mode,
#     "intervention_domain": intervention_domain,
#     "ground_truth": ground_truth,
#     "intervention_samples": D_I_noisy,
#     "intervention_samples_noiseless": D_I_noiseless,
#     "number_of_trials": n_trials,
#     "sample_anchor_points": True,
#     "seed_anchor_points": 1,
#     "num_anchor_points": n_anchor_points,
#     "random_state": random_state,
# }
# cbo = CBO(**CBO_input_params)
# cbo.run()
#
# opt_results_for_pickle = store_results(opt_results_for_pickle, "cbo_on_true_ei", cbo)
#
# # %%
# ##### CBO ON WRONG GRAPH
# post_wrong = [0.0] * len(graphs)
# wrong_graphs_cbo = {}
# for idx_wrong_g in range(len(graphs) - 1):
#     post_wrong[idx_wrong_g + 1] = 1.0
#
#     CBO_wrong_input_params = {
#         "G": graphs[idx_wrong_g + 1],
#         "sem": HealthcareSEM,
#         "base_target_variable": "Y",
#         "make_sem_estimator": build_sem_hat,
#         "exploration_sets": exploration_sets,
#         "observation_samples": D_O,
#         "debug_mode": debug_mode,
#         "intervention_domain": intervention_domain,
#         "ground_truth": ground_truth,
#         "intervention_samples": D_I_noisy,
#         "intervention_samples_noiseless": D_I_noiseless,
#         "number_of_trials": n_trials,
#         "sample_anchor_points": True,
#         "seed_anchor_points": 1,
#         "num_anchor_points": n_anchor_points,
#         "random_state": random_state,
#     }
#
#     wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)] = CBO(**CBO_wrong_input_params)
#     wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)].run()
#
#     opt_results_for_pickle = store_results(
#         opt_results_for_pickle,
#         "cbo_w_wrong_ei_{0}".format(idx_wrong_g),
#         wrong_graphs_cbo["cbo_w_wrong_ei_{0}".format(idx_wrong_g)],
#     )
#
#     # Reset
#     for i in range(len(post_wrong)):
#         post_wrong[i] = 0
#
# os.chdir("..")
#
# from pathlib import Path
#
# Path("./results/ultimo_health").mkdir(parents=True, exist_ok=True)
#
# assert os.path.isdir('results/ultimo_health')
# # # # # # # # # # # # # # # # # #
# noise = "with_noise_HEALTH"
# unique_id = time.time()
# with open(
#         "results/ultimo_health/results_toy_faster_seeds_{}_n_obs_{}__nanchor_{}_ntrials_{}_ID_{}_{}_leninitial_{}.pickle".format(
#             seeds_int_data, n_observational, n_anchor_points, n_trials, unique_id, noise, len(seeds_int_data)
#         ),
#         "wb",
# ) as handle:
#     #  We have to use dill because our object contains lambda functions.
#     pickle.dump(opt_results_for_pickle, handle)
#     handle.close()
# # # # # # # # # # # # # # # # # # # #
# ##### Plotting
# best_objective_values = true_objective_values
# ###### ###### ###### ###### ###### ###### ###### ###### ######
# # In this part you need to hardcode directory and a string that uniquely indentifies results to grab
# directory = os.getcwd() + '/results/ultimo_health'
# data = []
#
# for entry in os.scandir(directory):
#     if (entry.path.endswith(".pickle")) and entry.is_file():
#         if str(unique_id):
#         # if "trials_30" in entry.path:
#         # # if "1642718848.265944" in entry.path or \
#         # #         "1642718894.928093" in entry.path or \
#         # #         "1642737649.571897" in entry.path or \
#         # #         "1642718822.9783309" in entry.path:
#             with open(entry.path, "rb") as pickle_off:
#                 data.append(pickle.load(pickle_off))
# ###### ###### ###### ###### ###### ###### ###### ###### ######
# # From here it's MOSTLY automated
# methods_list = list(set.union(set([method for i in range(len(data)) for method in data[i].keys()])))
# from matplotlib import cm
#
# # num_wrong_graphs = sum('wrong' in method for method in data)
# linestyles = cycle(['-', '--', ':', '-.', '-', '--'])
#
# costs, outs, var_outs = {method: [] for method in methods_list}, {method: [] for method in methods_list}, {
#     method: [] for method in methods_list}
# for i in range(len(data)):
#     for method in methods_list:
#         if method in data[i]:
#             costs[method].append(np.cumsum(data[i][method][0][0][0]))
#             outs[method].append(data[i][method][0][1][0])
# for method in methods_list:
#     costs[method], outs[method] = np.vstack(costs[method]), np.vstack(outs[method])
#     costs[method], outs[method], var_outs[method] = costs[method].mean(0), outs[method].mean(0), outs[method].var(0)
#
# # WIll  average results across wrong graphs
# wrong_graphs = [method for method in methods_list if "wrong" in method]
#
# costs_wrong, outs_wrong, var_outs_wrong = [], [], []
#
# for wg in wrong_graphs:
#     costs_wrong.append(costs[wg])
#     outs_wrong.append(outs[wg])
#     var_outs_wrong.append(var_outs[wg])
# costs_wrong, outs_wrong, var_outs_wrong = np.vstack(costs_wrong).mean(0), np.vstack(outs_wrong).mean(0), np.vstack(
#     var_outs_wrong).mean(0)
#
# methods_list = ['ceo', 'cbo_on_true_ei', "cbo_wrong_avg"]
#
# costs["cbo_wrong_avg"] = costs_wrong
# outs["cbo_wrong_avg"] = outs_wrong
# var_outs["cbo_wrong_avg"] = var_outs_wrong
#
# # plt.rc('text', usetex=True)
# # plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
# # plt.rc('font', family='serif')
#
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
# # plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 25
# plt.rcParams['legend.fontsize'] = 25
# plt.rcParams['xtick.labelsize'] = 25
# plt.rcParams['ytick.labelsize'] = 25
# plt.rcParams['axes.labelpad'] = 10
# plt.rcParams['figure.figsize'] = [12, 8]
# plt.rcParams['legend.loc'] = 'upper right'
# plt.rcParams['lines.linewidth'] = 10
#
# cols = []
# color = iter(cm.rainbow(np.linspace(0, 1, len(methods_list) + 1)))  #
# for method in methods_list:
#     col = next(color)
#     cols.append(col)
#     if method == 'ceo':
#         label = 'CEO'
#     elif method == 'cbo_wrong_avg':
#         label = 'CBO with wrong G (avg)'
#     else:
#         label = 'CBO with true G'
#
#     plt.plot(costs[method], outs[method], label=label, c=col, ls=next(linestyles), lw=5)
#
# for i, method in enumerate(methods_list):
#     facecol = deepcopy(cols[i])
#     facecol[3] = 0.2
#     edgecolor = deepcopy(cols[i])
#     edgecolor[3] = 0.6
#     plt.fill_between(costs[method], outs[method] - np.sqrt(var_outs[method]) / np.sqrt(len(data)),
#                      outs[method] + np.sqrt(var_outs[method]) / np.sqrt(len(data)), edgecolor=edgecolor,
#                      facecolor=facecol, linewidth=3)
#
# curr_color = next(color)
# plt.plot(best_objective_values * 25, c=curr_color, label='optimum', lw=6)
# plt.xlim((0, 30.))
# plt.fill_between(range(0,len(costs['ceo']) * 2) , best_objective_values[0] - 0.02, best_objective_values[0] + 0.02, edgecolor=curr_color, facecolor=curr_color, linewidth=3 , alpha=0.2 )
# plt.legend()
# plt.xlabel('Cumulative intervention cost')
# plt.ylabel('Value of causal effect')
# plt.title('Healthcare graph, noisy')
# # plt.savefig('paper_noisy_toy.pdf', format='pdf',tight_layout=True)
# plt.show()