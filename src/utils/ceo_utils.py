import numpy as np
from matplotlib import pyplot as plt
from .sequential_intervention_functions import make_sequential_intervention_dictionary, reproduce_empty_intervention_blanket, assign_initial_intervention_level
from .sequential_sampling import sequential_sample_from_true_SEM
from .intervention_assignments import assign_initial_intervention_level, assign_intervention_level
from copy import deepcopy
from typing import Callable, Tuple, Union
from numpy import ndarray
from src.utils.utilities import make_column_shape_2D, normalize_log
from sklearn.neighbors import KernelDensity
import GPy

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)

def evaluate_target_function_all_for_ceo(
    noisy, random_state, initial_structural_equation_model, structural_equation_model, graph, exploration_set: tuple, all_vars, T: int,
):
    # Initialise temporal intervention dictionary
    intervention_blanket = make_sequential_intervention_dictionary(graph, T)
    keys = intervention_blanket.keys()

    def compute_target_function_all_for_ceo(current_target: str, intervention_levels: np.array, assigned_blanket: dict):

        # Split current target
        target_canonical_variable, target_temporal_index = current_target.split("_")
        target_temporal_index = int(target_temporal_index)

        # Populate the blanket in place
        if target_temporal_index == 0:
            intervention_blanket = reproduce_empty_intervention_blanket(T, keys)
            assign_initial_intervention_level(
                exploration_set=exploration_set,
                intervention_level=intervention_levels,
                intervention_blanket=intervention_blanket,
                target_temporal_index=target_temporal_index,
            )
        else:
            # Takes into account the interventions, assignments and target outcomes from the {t-1}
            intervention_blanket = deepcopy(assigned_blanket)
            assign_intervention_level(
                exploration_set=exploration_set,
                intervention_level=intervention_levels,
                intervention_blanket=intervention_blanket,
                target_temporal_index=target_temporal_index,
            )
        if noisy == False:
            static_noise_model = {k: np.zeros(T) for k in list(all_vars)}
        else:
            static_noise_model = None

        interventional_samples = sequential_sample_from_true_SEM(
            static_sem=initial_structural_equation_model,
            dynamic_sem=structural_equation_model,
            timesteps=T,
            epsilon=static_noise_model,
            interventions=intervention_blanket,
            random_state=random_state
        )

        # Compute the effect of intervention(s)
        target_response = compute_sequential_target_function_all_for_ceo(
            intervention_samples=interventional_samples,
            temporal_index=target_temporal_index,
            target_variable=target_canonical_variable,
        )
        return target_response

    return compute_target_function_all_for_ceo


def compute_sequential_target_function_all_for_ceo(
    intervention_samples: np.array, temporal_index: int, target_variable: str = "Y"
) -> np.array:
    if ~isinstance(temporal_index, int):
        temporal_index = int(temporal_index)
    # Function calculates the target provided the time-index is correct
    # assert intervention_samples[target_variable].shape[1]
    return intervention_samples


def update_posterior_observational(graphs, posterior, all_emission_fncs, new_observational_samples,
                                   total_timesteps, it, lr=.2):

    for graph_idx, emission_fncs in enumerate(all_emission_fncs):  # as many emission_fncs as graphs

        for temporal_index in range(total_timesteps):  # normally just one
            for pa in emission_fncs[temporal_index]:
                # Get relevant data for updating post
                xx, yy, inputs, output = get_sem_emit_obs(G=graphs[graph_idx],
                                                          sem_emit_fncs=deepcopy(emission_fncs), # TODO: check why it has problems when passing it the emissions
                                                          observational_samples=new_observational_samples,
                                                          t=temporal_index,
                                                          pa=pa,
                                                          t_index_data=None)

                # Learning rate
                posterior[graph_idx] += lr * log_likelihood(emission_fncs[temporal_index][pa], # the model
                                                             xx,
                                                             yy,
                                                             graph_idx,
                                                             inputs,
                                                             output, it)

    return posterior


def update_posterior_interventional(graphs, posterior, intervened_var, all_emission_fncs,
                                    interventional_samples,
                                    total_timesteps=1, it=0, lr=0.05):
    for graph_idx, emission_fncs in enumerate(all_emission_fncs):  # as many emission_fncs dicts as graphs

        for temporal_index in range(total_timesteps):  # normally just one
            # Here need to consider the truncated factorization
            for pa in emission_fncs[temporal_index]:
                # Get relevant data for updating post
                xx, yy, inputs, output = get_sem_emit_obs(G=graphs[graph_idx],
                                                          sem_emit_fncs=deepcopy(emission_fncs),
                                                          observational_samples=interventional_samples,
                                                          t=temporal_index,
                                                          pa=pa,
                                                          t_index_data=None)

                # Refit conditionals here  ?
                # data_X = emission_fncs[temporal_index][pa].X
                # data_X = np.vstack([data_X, xx])
                #
                # if isinstance(emission_fncs[temporal_index][pa], GPy.models.gp_regression.GPRegression):
                #
                #     data_y = emission_fncs[temporal_index][pa].Y
                #     data_y = np.vstack([data_y, yy])
                #
                #     #  Update in place
                #     emission_fncs[temporal_index][pa].set_XY(X=data_X, Y=data_y)
                #     emission_fncs[temporal_index][pa].optimize()
                #
                # elif isinstance(emission_fncs[temporal_index][pa], MyKDE):
                #     emission_fncs[temporal_index][pa] = emission_fncs[temporal_index][pa].fit_and_update(data_X)
                # else:
                #     raise NotImplementedError

                if isinstance(output,list):
                    assert len(output) == 1
                    output = output[0]

                if output in intervened_var:
                    # Here the truncated assumption comes in. Dont compute posterior
                    continue

                #  DEBUG
                # print("Intervening on :" + str(intervened_var) + " on graph: " + str(graph_idx))

                posterior[graph_idx] += lr * log_likelihood(emission_fncs[temporal_index][pa], # the model
                                                             xx,
                                                             yy,
                                                             graph_idx,
                                                             inputs,
                                                             output, it)

    # all_emission_fncs[0][0][(('X_0',), 'Z_0')].plot()
    # plt.title('X --> Z')
    # plt.show()
    #
    # all_emission_fncs[5][0][(('Z_0',), 'X_0')].plot()
    # plt.title('Z --> X')
    # plt.show()
    # print('Posterior: ', normalize_log(deepcopy(posterior)))
    return posterior


# This function has mostly debugging code; the logic is little (and simple)
def log_likelihood(model, X_test, y_test, graph_idx, inputs, output, it):
    if y_test is not None:
        ###################################################################################
        # inputs_plot = ""
        # for e in inputs:
        #     if len(e.split("_")) == 3:
        #         inputs_plot += e.split("_")[1] + " "
        #     else:
        #         inputs_plot += e.split("_")[0] + " "
        #
        # if not X_test.shape[1] > 1:
        #     limit_x = [
        #         np.min([np.min(X_test), np.min(model.X)]) - 0.5 * np.absolute(
        #             np.min([np.min(X_test), np.min(model.X)])),
        #         np.max([np.max(X_test), np.max(model.X)]) + 0.5 * np.absolute(
        #             np.max([np.max(X_test), np.max(model.X)]))]
        #     limit_y = [
        #         np.min([np.min(y_test), np.min(model.Y)]) - 0.5 * np.absolute(
        #             np.min([np.min(y_test), np.min(model.Y)])),
        #         np.max([np.max(y_test), np.max(model.Y)]) + 0.5 * np.absolute(
        #             np.max([np.max(y_test), np.max(model.Y)]))]
        #
        #     limit_x = [np.min([limit_x[0], -5.]), np.max([limit_x[1], 20.])]
        #
        #     grid_inputs = np.linspace(limit_x[0], limit_x[1], 1000).reshape(-1, 1)
        #
        #     f1 = lambda x: np.exp(-x)
        #     f2 = lambda x: np.cos(x) - np.exp(-x / 20.)
        #
        #     flag = False
        #
        #     # if inputs_plot == "X " and output == "Z":
        #     #     m, C = model.predict_noiseless(grid_inputs, full_cov=False)
        #     #     plot_gp(grid_inputs, m, C, training_points=(model.X, model.Y))
        #     #     plt.plot(np.linspace(-5, 5, 100), np.exp(-np.linspace(-5, 5, 100)), c='g', linewidth=3)
        #     #     flag = True
        #     #
        #     # elif inputs_plot == "Z " and output == "Y":
        #     #     m, C = model.predict_noiseless(grid_inputs, full_cov=False)
        #     #     plot_gp(grid_inputs, m, C, training_points=(model.X, model.Y))
        #     #     plt.plot(np.linspace(-5, 20, 100),
        #     #              np.cos(np.linspace(-5, 20, 100)) - np.exp(-np.linspace(-5, 20, 100) / 20.),
        #     #              c='g', linewidth=3)
        #     #     flag = True
        #
        #     # else:
        #     #     pass
        #     #
        #     # evaluations1 = f1(grid_inputs.flatten())
        #     # evaluations2 = f2(grid_inputs.flatten())
        #     # min_evaluations1 = np.min(evaluations1)
        #     # min_evaluations2 = np.min(evaluations2)
        #     # max_evaluations1 = np.max(evaluations1)
        #     # max_evaluations2 = np.max(evaluations2)
        #     #
        #     # limit_y = [np.min([min_evaluations1, min_evaluations2, limit_y[0]]),
        #     #            np.max([max_evaluations1, max_evaluations2, limit_y[1]])]
        #
        #     # plt.scatter(X_test, y_test, c='r', marker='o', s=60, clip_on=False)
        #     # plt.title("Graph " + str(graph_idx) + " Conditional: " + inputs_plot + " to " + output + " It. " + str(it))
        #
        #     if flag:
        #         # plt.xlim(limit_x)
        #         # # plt.ylim((limit_y[0], 20.))
        #         # plt.ylim(limit_y)
        #         pass
        #
        #     else:
        #         grid = np.linspace(np.min(model.X) - 1., np.max(model.X) + 1., 1000).reshape(-1, 1)
        #         # m, C = model.predict_noiseless(grid, full_cov=False)
        #         # plot_gp(grid, m, C, training_points=(model.X, model.Y))
        #
        #     # plt.show()
        ####################################################################################
        log_lik = model.log_predictive_density(x_test=X_test, y_test=y_test)

        res = log_lik.flatten().sum()
        inputs = ','.join(map(str,inputs))
        output = output[0]
        # print('Log lik at it. ' + str(it) + " : " + str(
        #     res) + " |  Conditional: " + inputs + " to " + output + " for graph: " + str(graph_idx))

    else:
        # Marginal
        log_lik = model.score_samples(X_test)
        res = log_lik.flatten().sum()

        # print('Log lik at it. ' + str(it) + " : " + str(res) + " |  Marginal: " + output + " for graph: " + str(
        #     graph_idx))

    return res

def plot_gp(X, m, C, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval
    plt.fill_between(X[:, 0],
                     m[:, 0] - 1.96 * np.sqrt(np.diag(C)),
                     m[:, 0] + 1.96 * np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])

    plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])

def store_results(opt_results_for_pickle, key, model):
    all_posteriors = []
    if hasattr(model,'all_posteriors'):
        all_posteriors = model.all_posteriors
    noisy_optimal_outcome_values_during_trials = None
    if hasattr(model,'noisy_optimal_outcome_values_during_trials'):
        noisy_optimal_outcome_values_during_trials = model.noisy_optimal_outcome_values_during_trials

    if key in opt_results_for_pickle:
        opt_results_for_pickle[key].append(
            (
                model.per_trial_cost,
                model.optimal_outcome_values_during_trials,
                noisy_optimal_outcome_values_during_trials,
                # model.optimal_intervention_sets,
                all_posteriors,  # this will  be [] for CBO
                model.interventional_data_x,
                model.interventional_data_y,
            )
        )
    else:
        opt_results_for_pickle[key] = [
            (
                model.per_trial_cost,
                model.optimal_outcome_values_during_trials,
                noisy_optimal_outcome_values_during_trials,
                # model.optimal_intervention_sets,
                all_posteriors,  # this will  be [] for CBO
                model.interventional_data_x,
                model.interventional_data_y,
            )
        ]
    return opt_results_for_pickle

# TODO: this is really a repetition since I needed this here outside the classs of CEO/CBO. I copied this from root.
def get_sem_emit_obs(G, sem_emit_fncs, observational_samples, t: int, pa: tuple, t_index_data: int = None,
) -> object:

    if t_index_data is not None:
        assert t_index_data - 1 == t, (t_index_data, t)
        #  Use past conditional
        t = t_index_data

    if len(pa) == 2 and pa[0] == None:
        # Source node
        pa_y = pa[1].split("_")[0]
        xx = make_column_shape_2D(observational_samples[pa_y])

        return (xx, None, 'Source', pa_y) #Changed for CEO

    # elif len(pa) == 3:
    #     # A fork in which a node has more than one child
    #     a, b = pa[0].split("_")[0], pa[2].split("_")[0]
    #     xx = make_column_shape_2D(observational_samples[a][t])
    #     yy = make_column_shape_2D(observational_samples[b][t])
    #     vv = [a]
    #     yyy = [b]
    else:
        # Loop over all parents / explanatory variables
        xx = []
        vv = []
        outputvar = pa[1].split("_")[0]

        for v in pa[0]:
            temp_v = v.split("_")[0]
            vv.append(temp_v)
            x = make_column_shape_2D(observational_samples[temp_v])
            xx.append(x)
        xx = np.hstack(xx)

        # Estimand (looks only at within time-slice targets)
        # ys = set.intersection(*map(set, [G.successors(v) for v in pa[0]]))
        # if len(ys) == 1:
        #     for y in ys:
        #         temp_y = y.split("_")[0]
        #         yyy.append(temp_y)
        #         yy = make_column_shape_2D(observational_samples[temp_y][t])
        # else:
        #     raise NotImplementedError("Have not covered DAGs with this type of connectivity.", (pa, ys))
        yy = make_column_shape_2D(observational_samples[outputvar])

    assert len(xx.shape) == 2
    assert len(yy.shape) == 2
    assert xx.shape[0] == yy.shape[0]  # Column arrays

    if xx.shape[0] != yy.shape[0]:
        min_rows = np.min((xx.shape[0], yy.shape[0]))
        xx = xx[: int(min_rows)]
        yy = yy[: int(min_rows)]

    return xx, yy, vv, [outputvar]

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
