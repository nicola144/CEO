from copy import deepcopy
from random import choice
from typing import Callable, Tuple, Union
from networkx.algorithms.dag import topological_sort

import numpy as np
from matplotlib import pyplot as plt
from networkx.classes.multidigraph import MultiDiGraph
from numpy.core.multiarray import ndarray
from numpy.core.numeric import nan, squeeze
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from src.bayes_opt.cost_functions import define_costs, total_intervention_cost
from src.utils.gp_utils import update_sufficient_statistics_hat
from src.utils.sequential_intervention_functions import (
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)
from src.utils.sequential_sampling import sequentially_sample_model
from src.utils.utilities import (
    assign_blanket,
    assign_blanket_hat,
    check_blanket,
    check_reshape_add_data,
    convert_to_dict_of_temporal_lists,
    create_intervention_exploration_domain,
    initialise_DCBO_parameters_and_objects_filtering,
    initialise_global_outcome_dict_new,
    initialise_optimal_intervention_level_list,
    make_column_shape_2D,
    normalize_log,
)
from src.utils.ceo_utils import evaluate_target_function_all_for_ceo, update_posterior_interventional, set_share_axes
from src.utils.ces_utils import MyKDENew, sample_global_xystar, update_arm_dist, build_pystar, to_prob
from functools import partial
import seaborn as sns

from notebooks.ceo_display_epidem import set_plotting


class BaseClassCeo:
    """
    Base class .
    """

    def __init__(
        self,
        # G: str,
        random_state,
        graphs: list,  # NEW
        init_posterior: list,  # NEW
        sem: classmethod,
        observation_samples: dict,
        intervention_domain: dict,
        make_sem_estimator: Callable = None,
        intervention_samples: dict = None,
        intervention_samples_noiseless: dict = None,
        exploration_sets: list = None,
        estimate_sem: bool = False,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        use_mc: bool = False,
        number_of_trials=10,
        ground_truth: ndarray = None,
        n_restart: int = 1,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
        do_cdcbo = False,
    ):
        if args_sem is None and change_points is None:
            true_sem = sem()
        elif args_sem and change_points is None:
            true_sem = sem(args_sem[0], args_sem[1])
        else:
            true_sem = sem(change_points.index(True))

        # These will be used in the target function evaluation
        self.true_initial_sem = true_sem.static()  # for t = 0
        self.true_sem = true_sem.dynamic()  # for t > 0
        self.make_sem_hat = make_sem_estimator

        self.random_state = random_state
        self.do_cdcbo = do_cdcbo

        # Graphs
        self.graphs = graphs

        G = graphs[0]  # NEW. True graph is first
        for g in graphs:
            assert isinstance(g, MultiDiGraph)
        self.T = int(list(G.nodes())[-1].split("_")[-1]) + 1  # Total time-steps in DAG
        for g in graphs:
            g.T = self.T

        self.G = G

        all_sorted_nodes = {}
        for g in graphs:
            sorted_nodes = {val: ix for ix, val in enumerate(topological_sort(g))}
            all_sorted_nodes[g] = sorted_nodes

        self.all_sorted_nodes = all_sorted_nodes
        self.debug_mode = debug_mode
        # Number of optimization restart for GPs
        self.n_restart = n_restart
        self.online = online
        self.use_mc = use_mc

        self.observational_samples = observation_samples
        self.base_target_variable = base_target_variable  # This has to be reflected in the CGM
        self.index_name = 0
        self.number_of_trials = number_of_trials

        #  Parents of all nodes
        # self.node_pars = {node: None for node in G.nodes}  # IS THIS USED ?
        # for node in G.nodes:
        #     self.node_pars[node] = tuple(G.predecessors(node))

        # Check that we are either minimising or maximising the objective function
        assert task in ["min", "max"], task
        self.task = task
        if task == "min":
            self.blank_val = 1e7  # Positive "infinity" (big number)
        elif task == "max":
            self.blank_val = -1e7  # Negative "infinity" (small number)

        # Instantiate blanket that will form final solution
        self.optimal_blanket = make_sequential_intervention_dictionary(self.G, self.T)

        # Contains all values a assigned as the DCBO walks through the graph; optimal intervention level are assigned at the same temporal level, for which we then use spatial SEMs to predict the other variable levels on that time-slice.
        self.assigned_blanket = deepcopy(self.optimal_blanket)
        self.empty_intervention_blanket = make_sequential_intervention_dictionary(self.G, self.T)

        # Canonical manipulative variables
        if manipulative_variables is None:
            self.manipulative_variables = list(
                filter(lambda k: self.base_target_variable not in k, self.observational_samples.keys(),)
            )
        else:
            self.manipulative_variables = manipulative_variables

        self.interventional_variable_limits = intervention_domain
        # assert self.manipulative_variables == list(intervention_domain.keys()) TODO
        if exploration_sets:
            assert isinstance(exploration_sets, list)
            self.exploration_sets = exploration_sets
        else:
            # When the only intervention is on the parents of the target variable
            self.exploration_sets = [tuple(self.manipulative_variables)]

        # Extract all target variables from the causal graphical model
        self.all_target_variables = list(filter(lambda k: self.base_target_variable in k, self.G.nodes))

        # Objective function params
        self.bo_model = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}

        # Target functions for Bayesian optimisation
        self.target_functions = deepcopy(self.bo_model)

        self.target_functions_noiseless = deepcopy(self.bo_model)

        # Store true objective function
        self.ground_truth = ground_truth
        # Number of points where to evaluate acquisition function
        self.num_anchor_points = num_anchor_points

        # Get the interventional grids
        self.interventional_grids = get_interventional_grids(
            self.exploration_sets, intervention_domain, size_intervention_grid=self.num_anchor_points
        )

        ##################################################################################################################
        # Changed/Added for CEO
        self.init_functions_mean_var = [
            {t: {es: None for es in self.exploration_sets} for t in range(self.T)} for _ in graphs
        ]

        self.init_mean_fun_agg = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}
        self.init_var_fun_agg = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}

        self.arm_mapping_es_to_num = dict(zip(set(self.exploration_sets), range(len(self.exploration_sets))))
        self.arm_mapping_num_to_es = dict(zip(range(len(self.exploration_sets)), set(self.exploration_sets)))
        # Uniform prior over arms - could use prior GP
        self.arm_distribution = [1.0 / len(self.exploration_sets)] * len(
            self.exploration_sets
        )  # TODO: Maybe move this to child class

        # Assigned during optimisation
        self.mean_functions = deepcopy(self.init_functions_mean_var)
        self.variance_functions = deepcopy(self.init_functions_mean_var)
        # True mf is an aggregate
        self.mean_function = deepcopy(self.init_mean_fun_agg)
        self.variance_function = deepcopy(self.init_var_fun_agg)

        # Store the dict for mean and var values computed in the acquisition function
        self.all_mean_dict_store = [
            {t: {es: {} for es in self.exploration_sets} for t in range(self.T)} for _ in graphs
        ]
        self.all_var_dict_store = deepcopy(self.all_mean_dict_store)

        ##################################################################################################################

        # For logging
        self.sequence_of_interventions_during_trials = [[] for _ in range(self.T)]

        ##################################################################################################################
        # Initial optimal solutions
        if intervention_samples and intervention_samples_noiseless:
            self.initial_intervention_samples = intervention_samples
            # Provide initial interventional data
            (
                initial_optimal_sequential_intervention_sets,
                initial_optimal_target_values,
                initial_optimal_sequential_intervention_levels,
                self.interventional_data_x,
                self.interventional_data_y,
            ) = initialise_DCBO_parameters_and_objects_filtering(
                exploration_sets=self.exploration_sets,
                interventional_data=intervention_samples,
                interventional_data_noiseless=intervention_samples_noiseless,
                base_target=self.base_target_variable,
                total_timesteps=self.T,
                task=self.task,
                index_name=0,
                nr_interventions=None,  # There are interventions, we just don't sub-sample them. TODO:hard coded
            )

        else:
            # No initial interventional data
            initial_optimal_sequential_intervention_sets = [choice(self.exploration_sets)] + (self.T - 1) * [None]
            initial_optimal_target_values = self.T * [None]
            initial_optimal_sequential_intervention_levels = self.T * [None]
            self.interventional_data_x = deepcopy(self.bo_model)
            self.interventional_data_y = deepcopy(self.bo_model)
            self.initial_intervention_samples = None
        ##################################################################################################################

        # Dict indexed by the global exploration sets, stores the best
        self.outcome_values = initialise_global_outcome_dict_new(self.T, initial_optimal_target_values, self.blank_val)
        # NEW for CEO
        # self.noisy_outcome_values = initialise_global_outcome_dict_new(
        #     self.T, deepcopy(initial_optimal_target_values), deepcopy(self.blank_val)
        # )

        self.optimal_outcome_values_during_trials = [[] for _ in range(self.T)]

        self.optimal_intervention_levels = initialise_optimal_intervention_level_list(
            self.T,
            self.exploration_sets,
            initial_optimal_sequential_intervention_sets,
            initial_optimal_sequential_intervention_levels,
            number_of_trials,
        )
        self.best_initial_es = initial_optimal_sequential_intervention_sets[0]  # 0 indexes the first time-step
        # NEW IN CEO : NEED ALL THE VARIABLES
        for temporal_index in range(self.T):
            for es in self.exploration_sets:
                self.target_functions[temporal_index][es] = evaluate_target_function_all_for_ceo(
                    random_state=self.random_state,
                    initial_structural_equation_model=self.true_initial_sem,
                    structural_equation_model=self.true_sem,
                    graph=self.G,
                    exploration_set=es,
                    all_vars=self.observational_samples.keys(),
                    T=self.T,
                    noisy=True,
                )

                self.target_functions_noiseless[temporal_index][es] = evaluate_target_function_all_for_ceo(
                    random_state=self.random_state,
                    initial_structural_equation_model=self.true_initial_sem,
                    structural_equation_model=self.true_sem,
                    graph=self.G,
                    exploration_set=es,
                    all_vars=self.observational_samples.keys(),
                    T=self.T,
                    noisy=False,
                )

        # Parameter space for optimisation
        self.intervention_exploration_domain = create_intervention_exploration_domain(
            self.exploration_sets, intervention_domain,
        )

        # Optimisation specific parameters to initialise
        self.trial_type = [[] for _ in range(self.T)]  # If we observed or intervened during the trial
        self.cost_functions = define_costs(self.manipulative_variables, self.base_target_variable, cost_type)
        self.per_trial_cost = [[] for _ in range(self.T)]
        self.optimal_intervention_sets = [None for _ in range(self.T)]

        # Acquisition function specifics
        self.y_acquired = {es: None for es in self.exploration_sets}
        self.corresponding_x = deepcopy(self.y_acquired)
        self.estimate_sem = estimate_sem
        if self.estimate_sem:
            self.assigned_blanket_hat = deepcopy(self.optimal_blanket)

    def node_parents(self, node: str, temporal_index: int = None, graph=None) -> tuple:  # Graph added
        """
        Returns the parents of this node *IN graph* with optional filtering on the time-index.

        Parameters
        ----------
        node : str
            The node of interest
        temporal_index : int, optional
            Select from which time-slice we want nodes only, by default None

        Returns
        -------
        tuple
            Parents of the node, optionally filtered
        """
        assert graph is not None
        graph_idx = self.graphs.index(graph)

        # DEBUG
        # if not list(graph.edges) == [('X_0', 'Z_0', 0), ('Z_0', 'Y_0', 0)]:
        #     print(graph.edges)

        if temporal_index is not None:
            #  This return has to have this complex form because the fitted SEM functions expect multivariate inputs
            # in a specific order (the topological order) of the nodes. Hence the additional sorting.

            res = tuple(
                sorted(
                    filter(lambda x: x.endswith(str(temporal_index)), graph.predecessors(node)),
                    key=self.all_sorted_nodes[graph].get,
                )
            )
            if not res:
                return res

            # DEBUG
            # if not list(graph.edges) == [('X_0', 'Z_0', 0), ('Z_0', 'Y_0', 0)]:
            #     print("node: ", node)
            #     print("The emissions in this graph are (showing keys only)" + str(
            #         list(self.all_sem_emit_fncs[graph_idx][temporal_index].keys())))
            #     print("But what will be returned by node_parents for this node" + node + " is " + str(res))

            for k in self.all_sem_emit_fncs[graph_idx][temporal_index].keys():
                # if len(k) == 2:
                assert len(k) == 2
                if k[-1] == node and res == k[0]:
                    return k
            return res

        else:
            return tuple(graph.predecessors(node))

    # def node_parents(self, node: str, temporal_index: int = None, graph = None) -> tuple:
    #     """
    #     Returns the parents of this node *IN graph* with optional filtering on the time-index.
    #
    #     Parameters
    #     ----------
    #     node : str
    #         The node of interest
    #     temporal_index : int, optional
    #         Select from which time-slice we want nodes only, by default None
    #
    #     Returns
    #     -------
    #     tuple
    #         Parents of the node, optionally filtered
    #     """
    #     assert graph is not None
    #     graph_idx = self.graphs.index(graph)
    #
    #     # DEBUG
    #     if not list(graph.edges) == [('X_0', 'Z_0', 0), ('Z_0', 'Y_0', 0)]:
    #         print(graph.edges)
    #
    #     if temporal_index is not None:
    #         #  This return has to have this complex form because the fitted SEM functions expect multivariate inputs
    #         # in a specific order (the topological order) of the nodes. Hence the additional sorting.
    #
    #         res = tuple(
    #             sorted(
    #                 filter(lambda x: x.endswith(str(temporal_index)), graph.predecessors(node)),
    #                 key=self.all_sorted_nodes[graph].get,
    #             )
    #         )
    #         # DEBUG
    #         if not list(graph.edges) == [('X_0', 'Z_0', 0), ('Z_0', 'Y_0', 0)]:
    #             print("node: ", node)
    #             print("The emissions in this graph are (showing keys only)" + str(list(self.all_sem_emit_fncs[graph_idx][temporal_index].keys())) )
    #             print("But what will be returned by node_parents for this node" + node +" is " +  str(res))
    #         return res
    #     else:
    #         return tuple(graph.predecessors(node))

    def _get_sem_emit_obs(
        self, t: int, pa: tuple, t_index_data: int = None
    ) -> Tuple[Union[None, ndarray], Union[None, ndarray]]:
        print("should not go here for CEO because t > 1")
        print("_get_sem_emit_obs")
        exit()
        if t_index_data is not None:
            assert t_index_data - 1 == t, (t_index_data, t)
            #  Use past conditional
            t = t_index_data

        if len(pa) == 2 and pa[0] == None:
            # Source node
            pa_y = pa[1].split("_")[0]
            xx = make_column_shape_2D(self.observational_samples[pa_y][t])
            # TODO: use statsmodels
            # self.sem_emit_fncs[t][pa_y] = KernelDensity(kernel="gaussian").fit(xx)
            self.sem_emit_fncs[t][pa_y] = sm.nonparametric.KDEUnivariate(xx)
            return (None, None)
        elif len(pa) == 3 and isinstance(pa[1], int):
            # A fork in which a node has more than one child
            a, b = pa[0].split("_")[0], pa[2].split("_")[0]
            xx = make_column_shape_2D(self.observational_samples[a][t])
            yy = make_column_shape_2D(self.observational_samples[b][t])
        else:
            # Loop over all parents / explanatory variables
            xx = []
            for v in pa:
                x = make_column_shape_2D(self.observational_samples[v.split("_")[0]][t])
                xx.append(x)
            xx = np.hstack(xx)
            # Estimand (looks only at within time-slice targets)
            ys = set.intersection(*map(set, [self.G.successors(v) for v in pa]))
            if len(ys) == 1:
                for y in ys:
                    yy = make_column_shape_2D(self.observational_samples[y.split("_")[0]][t])
            else:
                raise NotImplementedError("Have not covered DAGs with this type of connectivity.", (pa, ys))

        assert len(xx.shape) == 2
        assert len(yy.shape) == 2
        assert xx.shape[0] == yy.shape[0]  # Column arrays

        if xx.shape[0] != yy.shape[0]:
            min_rows = np.min((xx.shape[0], yy.shape[0]))
            xx = xx[: int(min_rows)]
            yy = yy[: int(min_rows)]

        return xx, yy

    def _plot_surrogate_model_first(self, temporal_index):
        # Plot model
        for es in self.exploration_sets:
            if len(es) == 1:
                inputs = np.asarray(self.interventional_grids[es])

                if self.bo_model[temporal_index][es] is not None:
                    mean, var = self.bo_model[temporal_index][es].predict(self.interventional_grids[es])
                    var = 1.96 * np.sqrt(var)
                    print("\n\t\t[1] The BO model exists for ES: {} at t == {}.\n".format(es, temporal_index))
                    print("Assigned blanket", self.assigned_blanket)
                else:
                    return

                true = make_column_shape_2D(self.ground_truth[temporal_index][es])

                if (
                    self.interventional_data_x[temporal_index][es] is not None
                    and self.interventional_data_y[temporal_index][es] is not None
                ):
                    plt.scatter(
                        self.interventional_data_x[temporal_index][es], self.interventional_data_y[temporal_index][es],
                        marker='D',c='m',s=80,
                    )

                # plt.scatter(
                #     self.observational_samples[es[0]][temporal_index],
                #     self.observational_samples["Y"][temporal_index],
                #     c="k",
                #     s=50,
                # )

                plt.fill_between(inputs[:, 0], (mean - var)[:, 0], (mean + var)[:, 0], edgecolor=(0. , 0.4, 1., 0.99) , facecolor=(0. , 0.4, 1., 0.5))
                plt.plot(
                    inputs, mean, "b", label="Causal prior for $do{}$ ".format(es),
                )
                plt.plot(inputs, true, "r", lw=2.5, label="True causal effect")
                plt.legend()
                plt.show()

    def _plot_surrogate_model(
        self,
        temporal_index,
        it,
        inputs_acq_dict,
        improvs_dict,
        pystar_samples,
        pxstar_samples,
        samples_global_ystar,
        samples_global_xstar,
        arm_mapping_es_to_num,
    ):
        # Plot model
        set_plotting()
        n_points = 0
        for es in self.exploration_sets:
            if len(es) == 1:
                if es[0] in ["Z", "X"] or es[0] in ["R", "S"]:  # will not plot the multi dim one
                    inputs_acq, improvs = inputs_acq_dict[es], improvs_dict[es]
                    corresponding_idx = arm_mapping_es_to_num[es]
                    inputs = np.asarray(self.interventional_grids[es])

                    if self.bo_model[temporal_index][es] is not None:
                        mean, var = self.bo_model[temporal_index][es].predict(self.interventional_grids[es])
                        print("\n\t\t[1] The BO model exists for ES: {} at t == {}.\n".format(es, temporal_index))
                        print("Assigned blanket", self.assigned_blanket)
                    else:
                        mean_f = partial(
                            self.mean_function,
                            t=temporal_index,
                            exploration_set=es,
                            p=normalize_log(deepcopy(self.posterior)),
                        )
                        mean = mean_f(self.interventional_grids[es])
                        var_f = partial(
                            self.variance_function,
                            t=temporal_index,
                            exploration_set=es,
                            p=normalize_log(deepcopy(self.posterior)),
                        )
                        var = var_f(self.interventional_grids[es]) + np.ones_like(var_f(self.interventional_grids[es]))

                    true = make_column_shape_2D(self.ground_truth[temporal_index][es])

                    if (
                        self.interventional_data_x[temporal_index][es] is not None
                        and self.interventional_data_y[temporal_index][es] is not None
                    ):
                        n_points = self.interventional_data_x[temporal_index][es].shape[0]

                    fig, ax1 = plt.subplots(4, 2, gridspec_kw={"height_ratios": [3, 1, 1, 1], "width_ratios": [1, 3]})
                    set_share_axes(ax1[:, 1], sharex=True)
                    set_share_axes(ax1[0, :], sharey=True)

                    gs = ax1[3, 0].get_gridspec()
                    # remove the underlying axes
                    for cax in ax1[3, :]:
                        cax.remove()

                    axbig = fig.add_subplot(gs[3, :])

                    if samples_global_ystar is not None:
                        sns.kdeplot(samples_global_ystar.squeeze(), ax=axbig, alpha=0.22)
                        axbig.set_title("Global y star")
                    else:
                        axbig.axis("off")

                    ################## ######### #########
                    # Plot predictive mean and var
                    var = np.sqrt(var)

                    ax1[0][1].fill_between(
                        inputs[:, 0],
                        (mean - 2.0 * var)[:, 0],
                        (mean + 2.0 * var)[:, 0],
                        edgecolor=(0.0, 0.4, 1.0, 0.99),
                        facecolor=(0.0, 0.4, 1.0, 0.6),
                    )
                    ax1[0][1].plot(
                        inputs, mean, c="b", label="$do{}$".format(es), lw=5.0,
                    )
                    ################## ######### #########
                    copy_bo_model = deepcopy(self.bo_model[temporal_index][es])
                    copy_bo_model.model.kern.variance_adjustment = partial(
                        self.variance_function_lessnoise,
                        t=temporal_index,
                        exploration_set=es,
                        posterior=normalize_log(deepcopy(self.posterior)),
                    )

                    copymean, copyvar = copy_bo_model.predict(self.interventional_grids[es])
                    copyvar = np.sqrt(copyvar)

                    ax1[0][1].fill_between(
                        inputs[:, 0],
                        (copymean - 2.0 * copyvar)[:, 0],
                        (copymean + 2.0 * copyvar)[:, 0],
                        edgecolor=(0.0, 0.4, 1.0, 0.55),
                        facecolor=(0.0, 0.4, 1.0, 0.3),
                    )

                    ################## ######### #########

                    ax1[0][1].plot(inputs, true, "r", label="True ", lw=5.0)
                    leg1 = ax1[0][1].legend()

                    ints = ax1[0][1].scatter(
                        self.interventional_data_x[temporal_index][es][:-1],
                        self.interventional_data_y[temporal_index][es][:-1],
                        s=200,
                        marker="D",
                        c="m",
                    )

                    latest_int = ax1[0][1].scatter(
                        self.interventional_data_x[temporal_index][es][-1],
                        self.interventional_data_y[temporal_index][es][-1],
                        s=300,
                        marker="X",
                        c="r",
                    )

                    obs = ax1[0][1].scatter(
                        self.observational_samples[es[0]][temporal_index],
                        self.observational_samples["Y"][temporal_index],
                        c="k",
                        s=200,
                    )

                    leg2 = ax1[0][1].legend(
                        [ints, latest_int, obs], ["Interventions", "Latest interv. ", "Observations"], loc=4
                    )
                    ax1[0][1].add_artist(leg1)

                    ax1[0][1].set_ylim((11.,15.))

                    temp = (
                        normalize_log(deepcopy(self.posterior)).tolist()
                        + to_prob(deepcopy(self.arm_distribution)).squeeze().tolist()
                    )
                    ax1[1][0].bar(range(len(self.graphs) + len(self.arm_distribution)), temp)
                    ax1[1][0].set_yticks([0.005, 0.1, 0.25, 0.5, 0.75, 1.0])
                    str_sets = [str(e) for e in self.exploration_sets]
                    ax1[1][0].set_xticks(list(range(len(self.graphs))))
                    ax1[1][0].set_title("Graph and Arm distributions")

                    if pystar_samples is not None:
                        sns.kdeplot(
                            y=pystar_samples[corresponding_idx, :].squeeze(),
                            ax=ax1[0, 0],
                            alpha=0.22,
                            clip=(np.min(inputs), np.max(inputs)),
                        )
                        # ax1[0][0].hist(pymin_samples, bins=100, color='darkorange', density=True, orientation='horizontal')
                        # ax1[0][0].set_ylim(ax1[0][1].get_ylim())
                        ax1[0][0].set_xticks([])
                        ax1[0][0].set_title("Optimal value of Y density (local)")
                    else:
                        ax1[0, 0].axis("off")

                    if pxstar_samples is not None:
                        sns.kdeplot(
                            pxstar_samples[corresponding_idx].squeeze(),
                            ax=ax1[2, 1],
                            alpha=0.22,
                            clip=(np.min(inputs), np.max(inputs)),
                        )
                        # sns.displot(pxstar_samples[corresponding_idx].squeeze(), ax=ax1[2, 1],  alpha=0.22 )

                        # ax1[2][1].hist(pxstar_samples[corresponding_idx, :].squeeze(), bins=100, color='darkviolet', density=True, orientation='vertical')
                        # ax1[2][1].set_xticks(ax1[0][1].get_xticks())
                        # ax1[2][1].set_xlim(ax1[0][1].get_xlim())
                        ax1[2][1].set_yticks([])
                        ax1[2][1].set_title("Optimal value of " + str(es[0]) + " density (local)")
                    else:
                        ax1[2][1].axis("off")

                    fig.canvas.draw()

                    labels = [item.get_text() for item in ax1[1][0].get_xticklabels()]
                    labels = ["G" + str(e) for e in labels]
                    labels = labels + str_sets  # matplotlib does not like this for some reasons

                    # ax1[1][0].set_xticklabels(labels)
                    ax1[1][0].set_title("Graph and arm posterior")

                    if not it == 0:
                        ax1[1][1].scatter(inputs_acq.squeeze().tolist(), improvs.squeeze().tolist(), c="g", s=150)
                        # ax1[1][1].set_xlim(ax1[0][1].get_xlim())
                        ax1[1][1].set_xticks([])
                        # ax1[1][1].set_yticks([])

                        ax1[1][1].set_title("CES (for next iteration)")

                    ax1[0][1].set_title(" CEO " + " Iteration: " + str(it) + " N. of points: " + str(n_points))

                    if pxstar_samples is not None and pystar_samples is not None:
                        a = pxstar_samples[corresponding_idx].squeeze()
                        b = pystar_samples[corresponding_idx, :].squeeze()

                        sns.kdeplot(a, b, ax=ax1[2, 0], fill=True)
                        ax1[2][0].set_title("Joint local optimum density")
                    else:
                        ax1[2][0].axis("off")

                    fig.tight_layout(pad=5.0)

                    # plt.show()
                    # res_str = '../results/figg_it_' + str(it) + '_es_' + str(es[0]) + '.pdf'
                    # fig.savefig(res_str, dpi=fig.dpi)
                    # plt.close()

    def _update_opt_params(self, it: int, temporal_index: int, best_es: tuple) -> None:

        # When observed append previous optimal values for logs
        # Outcome values at previous step
        self.outcome_values[temporal_index].append(self.outcome_values[temporal_index][-1])
        # self.noisy_outcome_values[temporal_index].append(self.noisy_outcome_values[temporal_index][-1])

        if it == 0:
            # Special case for first time index
            # Assign outcome values that is the same as the initial value in first trial
            self.optimal_outcome_values_during_trials[temporal_index].append(self.outcome_values[temporal_index][-1])

            if self.interventional_data_x[temporal_index][best_es] is None:
                self.optimal_intervention_levels[temporal_index][best_es][it] = nan

            self.per_trial_cost[temporal_index].append(0.0)

        elif it > 0:
            # Get previous one cause we are observing thus we no need to recompute it
            self.optimal_outcome_values_during_trials[temporal_index].append(
                self.optimal_outcome_values_during_trials[temporal_index][-1]
            )
            self.optimal_intervention_levels[temporal_index][best_es][it] = self.optimal_intervention_levels[
                temporal_index
            ][best_es][it - 1]
            # The cost of observation is the same as the previous trial.
            self.per_trial_cost[temporal_index].append(self.per_trial_cost[temporal_index][-1])

    def _check_new_point(self, best_es, temporal_index):
        assert best_es is not None, (best_es, self.y_acquired)
        assert best_es in self.exploration_sets

        # Check that new intervenƒtion point is in the allowed intervention domain
        assert self.intervention_exploration_domain[best_es].check_points_in_domain(self.corresponding_x[best_es])[0], (
            best_es,
            temporal_index,
            self.y_acquired,
            self.corresponding_x,
        )

    def _check_optimization_results(self, temporal_index):
        # Check everything went well with the trials
        assert len(self.optimal_outcome_values_during_trials[temporal_index]) == self.number_of_trials, (
            len(self.optimal_outcome_values_during_trials[temporal_index]),
            self.number_of_trials,
        )
        assert len(self.per_trial_cost[temporal_index]) == self.number_of_trials, len(self.per_trial_cost)

        if temporal_index > 0:
            assert all(
                len(self.optimal_intervention_levels[temporal_index][es]) == self.number_of_trials
                for es in self.exploration_sets
            ), [len(self.optimal_intervention_levels[temporal_index][es]) for es in self.exploration_sets]

        assert self.optimal_intervention_sets[temporal_index] is not None, (
            self.optimal_intervention_sets,
            self.optimal_intervention_levels,
            temporal_index,
        )

    def _safe_optimization(
        self, temporal_index, exploration_set, lower_bound_var=1e-05, upper_bound_var=2.0, bound_len=20.0
    ):
        print("Fitting BO model for " + str(exploration_set))
        if self.bo_model[temporal_index][exploration_set].model.kern.variance[0] < lower_bound_var:
            print("safe optimization: resetting kernel var")
            self.bo_model[temporal_index][exploration_set].model.kern.variance[0] = 1.0

        if self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] > bound_len:
            print("safe optimization: resetting kernel lengthscale")
            self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] = 1.0

        if self.bo_model[temporal_index][exploration_set].model.likelihood.variance[0] > upper_bound_var:
            print("safe optimization: resetting lik var")
            self.bo_model[temporal_index][exploration_set].model.likelihood.variance[0] = upper_bound_var

        if self.bo_model[temporal_index][exploration_set].model.likelihood.variance[0] < lower_bound_var:
            print("safe optimization: resetting lik var")
            self.bo_model[temporal_index][exploration_set].model.likelihood.variance[0] = 1.

    def _get_updated_interventional_data(self, new_interventional_data_x, y_new, best_es, temporal_index):
        data_x, data_y = check_reshape_add_data(
            self.interventional_data_x,
            self.interventional_data_y,
            new_interventional_data_x,
            y_new,
            best_es,
            temporal_index,
        )
        self.interventional_data_x[temporal_index][best_es] = data_x
        self.interventional_data_y[temporal_index][best_es] = data_y

    def _plot_conditional_distributions(self, temporal_index, it):
        print("Time:", temporal_index)
        print("Iter:", it)
        print("\n### Emissions ###\n")
        for key in self.sem_emit_fncs[temporal_index]:
            if len(key) == 1:
                print("{}\n".format(key))
                if isinstance(self.sem_emit_fncs[temporal_index][key], dict):
                    for item in self.sem_emit_fncs[temporal_index][key]:
                        item.plot()
                        plt.show()
                else:
                    self.sem_emit_fncs[temporal_index][key].plot()
                    plt.show()

        print("\n### Transmissions ###\n")
        if callable(getattr(self.__class__, self.sem_trans_fncs)):
            for key in self.sem_trans_fncs.keys():
                if len(key) == 1:
                    print(key)
                    self.sem_trans_fncs[key].plot()
                    plt.show()

    def _update_sufficient_statistics(
        self, target: str, temporal_index: int, dynamic: bool, assigned_blanket: dict, all_updated_sem=None
    ) -> None:
        """
        Method to update mean and variance functions of the causal prior (GP).

        Parameters
        ----------
        target : str
            The full node name of the target variable.
        temporal_index : int
            The temporal index currently being explored by the algorithm.
        dynamic : bool
            Tells the algorithm whether or not to use horizontal information (i.e. transition information between temporal slices).
        assigned_blanket : dict
            The assigned values thus far, per time-slice, per node in the CGM.
        all_updated_sem : All sems
            Structural equation model.
        """

        # Check which current target we are dealing with, and in consequence where we are in time
        target_variable, target_temporal_index = target.split("_")
        assert int(target_temporal_index) == temporal_index

        mean_function_all, var_function_all = [], []
        # this was for debugging conditionals
        # fig, ax = plt.subplots(len(self.graphs), len(self.exploration_sets))

        for idx_graph, (updated_sem, mean_dict_store, var_dict_store) in enumerate(
            zip(all_updated_sem, self.all_mean_dict_store, self.all_var_dict_store,)
        ):

            mean_dict_all, var_dict_all = (
                {t: {es: None for es in self.exploration_sets} for t in range(self.T)},
                {t: {es: None for es in self.exploration_sets} for t in range(self.T)},
            )

            for idx_es, es in enumerate(self.exploration_sets):

                (curr_mf, curr_vf,) = update_sufficient_statistics_hat(
                    temporal_index=temporal_index,
                    target_variable=target_variable,
                    exploration_set=es,
                    sem_hat=updated_sem,
                    node_parents=partial(self.node_parents, graph=self.graphs[idx_graph]),
                    dynamic=dynamic,
                    assigned_blanket=assigned_blanket,
                    mean_dict_store=mean_dict_store,  # Note: these dicts are passed by ref. so will be modified
                    var_dict_store=var_dict_store,  # so no need to create new var.
                )

                mean_dict_all[temporal_index][es] = curr_mf
                var_dict_all[temporal_index][es] = curr_vf

                # if es == ('X',):
                #     # this was debugging conditionals
                #     inputs = np.asarray(self.interventional_grids[es])
                #     meanplot = curr_mf(self.interventional_grids[es])
                #     varplot= curr_vf(self.interventional_grids[es])
                #     varplot = np.sqrt(varplot)
                #     plt.clf()
                #     plt.close()
                #     plt.plot(inputs, meanplot, "b")
                #     plt.scatter(self.observational_samples[es[0]][temporal_index],self.observational_samples["Z"][temporal_index] )
                #     plt.fill_between(inputs[:, 0], (meanplot - varplot)[:, 0], (meanplot + varplot)[:, 0], edgecolor=(0, 0, 0.54, 0.99),facecolor=(0, 0, 0.54, 0.20))
                #     plt.title("Graph " + str(idx_graph) + " And ES: " + str(es))
                #     plt.show()
                #     plt.close()

                # ax[idx_graph][idx_es].fill_between(inputs[:, 0], (meanplot - varplot)[:, 0], (meanplot + varplot)[:, 0], edgecolor=(0, 0, 0.54, 0.99),
                #                  facecolor=(0, 0, 0.54, 0.20))
                # ax[idx_graph][idx_es].plot(
                #     inputs, meanplot, "b"
                # )
                #
                # ax[idx_graph][idx_es].set_title("mean f for graph"+ str(idx_graph) + " and es " + str(self.exploration_sets[idx_es][0]))

            mean_function_all.append(mean_dict_all)
            var_function_all.append(var_dict_all)

        # plt.title("Debug cndit: " )
        # plt.show()

        self.mean_functions, self.variance_functions = mean_function_all, var_function_all

        #  E[Y | do(x)] = E[E[Y|do(x), G]] = \sum_g  p(G = g) E[Y | do(x), G = g]
        def _aggregate_mean_function(x, t, exploration_set, posterior):
            unweighted_effects = np.asarray([f[t][exploration_set](x).flatten() for f in self.mean_functions])
            res = posterior.dot(unweighted_effects)
            return res[:, np.newaxis]

        #  V[Y | do(x)] = E[V[Y|do(x),G]] + V[E[Y|do(x), G]]
        #               = \sum_g p(G = g) V[Y|do(x),G]  + (added terms)   E[ (E[Y|do(x), G])^2 ] - (E[Y | do(x)])**2
        def _aggregate_var_function(x, t, exploration_set, posterior):

            unweighted_vars = np.asarray([f[t][exploration_set](x).flatten() for f in self.variance_functions])
            unweighted_second_moments = np.asarray(
                [f[t][exploration_set](x).flatten() ** 2 for f in self.mean_functions]
            )

            res = posterior.dot(unweighted_vars)

            # Added terms
            first_moment = _aggregate_mean_function(x, t, exploration_set, posterior)
            res += posterior.dot(unweighted_second_moments)
            res -= first_moment.squeeze() ** 2

            return res[:, np.newaxis]

        # This is just V[Y | do(x)] ~~ E[V[Y|do(x),G]]
        def _aggregate_var_function_lessnoise(x, t, exploration_set, posterior):

            unweighted_vars = np.asarray([f[t][exploration_set](x).flatten() for f in self.variance_functions])
            res = posterior.dot(unweighted_vars)

            return res[:, np.newaxis]

        self.mean_function, self.variance_function, self.variance_function_lessnoise = (
            _aggregate_mean_function,
            _aggregate_var_function,
            _aggregate_var_function_lessnoise,
        )

    def _update_observational_data(self, temporal_index):
        if temporal_index > 0:
            if self.online:
                if isinstance(self.n_obs_t, list):
                    local_n_t = self.n_obs_t[temporal_index]
                else:
                    local_n_t = self.n_obs_t
                assert local_n_t is not None

                # Sample new data
                set_observational_samples = sequentially_sample_model(
                    static_sem=self.true_initial_sem,
                    dynamic_sem=self.true_sem,
                    total_timesteps=temporal_index + 1,
                    sample_count=local_n_t,
                    use_sem_estimate=False,
                    interventions=self.assigned_blanket,
                )

                # Reshape data
                set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                for var in self.observational_samples.keys():
                    self.observational_samples[var][temporal_index] = set_observational_samples[var][temporal_index]
            else:
                if isinstance(self.n_obs_t, list):
                    local_n_obs = self.n_obs_t[temporal_index]

                    n_stored_observations = len(
                        self.observational_samples[list(self.observational_samples.keys())[0]][temporal_index]
                    )

                    if self.online is False and local_n_obs != n_stored_observations:
                        # We already have the same number of observations stored
                        set_observational_samples = sequentially_sample_model(
                            static_sem=self.true_initial_sem,
                            dynamic_sem=self.true_sem,
                            total_timesteps=temporal_index + 1,
                            sample_count=local_n_obs,
                            use_sem_estimate=False,
                        )
                        # Reshape data
                        set_observational_samples = convert_to_dict_of_temporal_lists(set_observational_samples)

                        for var in self.observational_samples.keys():
                            self.observational_samples[var][temporal_index] = set_observational_samples[var][
                                temporal_index
                            ]

    def _post_optimisation_assignments(self, target: tuple, t: int, DCBO: bool = False) -> None:

        # Index of the best value of the objective function
        best_objective_fnc_value_idx = self.outcome_values[t].index(eval(self.task)(self.outcome_values[t])) - 1

        # 1) Best intervention for this temporal index
        for es in self.exploration_sets:
            if isinstance(self.optimal_intervention_levels[t][es][best_objective_fnc_value_idx], ndarray,):
                # Check to see that the optimal intervention is not None
                check_val = self.optimal_intervention_levels[t][es][best_objective_fnc_value_idx]

                assert check_val is not None, (
                    t,
                    self.optimal_intervention_sets[t],
                    best_objective_fnc_value_idx,
                    es,
                )
                # This is the, overall, best intervention set for this temporal index.
                self.optimal_intervention_sets[t] = es
                break  # There is only one so we can break here

        # 2) Blanket stores optimal values (interventions and targets) found during DCBO.
        self.optimal_blanket[self.base_target_variable][t] = eval(self.task)(self.outcome_values[t])

        # 3) Write optimal interventions to the optimal blanket
        # for i, es_member in enumerate(set(es).intersection(self.manipulative_variables)):
        #     self.optimal_blanket[es_member][t] = float(
        #         self.optimal_intervention_levels[t][self.optimal_intervention_sets[t]][best_objective_fnc_value_idx][
        #             :, i
        #         ]
        #     )

        if DCBO:
            # 4) Finally, populate the summary blanket with info found in (1) to (3)
            assign_blanket_hat(
                self.assigned_blanket_hat,
                self.optimal_intervention_sets[t],  # Exploration set
                self.optimal_intervention_levels[t][self.optimal_intervention_sets[t]][
                    best_objective_fnc_value_idx
                ],  # Intervention level
                target=target,
                target_value=self.optimal_blanket[self.base_target_variable][t],
            )
            check_blanket(self.assigned_blanket_hat, self.base_target_variable, t, self.manipulative_variables)

        # 4) Finally, populate the summary blanket with info found in (1) to (3)
        # assign_blanket(
        #     self.true_initial_sem,
        #     self.true_sem,
        #     self.assigned_blanket,
        #     self.optimal_intervention_sets[t],
        #     self.optimal_intervention_levels[t][self.optimal_intervention_sets[t]][best_objective_fnc_value_idx],
        #     target=target,
        #     target_value=self.optimal_blanket[self.base_target_variable][t],
        #     G=self.G,
        # )
        # check_blanket(
        #     self.assigned_blanket, self.base_target_variable, t, self.manipulative_variables,
        # )
        #
        # # Check optimization results for the current temporal index before moving on
        # self._check_optimization_results(t)

    def _per_trial_computations(self, t: int, it: int, target: str, assigned_blanket: dict, method: str = None):

        if self.debug_mode:
            print("\n\n>>>")
            print("Iteration:", it)
            print("<<<\n\n")
            self._plot_surrogate_model_first(t)

        # Presently find the optimal value of Y_t
        current_best_global_target = eval(self.task)(self.outcome_values[t])

        # Presently find the optimal value of Y_t | NOISY! NEW for CEO
        # current_best_global_target_noisy = eval(self.task)(self.noisy_outcome_values[t])

        #  Just to indicate that in this trial we are explicitly intervening in the system
        self.trial_type[t].append("i")

        # TODO: CHANGE TO HANDLE NO INITIAL DATA
        if not all(value is None for value in self.bo_model[t].values()): # Should be none iff no initial data
            self.arm_distribution = update_arm_dist(
                deepcopy(self.arm_distribution),
                self.bo_model,
                self.interventional_grids,
                t,
                self.task,
                self.arm_mapping_es_to_num,
            )

            # CEO stuff
            print("Going to compute initial p y  star locals")
            pystar_samples, pxstar_samples = build_pystar(
                arm_mapping=self.arm_mapping_es_to_num,
                bo_model=self.bo_model[t],
                parameter_int_domain=self.intervention_exploration_domain,
                int_grids=self.interventional_grids,
                task=self.task,
                seed_anchor_points=0,
            )  # TODO
            samples_global_ystar, samples_global_xstar = sample_global_xystar(
                n_samples_mixture=1000,
                all_ystar=pystar_samples,
                all_xstar=pxstar_samples,
                arm_dist=to_prob(deepcopy(self.arm_distribution), task=self.task),
                arm_mapping_n_to_es=self.arm_mapping_num_to_es,
            )
            # Shared between arms !!!
            kde_globalystar = MyKDENew(samples_global_ystar)
            try:
                kde_globalystar.fit()
            except:
                kde_globalystar.fit(bw=0.5)


            # Compute acquisition function given the updated BO models for the interventional data. IN CEO WE USE NOISY OPTIMAL VALUE .
            # Note: this sets self.y_acquired . A bit shady
            inputs_acq_dict, improvs_dict = self._evaluate_acquisition_functions(
                t,
                current_best_global_target,
                it,
                graphs=self.graphs,
                posterior=deepcopy(self.posterior),
                kde_globalystar=deepcopy(kde_globalystar),
                pxstar_samples=deepcopy(pxstar_samples),
                pystar_samples=deepcopy(pystar_samples),
                samples_global_ystar=deepcopy(samples_global_ystar),
                samples_global_xstar=deepcopy(samples_global_xstar),
            )
        else:
            # USING MANUAL CEI
            inputs_acq_dict, improvs_dict = self._evaluate_acquisition_functions(
                t,
                current_best_global_target,
                it,
                posterior=deepcopy(self.posterior),
                graphs=self.graphs,
            )

        # Best exploration set based on acquired target-values [best_es == set of all manipulative varibles for BO and ABO]
        best_es = eval("max")(self.y_acquired, key=self.y_acquired.get)

        # Get the correspoding values for this intervention set
        if method == "ABO":
            # Discard the time dimension
            self.corresponding_x[best_es] = self.corresponding_x[best_es][:, :-1]
        new_interventional_data_x = self.corresponding_x[best_es]
        self._check_new_point(best_es, t)

        # Get the correspoding outcome values for this intervention set
        all_variables_new = self.target_functions[t][best_es](
            current_target=target,
            intervention_levels=squeeze(new_interventional_data_x),
            assigned_blanket=assigned_blanket,
        )

        all_variables_new_noiseless = self.target_functions_noiseless[t][best_es](
            current_target=target,
            intervention_levels=squeeze(new_interventional_data_x),
            assigned_blanket=assigned_blanket,
        )

        y_new = float(all_variables_new[self.base_target_variable].squeeze())
        y_new_noiseless = float(all_variables_new_noiseless[self.base_target_variable].squeeze())

        if self.debug_mode:
            print("Selected set:", best_es)
            print("Intervention value:", new_interventional_data_x)
            print("Outcome:", y_new_noiseless)

        self.posterior = update_posterior_interventional(
            graphs=self.graphs,
            posterior=self.posterior,
            intervened_var=best_es,
            all_emission_fncs=self.all_sem_emit_fncs,
            interventional_samples=all_variables_new, #pass noisy data
            total_timesteps=self.T,
        )

        # Update interventional data
        self._get_updated_interventional_data(new_interventional_data_x, y_new, best_es, t)

        # Evaluate cost of intervention
        self.per_trial_cost[t].append(
            total_intervention_cost(best_es, self.cost_functions, self.interventional_data_x[t][best_es],)
        )

        # Store local optimal exploration set corresponding intervention levels
        self.outcome_values[t].append(y_new_noiseless)
        self.optimal_outcome_values_during_trials[t].append(eval(self.task)(y_new_noiseless, current_best_global_target))

        # Store the intervention
        if len(new_interventional_data_x.shape) != 2:
            self.optimal_intervention_levels[t][best_es][it] = make_column_shape_2D(new_interventional_data_x)
        else:
            self.optimal_intervention_levels[t][best_es][it] = new_interventional_data_x

        # Store the currently best intervention set
        self.sequence_of_interventions_during_trials[t].append(best_es)

        #  Update the best_es BO model
        self._update_bo_model(temporal_index=t, exploration_set=best_es,it=it)

        # Causal prior now updated with new posterior !!!
        self._update_sufficient_statistics(
            target=target,
            temporal_index=t,
            dynamic=False,
            assigned_blanket=self.empty_intervention_blanket,
            all_updated_sem=self.all_sem_hat,
        )
        # TODO: SUBPOTIMAL
        if all(value is not None for value in self.bo_model[t].values()):
            self.arm_distribution = update_arm_dist(
                self.arm_distribution, self.bo_model, self.interventional_grids, t, self.task, self.arm_mapping_es_to_num
            )

        if self.debug_mode:
            self._plot_surrogate_model_first(t)

        # if (it == 10 or it == 9) and not all(value is None for value in self.bo_model[t].values()):
        #     print(">>> Results of optimization")
        #     self._plot_surrogate_model(t, it, inputs_acq_dict, improvs_dict, pystar_samples, pxstar_samples, samples_global_ystar, samples_global_xstar, self.arm_mapping_es_to_num)
        #     print(
        #         "### Optimized model: ###", best_es, self.bo_model[t][best_es].model,
        #     )
