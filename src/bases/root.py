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
from src.bayes_opt.cost_functions import define_costs, total_intervention_cost
from src.utils.gp_utils import update_sufficient_statistics_hat
from src.utils.sequential_intervention_functions import (
    evaluate_target_function,
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
    initialise_interventional_objects,
    initialise_DCBO_parameters_and_objects_filtering,
    initialise_global_outcome_dict_new,
    initialise_optimal_intervention_level_list,
    make_column_shape_2D,
)


class Root:
    """
    Base class with common operations, variables and functions for all BO methods.
    """

    def __init__(
        self,
        random_state,
        G: str,
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

        assert isinstance(G, MultiDiGraph)
        self.T = int(list(G.nodes())[-1].split("_")[-1]) + 1  # Total time-steps in DAG
        G.T = self.T
        self.G = G
        self.sorted_nodes = {val: ix for ix, val in enumerate(topological_sort(G))}
        self.debug_mode = debug_mode
        # Number of optimization restart for GPs
        self.n_restart = n_restart
        self.online = online
        self.use_mc = use_mc

        self.observational_samples = observation_samples
        self.base_target_variable = base_target_variable  # This has to be reflected in the CGM
        self.index_name = 0
        self.number_of_trials = number_of_trials

        self.random_state = random_state

        #  Parents of all nodes.  NOT USED !
        # self.node_pars = {node: None for node in G.nodes}
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
        # assert self.manipulative_variables == list(intervention_domain.keys())
        if exploration_sets:
            assert isinstance(exploration_sets, list)
            self.exploration_sets = exploration_sets
        else:
            # When the only intervention is on the parents of the target variable
            self.exploration_sets = [tuple(self.manipulative_variables)]

        # Extract all target variables from the causal graphical model
        self.all_target_variables = list(filter(lambda k: self.base_target_variable in k, self.G.nodes))

        # Get the interventional grids
        self.interventional_grids = get_interventional_grids(
            self.exploration_sets, intervention_domain, size_intervention_grid=num_anchor_points
        )

        # Objective function params
        self.bo_model = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}
        # Target functions for Bayesian optimisation - ground truth
        self.target_functions = deepcopy(self.bo_model)
        self.target_functions_noiseless = deepcopy(self.bo_model)
        # Store true objective function
        self.ground_truth = ground_truth
        # Number of points where to evaluate acquisition function
        self.num_anchor_points = num_anchor_points
        # Assigned during optimisation
        self.mean_function = deepcopy(self.bo_model)
        self.variance_function = deepcopy(self.bo_model)
        # Store the dict for mean and var values computed in the acquisition function
        self.mean_dict_store = {t: {es: {} for es in self.exploration_sets} for t in range(self.T)}
        self.var_dict_store = deepcopy(self.mean_dict_store)
        # For logging
        self.sequence_of_interventions_during_trials = [[] for _ in range(self.T)]
        # CHANGING FOR MULTIPLE INIT DATA:
        # Initial optimal solutions
        # if intervention_samples:
        #     # Provide initial interventional data
        #     (
        #         initial_optimal_sequential_intervention_sets,
        #         initial_optimal_target_values,
        #         initial_optimal_sequential_intervention_levels,
        #         self.interventional_data_x,
        #         self.interventional_data_y,
        #     ) = initialise_interventional_objects(
        #         self.exploration_sets,
        #         intervention_samples,
        #         self.base_target_variable,
        #         self.T,
        #         self.task,
        #         index_name=0,
        #         nr_interventions=None,  # There are interventions, we just don't sub-sample them.
        #     )
        # else:
        #     # No initial interventional data
        #     initial_optimal_sequential_intervention_sets = [choice(self.exploration_sets)] + (self.T - 1) * [None]
        #     initial_optimal_target_values = self.T * [None]
        #     initial_optimal_sequential_intervention_levels = self.T * [None]
        #     self.interventional_data_x = deepcopy(self.bo_model)
        #     self.interventional_data_y = deepcopy(self.bo_model)
        # Initial optimal solutions
        if intervention_samples:
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


        assert (
            len(initial_optimal_sequential_intervention_levels)
            == len(initial_optimal_target_values)
            == len(initial_optimal_sequential_intervention_levels)
            == self.T
        )

        # Dict indexed by the global exploration sets, stores the best
        self.outcome_values = initialise_global_outcome_dict_new(self.T, initial_optimal_target_values, self.blank_val)
        self.noisy_outcome_values = deepcopy(self.outcome_values)
        self.optimal_outcome_values_during_trials = [[] for _ in range(self.T)]
        self.noisy_optimal_outcome_values_during_trials = deepcopy(self.optimal_outcome_values_during_trials)
        
        self.optimal_intervention_levels = initialise_optimal_intervention_level_list(
            self.T,
            self.exploration_sets,
            initial_optimal_sequential_intervention_sets,
            initial_optimal_sequential_intervention_levels,
            number_of_trials,
        )
        self.best_initial_es = initial_optimal_sequential_intervention_sets[0]  # 0 indexes the first time-step

        for temporal_index in range(self.T):
            for es in self.exploration_sets:
                self.target_functions[temporal_index][es] = evaluate_target_function(
                    random_state=self.random_state,
                    initial_structural_equation_model= self.true_initial_sem,
                    structural_equation_model=self.true_sem,
                    graph=self.G,
                    exploration_set=es,
                    all_vars=self.observational_samples.keys(),
                    T=self.T,
                    noisy=True,
                )

                self.target_functions_noiseless[temporal_index][es] = evaluate_target_function(
                    random_state=self.random_state,
                    initial_structural_equation_model= self.true_initial_sem,
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

    def node_parents(self, node: str, temporal_index: int = None) -> tuple:
        """
        Returns the parents of this node with optional filtering on the time-index.

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
        if temporal_index is not None:
            #  This return has to have this complex form because the fitted SEM functions expect multivariate inputs in a specific order (the topological order) of the nodes. Hence the additional sorting.
            res = tuple(
                sorted(
                    filter(lambda x: x.endswith(str(temporal_index)), self.G.predecessors(node)),
                    key=self.sorted_nodes.get,
                )
            )
            if not res:
                return res

            for k in self.sem_emit_fncs[temporal_index].keys():
                if len(k) == 2:
                    if k[-1] == node and res == k[0]:
                        return k
            return res
        else:
            return tuple(self.G.predecessors(node))

    def _get_sem_emit_obs(
        self, t: int, pa: tuple, t_index_data: int = None
    ) -> Tuple[Union[None, ndarray], Union[None, ndarray]]:

        if t_index_data is not None:
            assert t_index_data - 1 == t, (t_index_data, t)
            #  Use past conditional
            t = t_index_data

        if len(pa) == 2 and pa[0] == None:
            # Source node
            pa_y = pa[1].split("_")[0]
            xx = make_column_shape_2D(self.observational_samples[pa_y][t])
            self.sem_emit_fncs[t][pa_y] = KernelDensity(kernel="gaussian").fit(xx)

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

    def _plot_surrogate_model(self, temporal_index):
        # Plot model
        for es in self.exploration_sets:
            if len(es) == 1:
                inputs = np.asarray(self.interventional_grids[es])

                if self.bo_model[temporal_index][es] is not None:
                    mean, var = self.bo_model[temporal_index][es].predict(self.interventional_grids[es])
                    print("\n\t\t[1] The BO model exists for ES: {} at t == {}.\n".format(es, temporal_index))
                    print("Assigned blanket", self.assigned_blanket)
                # else:
                #     mean = self.mean_function[temporal_index][es](self.interventional_grids[es])
                #     var = self.variance_function[temporal_index][es](self.interventional_grids[es]) + np.ones_like(
                #         self.variance_function[temporal_index][es](self.interventional_grids[es])
                #     )
                #     print("\n\t\t[0] The BO model does not exists for ES: {} at t == {}.\n".format(es, temporal_index))
                #     print("Assigned blanket", self.assigned_blanket)
                else:
                    return
                var = 1.96 * np.sqrt(var)
                true = make_column_shape_2D(self.ground_truth[temporal_index][es])

                if (
                    self.interventional_data_x[temporal_index][es] is not None
                    and self.interventional_data_y[temporal_index][es] is not None
                ):
                    plt.scatter(
                        self.interventional_data_x[temporal_index][es], self.interventional_data_y[temporal_index][es],
                        marker='D',c='m',s=80,
                    )

                plt.scatter(
                    self.observational_samples[es[0]][temporal_index],
                    self.observational_samples["Y"][temporal_index],
                    c="k",
                    s=50,
                )

                plt.fill_between(inputs[:, 0], (mean - var)[:, 0], (mean + var)[:, 0], edgecolor=(0. , 0.4, 1., 0.99) , facecolor=(0. , 0.4, 1., 0.5))
                plt.plot(
                    inputs, mean, "b", label="Causal prior for $do{}$ ".format(es),
                )
                plt.plot(inputs, true, "r", lw=2.5, label="True causal effect")
                plt.legend()
                plt.show()

    def _update_opt_params(self, it: int, temporal_index: int, best_es: tuple) -> None:

        # When observed append previous optimal values for logs
        # Outcome values at previous step
        self.outcome_values[temporal_index].append(self.outcome_values[temporal_index][-1])
        self.noisy_outcome_values[temporal_index].append(self.noisy_outcome_values[temporal_index][-1])

        if it == 0:
            # Special case for first time index
            # Assign outcome values that is the same as the initial value in first trial
            self.optimal_outcome_values_during_trials[temporal_index].append(self.outcome_values[temporal_index][-1])
            self.noisy_optimal_outcome_values_during_trials[temporal_index].append(self.noisy_outcome_values[temporal_index][-1])

            if self.interventional_data_x[temporal_index][best_es] is None:
                self.optimal_intervention_levels[temporal_index][best_es][it] = nan

            self.per_trial_cost[temporal_index].append(0.0)

        elif it > 0:
            # Get previous one cause we are observing thus we no need to recompute it
            self.optimal_outcome_values_during_trials[temporal_index].append(
                self.optimal_outcome_values_during_trials[temporal_index][-1]
            )
            self.noisy_optimal_outcome_values_during_trials[temporal_index].append(
                self.noisy_optimal_outcome_values_during_trials[temporal_index][-1]
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

    def _safe_optimization(self, temporal_index, exploration_set, bound_var=1e-02, bound_len=20.0):
        if self.bo_model[temporal_index][exploration_set].model.kern.variance[0] < bound_var:
            self.bo_model[temporal_index][exploration_set].model.kern.variance[0] = 1.0

        if self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] > bound_len:
            self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] = 1.0

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
        self, target: str, temporal_index: int, dynamic: bool, assigned_blanket: dict, updated_sem
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
        updated_sem : OrderedDict
            Structural equation model.
        """

        # Check which current target we are dealing with, and in consequence where we are in time
        target_variable, target_temporal_index = target.split("_")
        assert int(target_temporal_index) == temporal_index

        for es in self.exploration_sets:
            if self.estimate_sem:
                (
                    self.mean_function[temporal_index][es],
                    self.variance_function[temporal_index][es],
                ) = update_sufficient_statistics_hat(
                    temporal_index=temporal_index,
                    target_variable=target_variable,
                    exploration_set=es,
                    sem_hat=updated_sem,
                    node_parents=self.node_parents,
                    dynamic=dynamic,
                    assigned_blanket=assigned_blanket,
                    mean_dict_store=self.mean_dict_store,
                    var_dict_store=self.var_dict_store,
                )
            # Use true sem
            else:
                raise NotImplementedError("This function has to be updated to reflect recent changes in 'hat' version.")

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
        for i, es_member in enumerate(set(es).intersection(self.manipulative_variables)):
            self.optimal_blanket[es_member][t] = float(
                self.optimal_intervention_levels[t][self.optimal_intervention_sets[t]][best_objective_fnc_value_idx][
                    :, i
                ]
            )

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
        assign_blanket(
            self.true_initial_sem,
            self.true_sem,
            self.assigned_blanket,
            self.optimal_intervention_sets[t],
            self.optimal_intervention_levels[t][self.optimal_intervention_sets[t]][best_objective_fnc_value_idx],
            target=target,
            target_value=self.optimal_blanket[self.base_target_variable][t],
            G=self.G,
        )
        check_blanket(
            self.assigned_blanket, self.base_target_variable, t, self.manipulative_variables,
        )

        # Check optimization results for the current temporal index before moving on
        self._check_optimization_results(t)

    def _per_trial_computations(self, t: int, it: int, target: str, assigned_blanket: dict, method: str = None):

        if self.debug_mode:
            print("\n\n>>>")
            print("Iteration:", it)
            print("<<<\n\n")

            self._plot_surrogate_model(t)

        # Presently find the optimal value of Y_t
        current_best_global_target = eval(self.task)(self.outcome_values[t])
        noisy_current_best_global_target = eval(self.task)(self.noisy_outcome_values[t])


        #  Just to indicate that in this trial we are explicitly intervening in the system
        self.trial_type[t].append("i")

        # Compute acquisition function given the updated BO models for the interventional data. Notice that we use current_global and the costs to compute the acquisition functions.
        self._evaluate_acquisition_functions(t, noisy_current_best_global_target, it) # need to pass noisy best global here

        # Best exploration set based on acquired target-values [best_es == set of all manipulative varibles for BO and ABO]
        best_es = eval("max")(self.y_acquired, key=self.y_acquired.get)

        # Get the correspoding values for this intervention set
        if method == "ABO":
            # Discard the time dimension
            self.corresponding_x[best_es] = self.corresponding_x[best_es][:, :-1]
        new_interventional_data_x = self.corresponding_x[best_es]
        self._check_new_point(best_es, t)

        # Get the correspoding outcome values for this intervention set
        y_new = self.target_functions[t][best_es](
            current_target=target,
            intervention_levels=squeeze(new_interventional_data_x),
            assigned_blanket=assigned_blanket,
        )

        y_new_noiseless = self.target_functions_noiseless[t][best_es](
            current_target=target,
            intervention_levels=squeeze(new_interventional_data_x),
            assigned_blanket=assigned_blanket,
        )

        if self.debug_mode:
            print("Selected set:", best_es)
            print("Intervention value:", new_interventional_data_x)
            print("Outcome:", y_new)

        # Update interventional data
        self._get_updated_interventional_data(new_interventional_data_x, y_new, best_es, t)

        # Evaluate cost of intervention
        self.per_trial_cost[t].append(
            total_intervention_cost(best_es, self.cost_functions, self.interventional_data_x[t][best_es],)
        )

        # Store local optimal exploration set corresponding intervention levels
        self.outcome_values[t].append(y_new_noiseless)
        self.optimal_outcome_values_during_trials[t].append(eval(self.task)(y_new_noiseless, current_best_global_target))

        self.noisy_outcome_values[t].append(y_new)
        self.noisy_optimal_outcome_values_during_trials[t].append(eval(self.task)(y_new, noisy_current_best_global_target))

        # Store the intervention
        if len(new_interventional_data_x.shape) != 2:
            self.optimal_intervention_levels[t][best_es][it] = make_column_shape_2D(new_interventional_data_x)
        else:
            self.optimal_intervention_levels[t][best_es][it] = new_interventional_data_x

        # Store the currently best intervention set
        self.sequence_of_interventions_during_trials[t].append(best_es)

        #  Update the best_es BO model
        self._update_bo_model(t, best_es)

        if self.debug_mode:
            print(">>> Results of optimization")
            self._plot_surrogate_model(t)
            print(
                "### Optimized model: ###", best_es, self.bo_model[t][best_es].model,
            )
