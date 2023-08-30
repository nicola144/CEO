from typing import Callable

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.core import Mapping
import GPy
from GPy.core.parameterization import priors
from GPy.kern.src.rbf import RBF
from GPy.models import GPRegression
from src.bases.ceo_base import BaseClassCeo
from src.bayes_opt.causal_kernels import CausalRBF
from src.bayes_opt.intervention_computations import evaluate_acquisition_function
from src.utils.sem_utils.emissions import fit_sem_emit_fncs
from src.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    standard_mean_function,
    zero_variance_adjustment,
    normalize_log
)
from tqdm import trange

from src.utils.ceo_utils import update_posterior_observational,update_posterior_interventional
from src.utils.ces_utils import MyGPyModelWrapper
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from matplotlib import pyplot as plt

class CEO(BaseClassCeo):
    def __init__(
        self,
        # G: str,
        graphs: list,  # NEW
        init_posterior: list,  # NEW
        sem: classmethod,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        intervention_samples_noiseless: dict, # NEW
        exploration_sets: list,
        number_of_trials: int,
        base_target_variable: str,
        ground_truth: list = None,
        estimate_sem: bool = True,
        task: str = "min",
        n_restart: int = 1,
        cost_type: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        concat: bool = False,
        optimal_assigned_blankets: dict = None,
        n_obs_t: int = None,
        hp_i_prior: bool = True,
        num_anchor_points=2,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
        random_state = None,
        do_cdcbo = False,
    ):
        args = {
            # "G": G,
            "graphs": graphs, #NEW
            "init_posterior": init_posterior, #NEW
            "sem": sem,
            "make_sem_estimator": make_sem_estimator,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "intervention_samples_noiseless":intervention_samples_noiseless,
            "exploration_sets": exploration_sets,
            "estimate_sem": estimate_sem,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "use_mc": use_mc,
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
            "random_state": random_state,
            "do_cdcbo":do_cdcbo,
        }
        super().__init__(**args)

        self.concat = concat
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.n_obs_t = n_obs_t
        self.hp_i_prior = hp_i_prior
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points

        # Fit Gaussian processes to emissions . One for each graph
        self.all_sem_emit_fncs = []
        for g in self.graphs:
            sem_emit_fncs = fit_sem_emit_fncs(g, self.observational_samples)
            self.all_sem_emit_fncs.append(sem_emit_fncs)

        # UPDATING POSTERIOR
        self.posterior = np.log(np.asarray(init_posterior))
        self.all_posteriors = []
        self.all_posteriors.append(normalize_log(deepcopy(self.posterior)))
        # self.posterior = update_posterior_observational(graphs=self.graphs,
        #                                                 posterior=self.posterior,
        #                                                 all_emission_fncs=self.all_sem_emit_fncs,
        #                                                 new_observational_samples=self.observational_samples,
        #                                                 total_timesteps=self.T,
        #                                                 it=0)
        # self.all_posteriors.append(normalize_log(deepcopy(self.posterior)))

        if self.initial_intervention_samples:
            for k, interv_sample in self.initial_intervention_samples.items():
                # TODO
                self.posterior = update_posterior_interventional(graphs=graphs,
                                                                 posterior=self.posterior,
                                                                 intervened_var=k,
                                                                 all_emission_fncs=self.all_sem_emit_fncs,
                                                                 interventional_samples=interv_sample,
                                                                 total_timesteps=self.T,
                                                                 it=0)

        self.all_posteriors.append(normalize_log(deepcopy(self.posterior)))
        # Convert observational samples to dict of temporal lists. We do this because at each time-index we may have a different number of samples. Because of this, samples need to be stored one lists per time-step.
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def run(self):

        if self.debug_mode:
            assert self.ground_truth is not None, "Provide ground truth to plot surrogate models"

        # Walk through the graph, from left to right, i.e. the temporal dimension
        for temporal_index in trange(self.T, desc="Time index"):

            # Evaluate each target
            target = self.all_target_variables[temporal_index]
            # Check which current target we are dealing with, and in initial_sem sequence where we are in time
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Updating the observational and interventional data based on the online and concat options.
            self._update_observational_data(temporal_index=temporal_index)
            self._update_interventional_data(temporal_index=temporal_index)

            #  Online run option
            # Not really gonnna happen in CEO
            if temporal_index > 0 and (self.online or isinstance(self.n_obs_t, list)):
                print('should not go here')
                exit()
                self._update_sem_emit_fncs(temporal_index)

            # Get blanket to compute y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in tqdm(range(self.number_of_trials)):

                if it == 0:

                    self.trial_type[temporal_index].append("o")  # For 'o'bserve
                    # For CEO a list of sem_hat's
                    all_sem_hat = []
                    for g, emission_fncs_for_g in zip(self.graphs,self.all_sem_emit_fncs):
                        sem_hat = self.make_sem_hat(G=g, emission_fncs=emission_fncs_for_g,)
                        all_sem_hat.append(sem_hat)

                    self.all_sem_hat = all_sem_hat
                    # Create mean functions and var functions given the observational data. This updates the prior.
                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=False,
                        assigned_blanket=self.empty_intervention_blanket,
                        all_updated_sem=all_sem_hat,
                    )
                    # Update optimisation related parameters
                    self._update_opt_params(it, temporal_index, best_es)

                else:

                    # Surrogate models
                    if self.trial_type[temporal_index][-1] == "o":
                        # This is initial data?
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index=temporal_index, exploration_set=es, it=it)
                        if self.debug_mode:
                            self._plot_surrogate_model_first(temporal_index)

                    # This function runs the actual computation -- calls are identical for all methods
                    # Some changes are done in CEO
                    self._per_trial_computations(temporal_index, it, target, assigned_blanket)

                    self.all_posteriors.append(normalize_log(deepcopy(self.posterior)))
                    print(' \t \n CURRENT POSTERIOR: \t  ', normalize_log(deepcopy(self.posterior)))

            # Post optimisation assignments (post this time-index that is)
            # self._post_optimisation_assignments(target, temporal_index)

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it, posterior, graphs, kde_globalystar=None, pxstar_samples=None, pystar_samples=None, samples_global_ystar=None, samples_global_xstar=None):
        inputs_acq_dict = {es: None for es in self.exploration_sets}
        improvs_dict = {es: None for es in self.exploration_sets}

        for es in self.exploration_sets:
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None
                # What is this ?
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self.mean_function[temporal_index][es] = standard_mean_function
                    self.variance_function[temporal_index][es] = zero_variance_adjustment

            # We evaluate this function IF there is interventional data at this time index
            if self.seed_anchor_points is None:
                seed_to_pass = None
            else:
                seed_to_pass = int(self.seed_anchor_points * (temporal_index + 1) * it)

            (self.y_acquired[es], self.corresponding_x[es], inputs_acq_dict[es], improvs_dict[es]) = evaluate_acquisition_function(
                self.intervention_exploration_domain[es],
                bo_model,
                partial(self.mean_function, t=temporal_index,exploration_set=es, posterior=normalize_log(deepcopy(posterior))),
                partial(self.variance_function, t=temporal_index,exploration_set=es, posterior=normalize_log(deepcopy(posterior))),
                current_best_global_target,
                es,
                self.cost_functions,
                self.task,
                self.base_target_variable,
                dynamic=False,
                causal_prior=True,
                temporal_index=temporal_index,
                previous_variance=1.0,
                num_anchor_points=self.num_anchor_points,
                sample_anchor_points=self.sample_anchor_points,
                seed_anchor_points=seed_to_pass,
                posterior=posterior,
                node_parents=self.node_parents,
                graphs=graphs,
                all_emit_fncs=self.all_sem_emit_fncs,
                all_sem_hat=self.all_sem_hat,
                kde_globalystar=kde_globalystar,
                pxstar_samples=pxstar_samples,
                pystar_samples=pystar_samples,
                samples_global_ystar=samples_global_ystar,
                samples_global_xstar=samples_global_xstar,
                interventional_grid=self.interventional_grids[es],
                arm_distribution=deepcopy(self.arm_distribution),
                arm_mapping_es_to_num=self.arm_mapping_es_to_num,
                arm_mapping_num_to_es=self.arm_mapping_num_to_es,
                do_cdcbo=self.do_cdcbo,
            )
        # Careful with correct indentation here !
        return inputs_acq_dict, improvs_dict

    def _update_interventional_data(self, temporal_index):

        if temporal_index > 0 and self.concat:
            print('should not go here')
            exit()
            for var in self.interventional_data_x[0].keys():
                self.interventional_data_x[temporal_index][var] = self.interventional_data_x[temporal_index - 1][var]
                self.interventional_data_y[temporal_index][var] = self.interventional_data_y[temporal_index - 1][var]

    def _update_sem_emit_fncs(self, t: int) -> None:
        # Loop over all emission functions in this time-slice
        for pa in self.sem_emit_fncs[t]:
            # Get relevant data for updating emission functions
            xx, yy = self._get_sem_emit_obs(t, pa)
            if xx and yy:
                # Update in-place
                self.sem_emit_fncs[t][pa].set_XY(X=xx, Y=yy)
                self.sem_emit_fncs[t][pa].optimize()

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, it: int, alpha: float = 2, beta: float = 0.5,
    ) -> None:

        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)

        # Specify mean function
        mf = Mapping(input_dim, 1)
        mean_function = partial(self.mean_function, t=temporal_index, exploration_set=exploration_set, posterior=normalize_log(deepcopy(self.posterior)))
        mf.f = mean_function
        var_function = partial(self.variance_function, t=temporal_index, exploration_set=exploration_set, posterior=normalize_log(deepcopy(self.posterior)))

        mf.update_gradients = lambda a, b: None

        # Set kernel
        causal_kernel = CausalRBF(
            input_dim=input_dim,  # Indexed before passed to this function
            variance_adjustment=var_function,  # Variance function here
            lengthscale=1.0,
            variance=1.0,
            ARD=False,
        )

        #
        # # DEBUGGING MODEL
        # model = GPRegression(X=self.interventional_data_x[temporal_index][exploration_set],
        #                      Y=self.interventional_data_y[temporal_index][exploration_set],
        #                      kernel=RBF(input_dim, lengthscale=1., variance=1.),
        #                      noise_var=1e-5)


        #  Set model
        model = GPRegression(
            X=self.interventional_data_x[temporal_index][exploration_set],
            Y=self.interventional_data_y[temporal_index][exploration_set],
            kernel=causal_kernel,
            noise_var=1e-5,
            mean_function=mf,
        )

        # For helthcare
        # n = self.interventional_data_x[temporal_index][exploration_set].shape[0]
        # prior_len = GPy.priors.InverseGamma.from_EV(0.6, 0.05)
        # prior_sigma_f = GPy.priors.InverseGamma.from_EV(1 /  np.sqrt(n), .5)
        # prior_lik = GPy.priors.InverseGamma.from_EV(1, 0.5)

        # model.kern.lengthscale.set_prior(prior_len)
        # model.kern.variance.set_prior(prior_sigma_f)
        # model.likelihood.variance.set_prior(prior_lik)

        # Toy & health
        prior_len = priors.InverseGamma.from_EV(1, 1)
        prior_sigma_f = priors.InverseGamma.from_EV(1, 1)
        prior_lik = priors.InverseGamma.from_EV(1, 0.2)

        model.kern.lengthscale.set_prior(prior_len)
        model.kern.variance.set_prior(prior_sigma_f)
        model.likelihood.variance.set_prior(prior_lik)

        # For epi
        # n = self.interventional_data_x[temporal_index][exploration_set].shape[0]
        # prior_len = GPy.priors.InverseGamma.from_EV(2, 1.)
        # prior_sigma_f = GPy.priors.InverseGamma.from_EV( 2 /  (0.5* np.sqrt(n)), 1)
        # prior_lik = GPy.priors.InverseGamma.from_EV(2., 0.1)
        #
        # model.kern.lengthscale.set_prior(prior_len)
        # model.kern.variance.set_prior(prior_sigma_f)
        # model.likelihood.variance.set_prior(prior_lik)

        # model.likelihood.variance.fix()
        old_seed = np.random.get_state()
        np.random.seed(self.seed)
        model.optimize_restarts(num_restarts=5)
        np.random.set_state(old_seed)


        self.bo_model[temporal_index][exploration_set] = MyGPyModelWrapper(model)
        self._safe_optimization(temporal_index, exploration_set)

        # DEBUG
        # if not len(exploration_set) == 2 and it % 5 == 0:
        #     self.bo_model[temporal_index][exploration_set].model.plot()
        #     plt.show()


    def _get_assigned_blanket(self, temporal_index):
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket

        return assigned_blanket
