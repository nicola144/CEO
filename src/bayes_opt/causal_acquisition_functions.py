from typing import Tuple, Union
import numpy as np
import scipy.stats
from emukit.core import ParameterSpace
from src.utils.ces_utils import MyGPyModelWrapper, MyKDENew, sample_global_xystar, update_arm_dist_single_model, update_pystar_single_model, to_prob, fake_do_x
from src.utils.ceo_utils import normalize_log
from scipy.stats import entropy
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel
from emukit.bayesian_optimization.interfaces import IEntropySearchModel


from tqdm import tqdm
from copy import deepcopy

class ManualCausalExpectedImprovement(Acquisition):
    def __init__(
        self, current_global_min, task, mean_function, variance_function, previous_variance, jitter: float = float(0),
    ) -> None:
        """
        The improvement when a BO model has not yet been instantiated.

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param mean_function: the mean function for the current DCBO exploration at given temporal index
        :param variance_function: the mean function for the current DCBO exploration at given temporal index
        :param jitter: parameter to encourage extra exploration.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task
        self.previous_variance = previous_variance

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean = self.mean_function(x)

        # adjustment term          #initial kernel variance
        variance = self.previous_variance * np.ones((x.shape[0], 1)) + self.variance_function(
            x
        )  # See Causal GP def in paper

        standard_deviation = np.sqrt(variance.clip(0))
        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))

        return improvement

    @property
    def has_gradients(self) -> bool:
        """
        Returns that this acquisition does not have gradients.
        """
        return False


class CausalExpectedImprovement(Acquisition):
    def __init__(
        self,
        current_global_min,
        task,
        dynamic,
        causal_prior,
        temporal_index,
        model: Union[IModel, IDifferentiable],
        jitter: float = float(0),
    ) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task
        self.dynamic = dynamic
        self.causal_prior = causal_prior
        self.temporal_index = temporal_index

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        # Adding an extra time dimension for ABO
        if self.dynamic and self.causal_prior is False:
            x = np.hstack((x, np.repeat(self.temporal_index, x.shape[0])[:, np.newaxis]))

        mean, variance = self.model.predict(x)

        # Variance is computed with MonteCarlo so we might have some numerical stability
        # This is ensuring that negative values or nan values are not generated
        if np.any(np.isnan(variance)):
            variance[np.isnan(variance)] = 0
        elif np.any(variance < 0):
            variance = variance.clip(0.0001)

        standard_deviation = np.sqrt(variance)

        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """
        # Adding an extra time dimension for ABO
        # Restrict the input space via an additional function
        if self.dynamic and self.causal_prior is False:
            x = np.hstack((x, np.repeat(self.temporal_index, x.shape[0])[:, np.newaxis]))

        mean, variance = self.model.predict(x)

        # Variance is computed with MonteCarlo so we might have some numerical stability
        # This is ensuring that negative values or nan values are not generated
        if np.any(np.isnan(variance)):
            variance[np.isnan(variance)] = 0
        elif np.any(variance < 0):
            variance = variance.clip(0.0001)

        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == "min":
            improvement = standard_deviation * (u * cdf + pdf)
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            improvement = -(standard_deviation * (u * cdf + pdf))
            dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(
    x: np.array, mean: np.array, standard_deviation: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf


class CausalEntropySearch(Acquisition):
    def __init__(self, all_sem_hat, all_emit_fncs, graphs, node_parents, current_posterior, es, model: Union[IModel, IEntropySearchModel], space: ParameterSpace, interventional_grid, kde,
                 es_num_arm_mapping, num_es_arm_mapping, arm_distr, seed, task, all_xstar, all_ystar,
                 samples_global_ystar, samples_global_xstar, do_cdcbo=False) -> None:
        """
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        self.es, self.model, self.space, self.grid, self.pre_kde, self.es_num_mapping, self.num_es_arm_mapping, self.prev_arm_distr, self.seed, self.task = es, model, space, interventional_grid, kde, es_num_arm_mapping, num_es_arm_mapping, arm_distr, seed, task
        self.init_posterior = current_posterior
        self.node_parents = node_parents
        self.graphs = graphs
        self.all_sem_hat = all_sem_hat
        self.all_emit_fncs = all_emit_fncs
        self.prev_all_ystar = all_ystar
        self.prev_all_xstar = all_xstar
        self.prev_global_samples_ystar = samples_global_ystar
        self.prev_global_samples_xstar = samples_global_xstar
        self.do_cdcbo = do_cdcbo

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the predicted change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we evaluate x.
        :param x: points where the acquisition is evaluated.
        """

        # Make new aquisition points

        grid = self.grid if len(self.es) == 1 else x  # this was wrong

        initial_entropy = self.pre_kde.entropy # A scalar really
        initial_graph_entropy = entropy(normalize_log(self.init_posterior))
        n_fantasies = 5 # N. of fantasy observations
        n_acquisitions = x.shape[0]  # could  choose a subset of them to reduce computation

        n_samples_mixture = self.prev_global_samples_ystar.shape[0]
        new_entropies = np.empty((n_acquisitions,))  # first dimension is n anchor points
        new_entropies_opt = np.empty((n_acquisitions,))  # first dimension is n anchor points
        new_entropies_graph = np.empty((n_acquisitions,))  # first dimension is n anchor points

        # Stores the new samples from the updated p(y* | D, (x,y)). 
        new_samples_global_ystar_list = np.empty((n_acquisitions, n_fantasies, n_samples_mixture)) # TODO: check shapes

        # Keeping track of these just because of plotting later
        updated_models_list = [[] for _ in range(n_acquisitions)]  # shape will be n_acquisitions x n_fantasies

        const = np.pi ** -0.5

        # Approx integral with GQ
        xx, w = np.polynomial.hermite.hermgauss(n_fantasies)

        curr_normalized_graph = normalize_log(self.init_posterior)

        if curr_normalized_graph[0] > 0.90:
            print('graph is found')
            # If you found the graph, optimize
            for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                # Get samples from p(y | D, do(x) )
                if single_x.shape[0] == 1:
                    x_inp = single_x.reshape(-1, 1)
                else:
                    x_inp = single_x.reshape(1, -1)

                m, v = self.model.predict(x_inp)
                m, v = m.squeeze(), v.squeeze()

                # Fantasy samples from sigma points xx
                fantasy_ys = 2 ** 0.5 * np.sqrt(v) * xx + m

                new_entropies_unweighted = np.empty((n_fantasies,))

                for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                    updated_model = deepcopy(self.model)
                    prevx, prevy = updated_model.get_X(), updated_model.get_Y()

                    # squeezed_temp = prevx.squeeze()
                    # if len(squeezed_temp.shape) == 1:
                    #     prevx = prevx.reshape(-1, 1)

                    # if len(self.es) == 2:
                    #     print('hi')

                    tempx = np.concatenate([prevx, x_inp])

                    fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(-1, 1)

                    tempy = np.vstack([prevy, fantasy_y])

                    updated_model.set_XY(tempx, tempy)

                    # Keeping track of them just for plotting ie. debugging reasons
                    updated_models_list[id_acquisition].append(updated_model)

                    # Arm distr gets updated only because model gets updated
                    # start = time.time()#&&
                    new_arm_dist = update_arm_dist_single_model(deepcopy(self.prev_arm_distr), self.es, updated_model,
                                                                grid, self.task, self.es_num_mapping, self.space,
                                                                self.seed)
                    # end = time.time() #&&
                    # print("update_arm_dist_single_model    took: ", end - start) #&&

                    # Use this to build p(y*, x* | D, (x,y) )
                    # start = time.time() #&&
                    pystar_samples, pxstar_samples = update_pystar_single_model(arm_mapping=self.es_num_mapping,
                                                                                es=self.es,
                                                                                bo_model=updated_model,
                                                                                inputs=grid, task="min",
                                                                                all_xstar=self.prev_all_xstar,
                                                                                all_ystar=deepcopy(self.prev_all_ystar),
                                                                                space=self.space,
                                                                                seed=self.seed)

                    # end = time.time() #&&
                    # print("update_pystar_single_model    took: ", end - start) #&&

                    # start = time.time() #&&
                    new_samples_global_ystar, new_samples_global_xstar = sample_global_xystar(
                        n_samples_mixture=n_samples_mixture,
                        all_ystar=pystar_samples,
                        all_xstar=pxstar_samples,
                        arm_dist=to_prob(new_arm_dist, #checked , this works for min
                                         self.task),
                        arm_mapping_n_to_es=self.num_es_arm_mapping)

                    # end = time.time() #&&
                    # print("sample_global_xystar    took: ", end - start) #&&

                    # start = time.time() #&&

                    new_kde = MyKDENew(new_samples_global_ystar)
                    try:
                        new_kde.fit()
                    except RuntimeError:
                        new_kde.fit(bw=0.5)

                    new_entropy_ystar = new_kde.entropy  # this can be neg. as it's differential entropy

                    new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                    new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = new_samples_global_ystar

                    # end = time.time()#&&
                    # print("end    took: ", end - start)#&&

                # GQ average
                new_entropies[id_acquisition] = np.sum(w * const * new_entropies_unweighted)

                # Plotting
                # if not len(self.es) > 1:
                #     self.model.model.plot()
                #     plt.title('Pre-fake-intervention model')
                #     # plt.show()
                #     for modell in updated_models_list[0]:
                #         modell.model.plot()
                #         plt.title('Post-fake-intervention model')
                #         plt.show()

                # Remove  when debugging with  batch
            assert new_entropies.shape == (n_acquisitions,) or new_entropies == (n_acquisitions, 1)
            # Represents the improvement in (averaged over fantasy observations!) entropy (it's good if it lowers)
            # It can be negative.
            entropy_changes = initial_entropy - new_entropies

        else:
            print('graph is not found')

            if not self.do_cdcbo:
                # Keep finding graph and optimize JOINTLY
                intervened_vars = [s for s in self.es]
                # Calc updated graph entropy
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    updated_posterior = fake_do_x(
                        x=x_inp,
                        node_parents=self.node_parents,
                        graphs=self.graphs,
                        log_graph_post=deepcopy(self.init_posterior),
                        intervened_vars=intervened_vars,
                        all_emission_fncs=self.all_emit_fncs,
                        all_sem=self.all_sem_hat
                              )
                    new_entropies_graph[id_acquisition] = entropy(normalize_log(updated_posterior))

                entropy_changes_graph = initial_graph_entropy - new_entropies_graph

                # Optimization part
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):

                    # Get samples from p(y | D, do(x) )
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    m, v = self.model.predict(x_inp)
                    m, v = m.squeeze(), v.squeeze()

                    # Fantasy samples from sigma points xx
                    fantasy_ys = 2 ** 0.5 * np.sqrt(v) * xx + m

                    new_entropies_unweighted = np.empty((n_fantasies,))

                    for n_fantasy, fantasy_y in enumerate(fantasy_ys):

                        updated_model = deepcopy(self.model)
                        prevx, prevy = updated_model.get_X(), updated_model.get_Y()

                        # squeezed_temp = prevx.squeeze()
                        # if len(squeezed_temp.shape) == 1:
                        #     prevx = prevx.reshape(-1, 1)

                        # if len(self.es) == 2:
                        #     print('hi')

                        tempx = np.concatenate([prevx, x_inp])

                        fantasy_y, prevy = fantasy_y.reshape(-1, 1), prevy.reshape(-1, 1)

                        tempy = np.vstack([prevy, fantasy_y])

                        updated_model.set_XY(tempx, tempy)

                        # Keeping track of them just for plotting ie. debugging reasons
                        updated_models_list[id_acquisition].append(updated_model)

                        # Arm distr gets updated only because model gets updated
                        # start = time.time()#&&
                        new_arm_dist = update_arm_dist_single_model(deepcopy(self.prev_arm_distr), self.es, updated_model,
                                                                    grid, self.task, self.es_num_mapping, self.space,
                                                                    self.seed)
                        # end = time.time() #&&
                        # print("update_arm_dist_single_model    took: ", end - start) #&&

                        # Use this to build p(y*, x* | D, (x,y) )
                        # start = time.time() #&&
                        pystar_samples, pxstar_samples = update_pystar_single_model(arm_mapping=self.es_num_mapping,
                                                                                    es=self.es,
                                                                                    bo_model=updated_model,
                                                                                    inputs=grid, task="min",
                                                                                    all_xstar=self.prev_all_xstar,
                                                                                    all_ystar=deepcopy(self.prev_all_ystar),
                                                                                    space=self.space,
                                                                                    seed=self.seed)

                        # end = time.time() #&&
                        # print("update_pystar_single_model    took: ", end - start) #&&

                        # start = time.time() #&&
                        new_samples_global_ystar, new_samples_global_xstar = sample_global_xystar(
                            n_samples_mixture=n_samples_mixture,
                            all_ystar=pystar_samples,
                            all_xstar=pxstar_samples,
                            arm_dist=to_prob(new_arm_dist, #checked , this works for min
                                             self.task),
                            arm_mapping_n_to_es=self.num_es_arm_mapping)

                        # end = time.time() #&&
                        # print("sample_global_xystar    took: ", end - start) #&&

                        # start = time.time() #&&

                        new_kde = MyKDENew(new_samples_global_ystar)
                        try:
                            new_kde.fit()
                        except RuntimeError:
                            new_kde.fit(bw=0.5)

                        new_entropy_ystar = new_kde.entropy  # this can be neg. as it's differential entropy

                        new_entropies_unweighted[n_fantasy] = new_entropy_ystar
                        new_samples_global_ystar_list[id_acquisition, n_fantasy, :] = new_samples_global_ystar

                        # end = time.time()#&&
                        # print("end    took: ", end - start)#&&

                    # GQ average
                    new_entropies_opt[id_acquisition] = np.sum(w * const * new_entropies_unweighted)

                    # Plotting
                    # if not len(self.es) > 1:
                    #     self.model.model.plot()
                    #     plt.title('Pre-fake-intervention model')
                    #     # plt.show()
                    #     for modell in updated_models_list[0]:
                    #         modell.model.plot()
                    #         plt.title('Post-fake-intervention model')
                    #         plt.show()

                    # Remove  when debugging with  batch
                    # assert new_entropies.shape == (n_acquisitions,) or new_entropies == (n_acquisitions, 1)
                    # Represents the improvement in (averaged over fantasy observations!) entropy (it's good if it lowers)
                    # It can be negative.

                entropy_changes_opt = initial_entropy - new_entropies_opt

                entropy_changes = entropy_changes_graph + entropy_changes_opt
            else:
                # CD-CBO: only graph !
                # Keep finding graph and optimize jointly
                intervened_vars = [s for s in self.es]
                # Calc updated graph entropy
                for id_acquisition, single_x in tqdm(enumerate(x[:n_acquisitions])):
                    if single_x.shape[0] == 1:
                        x_inp = single_x.reshape(-1, 1)
                    else:
                        x_inp = single_x.reshape(1, -1)

                    updated_posterior = fake_do_x(
                        x=x_inp,
                        node_parents=self.node_parents,
                        graphs=self.graphs,
                        log_graph_post=deepcopy(self.init_posterior),
                        intervened_vars=intervened_vars,
                        all_emission_fncs=self.all_emit_fncs,
                        all_sem=self.all_sem_hat
                    )
                    new_entropies[id_acquisition] = entropy(normalize_log(updated_posterior))

                entropy_changes = initial_graph_entropy - new_entropies

            # end of inner if
        # end of outer if

        # Just in case any are negative, shift all, preserving the total order.
        if np.any(entropy_changes < 0.):
            smallest = np.absolute(np.min(entropy_changes))
            entropy_changes = entropy_changes + smallest

        # Plotting
        # fig, ax = plt.subplots(1, 1)
        # kwargs = {'levels': np.arange(0, 0.15, 0.01)}
        #
        # sns.kdeplot(self.prev_global_samples_ystar.squeeze(), ax=ax, label='prev global', alpha=0.22, **kwargs)
        # for i, j in zip(range(new_samples_global_ystar_list.shape[0]), range(new_samples_global_ystar_list.shape[1])):
        #     sns.kdeplot(new_samples_global_ystar_list[i, j, :].squeeze(), ax=ax,  shade=True,  alpha=0.22, **kwargs)
        #
        # plt.legend()
        # plt.title("P(y stars) by \"doing\" "+ str(self.es) +  "Best change: " + str(np.max(entropy_changes)))#+ "Acquis. and entropy changes: " + str(x.tolist()) + " " +  str(entropy_changes.tolist()) )
        # plt.show()

        print("Entropy changes for " + str(self.es) + ": ")
        print(str(entropy_changes.tolist()))
        assert entropy_changes.shape[0] == x.shape[0]

        # ax1[0][1].fill_between(inputs[:, 0], (mean - 2. * var)[:, 0], (mean + 2. * var)[:, 0], alpha=0.15)
        # ax1[0][1].plot(
        #     inputs,
        #     mean,
        #     c='b',
        #     label="$do{}$".format(es),
        #     lw=5.,
        # )

        return entropy_changes

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
