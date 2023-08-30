import statsmodels.api as sm
import GPy
from scipy.special import gammainc
import numpy as np
from numpy import random

from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from .sequential_sampling import sequential_sample_from_SEM_hat
from functools import partial
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from scipy.special import softmax
from copy import deepcopy
from src.utils.ceo_utils import update_posterior_interventional

class MyKDENew(sm.nonparametric.KDEUnivariate):
    def __init__(self, *args):
        super().__init__(*args)

    def sample(self, n_samples=1, random_state=None):
        u = np.random.uniform(0, 1, size=n_samples)
        i = (u * self.endog.shape[0]).astype(np.int64)

        # if self.kernel == 'gaussian':
        return np.atleast_2d(np.random.normal(self.endog[i], self.kernel.h)), self.endog[i]


class MyKDE(KernelDensity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = None

    def fit_and_update(self, X):
        self.X = X
        return super().fit(X)

    def predict(self):
        return np.mean(super().sample(n_samples=500)), np.var(super().sample(n_samples=500))

    # def sample(self, n_samples=1, random_state=None):
    #     """Generate random samples from the model.
    #     Currently, this is implemented only for gaussian and tophat kernels.
    #     Parameters
    #     ----------
    #     n_samples : int, default=1
    #         Number of samples to generate.
    #     random_state : int, RandomState instance or None, default=None
    #         Determines random number generation used to generate
    #         random samples. Pass an int for reproducible results
    #         across multiple function calls.
    #         See :term: `Glossary <random_state>`.
    #     Returns
    #     -------
    #     X : array-like of shape (n_samples, n_features)
    #         List of samples.
    #     """
    #     check_is_fitted(self)
    #     # TODO: implement sampling for other valid kernel shapes
    #     if self.kernel not in ['gaussian', 'tophat']:
    #         raise NotImplementedError()
    #
    #     data = np.asarray(self.tree_.data)
    #
    #     rng = check_random_state(random_state)
    #     u = rng.uniform(0, 1, size=n_samples)
    #     if self.tree_.sample_weight is None:
    #         i = (u * data.shape[0]).astype(np.int64)
    #     else:
    #         cumsum_weight = np.cumsum(np.asarray(self.tree_.sample_weight))
    #         sum_weight = cumsum_weight[-1]
    #         i = np.searchsorted(cumsum_weight, u * sum_weight)
    #     if self.kernel == 'gaussian':
    #         return np.atleast_2d(rng.normal(data[i], self.bandwidth)), data[i]  # return X value as well
    #
    #     elif self.kernel == 'tophat':
    #         # we first draw points from a d-dimensional normal distribution,
    #         # then use an incomplete gamma function to map them to a uniform
    #         # d-dimensional tophat distribution.
    #         dim = data.shape[1]
    #         X = rng.normal(size=(n_samples, dim))
    #         s_sq = row_norms(X, squared=True)
    #         correction = (gammainc(0.5 * dim, 0.5 * s_sq) ** (1. / dim)
    #                       * self.bandwidth / np.sqrt(s_sq))
    #         return data[i] + X * correction[:, np.newaxis]

# Following 5 functions from https://github.com/nchopin/particles
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A

def uniform_spacings(N):
    """ Generate ordered uniform variates in O(N) time.
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]

def multinomial_once(W):
    """ Sample once from a Multinomial distribution
    Parameters
    ----------
    W: (N,) ndarray
        normalized weights (>=0, sum to one)
    Returns
    -------
    int
        a single draw from the discrete distribution that generates n with
        probability W[n]
    Note
    ----
    This is equivalent to
       A = multinomial(W, M=1)
    but it is faster.
    """
    return np.searchsorted(np.cumsum(W), random.rand())

def multinomial(W, M):
    return inverse_cdf(uniform_spacings(M), W)

def stratified(W, M):
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)

# Needed for some functionality to access in CES
class MyGPyModelWrapper(GPyModelWrapper):
    def __init__(self, gpy_model: GPy.core.Model, n_restarts: int = 1):
        super().__init__(gpy_model, n_restarts)

    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        return self.model.posterior_samples(X, size=size, Y_metadata=Y_metadata, likelihood=likelihood,
                                            **predict_kwargs)

    def set_X(self, X):
        self.model.set_X(X)

    def set_XY(self, X, y):
        self.model.set_XY(X, y)

    def get_X(self):
        return self.model.X

    def get_Y(self):
        return self.model.Y

def sample_global_xystar(n_samples_mixture, all_ystar, all_xstar, arm_dist, arm_mapping_n_to_es):
    # Select indexes of mixture components to sample from first
    mixture_idxs = stratified(W=arm_dist, M=n_samples_mixture) # lower variance sampling

    # DEBUG info
    # arms_resampled = [arm_mapping_n_to_es[i] for i in mixture_idxs]

    local_pystars = []
    # This fits a KDE to each local p(Y*) i.e. p(Y*_(Z) | D), p(Y*_(X) | D)
    for mixt_idx in range(all_ystar.shape[0]):
        temp = all_ystar[mixt_idx, :].reshape(-1, 1)
        kde = MyKDENew(temp)
        try:
            kde.fit()
        except RuntimeError:
            kde.fit(bw=0.5)

        local_pystars.append(kde)

        # Plotting
        # temp = temp.flatten(),
        # plt.hist(temp, 100, density=True, facecolor='g', alpha=0.75)
        # grid = np.linspace(np.min(temp) - 2., np.max(temp) + 2., 1000)
        # plt.plot(grid, kde.evaluate(grid))
        # plt.title('histogram p(ystar) local for ' + str(arm_mapping_n_to_es[mixt_idx]))
        # plt.show()

    resy = np.empty(mixture_idxs.shape[0])
    # corresponding_x = [[] for _ in range(mixture_idxs.shape[0])]

    # Long for loop
    # for i, mixture_id in enumerate(mixture_idxs.tolist()):
    #     resy[i], corresponding_x[i] = y_single_sample_from_component(mixture_id , local_pystars)

    unique_mixture_idxs, counts = np.unique(mixture_idxs, return_counts=True)
    running_cumsum = 0

    corresponding_x = []  # TODO

    for j, (mix_id, count) in enumerate(zip(unique_mixture_idxs, counts)):
        if j == 0:
            resy[:count], _ = local_pystars[mix_id].sample(n_samples=count)
        else:
            resy[running_cumsum:running_cumsum + count], _ = local_pystars[mix_id].sample(n_samples=count)

        # temp = convert(temp)
        # corresponding_x = corresponding_x + temp
        running_cumsum += count

    # assert not np.isnan(resy).any() and np.isfinite(resy).all()
    # plt.hist(resy, 100, density=True, facecolor='r', alpha=0.75)
    # plt.title('histogram p(ystar) global')
    # plt.show()
    return resy, corresponding_x


def convert(lst):
    return list(map(lambda el: [el], lst))


def build_pystar(arm_mapping, bo_model, int_grids, parameter_int_domain, task, seed_anchor_points):
    sets = bo_model.keys()
    n_samples = 200  # samples to build local p(y*, x*)
    all_ystar = np.empty((len(sets), n_samples)) # |ES| x n_samples per local dist
    all_xstar = [[] for _ in range(len(sets))]  # different dimensions, cannot use numpy array

    for es, i in arm_mapping.items():
        model = bo_model[es]
        # sample from huge grid if len(es) > 1:
        if len(es) > 1:
            np.random.seed(seed_anchor_points)
            inps = parameter_int_domain[es].sample_uniform(point_count=100)
        else:
            inps = int_grids[es]

        sampless = model.posterior_samples(inps, size=n_samples).squeeze()

        if task == "min":
            all_ystar[i, :] = np.min(sampless, axis=0).squeeze()
            all_xstar[i] = inps[np.argmin(sampless, axis=0), :].squeeze()
        else:
            all_ystar[i, :] = np.max(sampless, axis=0).squeeze()
            all_xstar[i] = inps[np.argmax(sampless, axis=0), :].squeeze()

    # assert not np.isnan(all_ystar).any()
    # assert np.isfinite(all_ystar).all()

    return all_ystar, all_xstar  # impossible to define global x star . used only for plotting


def update_pystar_single_model(arm_mapping, es, bo_model, inputs, task, all_ystar, all_xstar, space, seed):
    corresponding_idx = arm_mapping[es]
    n_samples = all_ystar.shape[1]  # samples to build local p(y*, x*)

    sampless = bo_model.posterior_samples(inputs, size=n_samples)  # less samples to speed up
    sampless = sampless.squeeze()

    # if task == "min":
    # idxs_best = np.argmin(sampless, axis=[0,1])
    # best_values = sampless[idxs_best, :]
    # all_ystar[corresponding_idx, :] = best_values
    # all_xstar[corresponding_idx] = inputs[idxs_best, :]

    all_ystar[corresponding_idx, :] = np.min(sampless, axis=0)  # NOTE: it is really important all_ystar is the previouss one ! This is an UPDATE move
    # all_xstar[corresponding_idx] = inputs[np.argmin(sampless, axis=0), :].squeeze() # TODO

    # assert not np.isnan(all_ystar).any()  and np.isfinite(all_ystar).all()

    return all_ystar, all_xstar  # used only for plotting so not tracking x for now


def update_arm_dist(arm_distribution, updated_bo_model, inputs, temporal_index, task, arm_mapping_es_to_n, beta=0.1):
    for es in updated_bo_model[temporal_index].keys():
        corresponding_n = arm_mapping_es_to_n[es]
        inp = inputs[es]
        preds_mean, preds_var = updated_bo_model[temporal_index][es].predict(inp)  # Predictive mean
        # min or max for this ES
        if task == "min":
            arm_distribution[corresponding_n] = np.min(preds_mean) - beta * np.sqrt(
                preds_var[np.argmin(preds_mean)])  # inefficient! Doing min twice here
        elif task == "max":
            arm_distribution[corresponding_n] = np.max(preds_mean) + beta * np.sqrt(preds_var[np.argmax(preds_mean)])
        else:
            continue

    return arm_distribution


def update_arm_dist_single_model(arm_distribution, es, single_updated_bo_model, inputs, task, arm_mapping_es_to_n,
                                 parameter_int_domain, seed_anchor_points, beta=0.1):
    corresponding_n = arm_mapping_es_to_n[es]
    inps = inputs
    preds_mean, preds_var = single_updated_bo_model.predict(inps)  # Predictive mean    #

    # TODO: check exactly here to what to do when wanting to maximise. For now only considering minimization
    # if task == "min":
    arm_distribution[corresponding_n] = np.min(preds_mean) - beta * np.sqrt(preds_var[np.argmin(preds_mean)])
    # else:
    #     arm_distribution[corresponding_n] = np.max(preds_mean) + beta * np.sqrt(preds_var[np.argmax(preds_mean)])

    return arm_distribution


def to_prob(arm_values, task="min"):
    return softmax(-(1) * np.array(arm_values)) if task == "min" else softmax(np.array(arm_values))

def fake_do_x(x, node_parents, graphs, log_graph_post, intervened_vars, all_sem, all_emission_fncs,):
    # Get a set of all variables
    # all_vars = list(self.all_emission_pairs[0].keys())
    all_vars = list(all_sem[0]().static(0).keys())

    # This will hold the fake intervention
    intervention_blanket = {k: np.array([None]).reshape(-1, 1) for k in all_vars}

    for i, intervened_var in enumerate(intervened_vars):
        intervention_blanket[intervened_var] = np.array(x.reshape(1, -1)[0, i]).reshape(-1, 1)
    # Better than  MAP
    posterior_to_avg = []
    for idx_graph in range(len(all_sem)):
        sem_hat_map = all_sem[idx_graph]
        # interv_sample = sequential_sample_from_complex_model_hat_new(
        #     static_sem=sem_hat_map().static(moment=0), dynamic_sem=None
        #     , timesteps=1, emission_pairs=all_emission_pairs[idx_graph],
        #     interventions=intervention_blanket)
        interv_sample = sequential_sample_from_SEM_hat(
            static_sem=sem_hat_map().static(moment=0),
            dynamic_sem=None,
            timesteps=1,
            node_parents=partial(node_parents,
            graph=graphs[idx_graph]),
            interventions=intervention_blanket,
        )

        # In theory could/should replace Y with sample from surrogate model
        for var, val in interv_sample.items():
            interv_sample[var] = val.reshape(-1, 1)

        # P(G | D, (x,y) )  . avg over V_y  =  V \ (x,y)
        posterior_to_avg.append(update_posterior_interventional(graphs=graphs,
                                                         posterior=deepcopy(log_graph_post),
                                                         intervened_var=intervened_vars,
                                                         all_emission_fncs=all_emission_fncs,
                                                         interventional_samples=interv_sample,
                                                         total_timesteps=1,
                                                         it=0))

    posterior_to_avg = np.vstack(posterior_to_avg)
    # Average over intervention outcomes
    return np.average(posterior_to_avg, axis=0, weights=log_graph_post)