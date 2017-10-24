"""Metropolis-Hastings algorithm."""
import warnings
import numpy as np

from helpers.distributions import multivariate_gaussian
from helpers.cov_matrix import is_valid_covariance_matrix

from parameter.mcmc.output import plot_results
from parameter.mcmc.output import print_progress_report

from parameter.mcmc.performance_measures import compute_ess
from parameter.mcmc.performance_measures import compute_iact
from parameter.mcmc.performance_measures import compute_sjd

from parameter.mcmc.variance_reduction import zero_variance_linear

from parameter.mcmc.gradient_estimation import get_gradient
from parameter.mcmc.gradient_estimation import get_nat_gradient
from parameter.mcmc.hessian_estimation import get_hessian

warnings.filterwarnings("error")

class ParameterEstimator(object):
    """Metropolis-Hastings algorithm."""
    def __init__(self, model, self_type, settings):
        self.use_gradient_information = False
        self.use_hessian_information = False

        quasi_newton = {
            'memory_length': 20,
            'initial_hessian': 'fixed',
            'strategy': 'sr1',
            'bfgs_curvature_cond': 'ignore',
            'initial_hessian_scaling': 0.10,
            'initial_hessian_fixed': 0.01**2,
            'only_accepted_info': True
        }

        self.mhSettings = {'no_iters': 1000,
                           'no_burnin_iters': 250,
                           'step_size': 0.5,
                           'base_hessian': np.eye(3) * 0.10**2,
                           'init_params': (0.2, 0.5, 1.0),
                           'verbose': False,
                           'waitForENTER': False,
                           'trustRegionSize': None,
                           'quasi_newton' : quasi_newton,
                           'hessian_estimate': 'kalman',
                           'hessian_correction': 'replace',
                           'hessian_correction_verbose': False
                          }

        self.results = {}

        if self_type is 'mh0':
            self.name = "Zero-order Metropolis-Hastings with Kalman methods"

        if self_type is 'mh1':
            self.name = "First-order Metropolis-Hastings with Kalman methods"
            self.use_gradient_information = True

        if self_type is 'mh2':
            self.name = "Second-order Metropolis-Hastings with Kalman methods"
            self.use_gradient_information = True
            self.use_hessian_information = True

        print("Sampling from the parameter posterior using: " + self.name)
        self._check_settings()

        self.no_hessians_corrected = 0
        self.iter_hessians_corrected = []

        self.settings['no_obs'] = model.no_obs
        self.settings['params_to_estimate'] = model.params_to_estimate
        self.settings['no_params_to_estimate'] = model.no_params_to_estimate

        no_iters = self.settings['no_iters']
        no_burnin_iters = self.settings['no_burnin_iters']
        no_params_to_estimate = model.no_params_to_estimate
        no_obs = model.no_obs

        if no_burnin_iters >= no_iters:
            raise ValueError("metropolisHastings: no_burnin_iters cannot be " +
                             "larger or equal to no_iters.")

        self.no_params_to_estimate = no_params_to_estimate
        self.free_params = np.zeros((no_iters, no_params_to_estimate))
        self.params = np.zeros((no_iters, no_params_to_estimate))
        self.prop_free_params = np.zeros((no_iters, no_params_to_estimate))
        self.prop_params = np.zeros((no_iters, no_params_to_estimate))

        self.log_prior = np.zeros((no_iters, 1))
        self.log_likelihood = np.zeros((no_iters, 1))
        self.log_jacobian = np.zeros((no_iters, 1))
        self.states = np.zeros((no_iters, no_obs))
        self.prop_log_prior = np.zeros((no_iters, 1))
        self.prop_log_like = np.zeros((no_iters, 1))
        self.prop_log_jacovian = np.zeros((no_iters, 1))
        self.prop_states = np.zeros((no_iters, no_obs))

        self.accept_prob = np.zeros((no_iters, 1))
        self.accepted = np.zeros((no_iters, 1))
        self.no_effective_samples = np.zeros((no_iters, 1))

        self.gradient = np.zeros((no_iters, no_params_to_estimate))
        self.nat_gradient = np.zeros((no_iters, no_params_to_estimate))
        self.hessian = np.zeros((no_iters, no_params_to_estimate,
                                 no_params_to_estimate))
        self.prop_grad = np.zeros((no_iters, no_params_to_estimate))
        self.prop_nat_grad = np.zeros((no_iters, no_params_to_estimate))
        self.prop_hess = np.zeros((no_iters, no_params_to_estimate,
                                   no_params_to_estimate))
        self.current_iter = 0

    def run(self, state_estimator, model):
        """Runs the Metropolis-Hastings algorithm."""

        no_iters = self.settings['no_iters']
        no_burnin_iters = self.settings['no_burnin_iters']
        self.current_iter = 0

        self._initialise_params(state_estimator, model)

        for i in range(1, no_iters):
            self.current_iter = i
            self._propose_params(model)
            self._compute_accept_prob(state_estimator, model)
            if (np.random.random(1) < self.accept_prob[i, :]):
                self._accept_params()
            else:
                self._reject_params()

            if self.settings['verbose']:
                print("Current unrestricted parameters: " +
                      str(self.params[i, :]) + ".")
                print("Current restricted parameters: " +
                      str(self.free_params[i, :]) + ".")

                if self.settings['verbose_wait_enter']:
                    input("Press ENTER to continue...")

            flag = self.settings['no_iters_between_progress_reports']
            flag = np.remainder(i + 1, flag) == 0
            if flag:
                print_progress_report(self)
            #self.printSimulationToFile()

        idx = np.range(no_burnin_iters, no_iters)
        res = {}
        res.update({'param_est': np.mean(self.free_params[idx, :], axis=0)})
        res.update({'state_est': np.mean(self.states[idx, :], axis=0)})
        res.update({'state_est_var': np.var(self.states[idx, :], axis=0)})
        res.update({'trace': self.free_params[idx, :]})
        res.update({'prop_free_params': self.prop_free_params[idx, :]})
        res.update({'prop_grad': self.prop_nat_grad[idx, :]})
        self.results = res

    def _accept_params(self):
        """Record the accepted parameters."""
        i = self.current_iter
        self.free_params[i, :] = self.prop_free_params[i, :]
        self.params[i, :] = self.prop_params[i, :]
        self.log_jacobian[i] = self.prop_log_jacovian[i]
        self.log_prior[i, :] = self.prop_log_prior[i, :]
        self.log_likelihood[i, :] = self.prop_log_like[i, :]
        self.states[i, :] = self.prop_states[i, :]
        self.gradient[i, :] = self.prop_grad[i, :]
        self.nat_gradient[i, :] = self.prop_nat_grad[i, :]
        self.hessian[i, :, :] = self.prop_hess[i, :, :]
        self.accepted[i] = 1.0

    def _reject_params(self):
        """Record the rejected parameters."""
        offset = 1
        # if self.use_hessian_information:
        #     if self.settings['hessianEstimate'] is not 'Kalman':
        #         if self.current_iter > self.settings['memoryLength']:
        #             offset = self.settings['memoryLength']
        i = self.current_iter
        self.free_params[i, :] = self.free_params[i - offset, :]
        self.params[i, :] = self.params[i - offset, :]
        self.log_jacobian[i] = self.log_jacobian[i - offset]
        self.log_prior[i, :] = self.log_prior[i - offset, :]
        self.log_likelihood[i, :] = self.log_likelihood[i - offset, :]
        self.states[i, :] = self.states[i - offset, :]
        self.gradient[i, :] = self.gradient[i - offset, :]
        self.nat_gradient[i, :] = self.nat_gradient[i - offset, :]
        self.hessian[i, :, :] = self.hessian[i - offset, :, :]
        self.accepted[i] = 0.0

    def _initialise_params(self, state, model):
        """Initalise the Metropolis-Hastings algorithm."""
        model.store_free_params(self.settings['initial_params'])

        if model.check_parameters():
            self.log_jacobian[0] = model.log_jacobian()
            _, self.log_prior[0] = model.log_prior()

            state.smoother(model)
            self.log_likelihood[0] = state.log_likelihood
            self.states[0, :] = state.filt_state_est

            self.gradient[0, :] = get_gradient(self, state)
            self.hessian[0, :, :] = get_hessian(self, state, self.gradient[0, :])
            self.nat_gradient[0, :] = get_nat_gradient(self,
                                                       self.gradient[0, :],
                                                       self.hessian[0, :, :])

            self.free_params[0, :] = self.settings['initial_params']
            self.params[0, :] = model.get_params()
            self.accept_prob[0] = 1.0
        else:
            raise NameError("The initial values of the parameters does " +
                            "not result in a valid model.")

    def _propose_params(self, model):
        """Parameter proposal."""
        no_param = self.settings['no_params_to_estimate']
        cur_param = self.params[self.current_iter - 1, :]
        cur_nat_grad = self.nat_gradient[self.current_iter - 1, :]
        curr_hess = self.hessian[self.current_iter - 1, :, :]

        if no_param == 1:
            perturbation = np.sqrt(np.abs(curr_hess)) * np.random.normal()
        else:
            perturbation = np.random.multivariate_normal(np.zeros(no_param),
                                                         curr_hess)

        param_change = cur_nat_grad + perturbation
        prop_params = cur_param + param_change

        if self.settings['verbose']:
            print("Proposing unrestricted parameters: " + str(prop_params) +
                  " given " + str(cur_param) + ".")

        model.store_free_params(prop_params)
        self.prop_params[self.current_iter, :] = model.get_params()
        self.prop_free_params[self.current_iter, :] = model.get_free_params()

    def _compute_accept_prob(self, state, model):
        """Compute acceptance probability."""
        offset = 1
        # if self.use_hessian_information:
        #     if self.settings['hessianEstimate'] is not 'Kalman':
        #         if self.current_iter > self.settings['memoryLength']:
        #             offset = self.settings['memoryLength']

        cur_free_param = self.free_params[self.current_iter - offset, :]
        cur_param = self.params[self.current_iter - offset, :]
        cur_log_jacobian = self.log_jacobian[self.current_iter - offset, :]
        cur_log_prior = self.log_prior[self.current_iter - offset, :]
        cur_log_like = self.log_likelihood[self.current_iter - offset, :]
        cur_nat_grad = self.nat_gradient[self.current_iter - offset, :]
        curr_hess = self.hessian[self.current_iter - offset, :, :]

        prop_free_params = self.prop_free_params[self.current_iter, :]
        prop_params = self.prop_params[self.current_iter, :]
        model.storefree_params(prop_free_params)

        if model.areParametersValid():
            prop_log_jacobian = model.log_jacobian()
            _, prop_log_prior = model.log_prior()

            state.smoother(model)
            prop_log_like = state.log_likelihood
            prop_states = state.filt_state_est

            prop_grad = get_gradient(self, state)
            prop_hess = get_hessian(self, state, prop_grad)
            prop_nat_grad = get_nat_gradient(self, prop_grad, prop_hess)

            if is_valid_covariance_matrix(prop_hess):
                log_prior_diff = float(prop_log_prior - cur_log_prior)
                log_like_diff = float(prop_log_like - cur_log_like)

                cur_mean = cur_param + cur_nat_grad
                prop_prop = multivariate_gaussian.logpdf(prop_params, cur_mean,
                                                         curr_hess)

                prop_mean = prop_params + prop_nat_grad
                cur_prop = multivariate_gaussian.logpdf(cur_param, prop_mean,
                                                        prop_hess)

                log_prop_diff = float(prop_prop - cur_prop)
                log_jacob_diff = float(prop_log_jacobian - cur_log_jacobian)

                try:
                    accept_prob = np.exp(log_prior_diff + log_like_diff +
                                         log_prop_diff + log_jacob_diff)
                except:
                    if self.settings['verbose']:
                        print("Rejecting as overflow occured.")
                    accept_prob = 0.0
                if self.settings['trust_region_size']:
                    max_param_diff = prop_free_params - cur_free_param
                    max_param_diff = np.max(np.abs(max_param_diff))
                    if max_param_diff > self.settings['trust_region_size']:
                        accept_prob = 0.0
                        print("Rejected as parameters violate trust region.")
            else:
                print("iteration: " + str(self.current_iter) +
                      ", estimate of inverse Hessian is not PSD or is" +
                      " singular, so rejecting...")
                #print(prop_hess)
                #print(np.linalg.eig(prop_hess)[0])
                accept_prob = 0.0

            if self.settings['verbose']:
                if is_valid_covariance_matrix(prop_hess):
                    print("cur_free_param" + str(cur_free_param) + ".")
                    print("prop_free_params" + str(prop_free_params) + ".")
                    print("prop_log_like: " + str(prop_log_like) + ".")
                    print(": " + str() + ".")
                    print("loglike_diff: " + str(log_like_diff) + ".")
                    print("prop_prop: " + str(prop_prop) + ".")
                    print("cur_prop: " + str(cur_prop) + ".")
                    print("log_prop_diff: " + str(log_prop_diff) + ".")
                    print("log_jacob_diff: " + str(log_jacob_diff) + ".")
                    print("accept_prob: " + str(accept_prob) + ".")

            self.accept_prob[self.current_iter] = np.min((1.0, accept_prob))

            self.prop_log_jacovian[self.current_iter] = prop_log_jacobian
            self.prop_log_prior[self.current_iter] = prop_log_prior
            self.prop_log_like[self.current_iter] = prop_log_like
            self.prop_states[self.current_iter, :] = prop_states
            self.prop_nat_grad[self.current_iter, :] = prop_nat_grad
            self.prop_grad[self.current_iter, :] = prop_grad
            self.prop_hess[self.current_iter, :, :] = prop_hess
        else:
            self.accept_prob[self.current_iter] = 0.0
            print("Proposed parameters: " + str(prop_free_params) +
                  " results in an unstable system so rejecting.")

    def plot(self):
        """Plots the results from a run of the Metropolis-Hastings algorithm."""
        plot_results(self)

    def compute_ess(self, max_lag=None):
        """Computes the effective sample size from a run of the
           Metropolis-Hastings algorithm."""
        return compute_ess(self, max_lag)

    def compute_iact(self, max_lag=None):
        """Computes the integrated autocorrelation time from a run of the
           Metropolis-Hastings algorithm."""
        return compute_iact(self, max_lag)

    def compute_sjd(self):
        """Computes the squared jump distance from a run of the
           Metropolis-Hastings algorithm."""
        return compute_sjd(self)

    def zero_variance_linear(self):
        """Carries out zero-variance post-processing of the Markov chain."""
        return zero_variance_linear(self)

    def _check_settings(self):
        if not 'no_iters' in self.settings:
            self.settings.update({'no_iters': 1000})
            print("Missing settings: no_iters, defaulting to " +
                  str(self.settings['no_iters']) + ".")

        if not 'no_burin_iters' in self.settings:
            self.settings.update({'no_burin_iters': 250})
            print("Missing settings: no_burin_iters, defaulting to " +
                  str(self.settings['no_burin_iters']) + ".")

        if not 'step_size' in self.settings:
            self.settings.update({'step_size': 0.10})
            print("Missing settings: step_size, defaulting to " +
                  str(self.settings['step_size']) + ".")

        if not 'no_iters_between_progress_reports' in self.settings:
            self.settings.update({'no_iters_between_progress_reports': 100})
            print("Missing settings: no_iters_between_progress_reports, " +
                  "defaulting to " +
                  str(self.settings['no_iters_between_progress_reports']) + ".")
