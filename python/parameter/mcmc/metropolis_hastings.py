"""Metropolis-Hastings algorithm."""
import warnings
import numpy as np

from helpers.distributions import multivariate_gaussian
from helpers.cov_matrix import is_valid_covariance_matrix

from parameter.mcmc.output import plot_results
from parameter.mcmc.output import print_progress_report
from parameter.mcmc.output import store_results_to_file
from parameter.mcmc.output import store_results_to_db
from parameter.mcmc.output import print_greeting

from parameter.mcmc.performance_measures import compute_ess
from parameter.mcmc.performance_measures import compute_iact
from parameter.mcmc.performance_measures import compute_sjd

from parameter.mcmc.variance_reduction import zero_variance_linear

from parameter.mcmc.gradient_estimation import get_gradient
from parameter.mcmc.gradient_estimation import get_nat_gradient
from parameter.mcmc.hessian_estimation import get_hessian

from parameter.base_parameter_inference import BaseParameterInference

warnings.filterwarnings("error")

class MetropolisHastings(BaseParameterInference):
    """ Metropolis-Hastings algorithm.

        Implements the standard Metropolis-Hastings algorithm with a Gaussian
        proposal with some mean and covariance. The description of the algorithm
        is given by:

        1) Initialise Markov chain [_initialise_params]
        2) FOR iter in 1 to no_iters
            2a) Propose a candidate parameter. [_propose_params]
            2b) Compute the acceptance probability. [_compute_accept_prob]
            2c) Accept / reject step. [_accept_params / _reject_params]
            2d) Compile output and write to screen. [print_progress_report]
        3) Post-processing and writing results to file. [store_results_to_file]

        These steps are implemented as separate function in this file with the
        names given as above. The main steps are given in the function "run".

        The settings of the algorithms are encoded in the dict settings given
        at initialisation of the class. Standard settings are given otherwise.
        Also the user needs to provide the type of MH algorithm to be run.abs

        Args:
            model: a model class to conduct inference on.
            alg_type: the type of MH algorithm to use. (string) Choose from:
                mh0: standard MH with a Gaussian random walk proposal.
                mh1: the MALA algorithm which introduce a drift in the proposal
                     in the form of the gradient of the log-target. The step
                     sizes are difficult to tune for this model.
                mh2: the (simplified) manifold MALA algorithm which makes use
                     of both gradient and Hessian information of the log-target.
                     The Hessian is tricky to estimate but Kalman methods or
                     direct particle methods are used here.
                qmh: basically the same as mh2 but using quasi-Newton updates
                     to estimate the Hessian directly from gradient information.

            new_settings: a dict with the following settings:
                'no_iters': number of MH iterations to carry out. (integer)

                'no_burnin_iters': number of iterations to discard as burn-in.

                'step_size': the step length of the proposal. (float)

                'base_hessian': the Hessian estimate for the proposal. (array)

                'initial_params': parameter vector to initialise the Markov
                                  chain in. (array)

                'verbose': should additional information for debugging be
                           printed to screen? (boolean)

                'verbose_wait_enter': should the program wait for ENTER after
                                      each iteration. (boolean)

                'trust_region_size': should proposed parameters be rejected if
                                     they differ to much from the current
                                     parameter. (None: disables this function,
                                     otherwise give a float)

                'hessian_estimate': how should the Hessian be estimated:
                                    None: base_hessian is used
                                    'segal_weinstein': Segal Weinstein is used.
                                    'quasi_newton': BFGS or SR1 is used.
                                    [This is set automatically by alg_type]

                'hessian_correction': how should the Hessian be corrected.
                                      see _correct_hessian for more information.

                'hessian_correction_verbose': should each correction be printed
                                              to screen?

                'no_iters_between_progress_reports': how often shall progress
                                                     reports be given to user.

                'qn_memory_length': See quasi_newton.main.quasi_newton

                'qn_initial_hessian': See quasi_newton.main.quasi_newton

                'qn_strategy': See quasi_newton.main.quasi_newton

                'qn_bfgs_curvature_cond': See quasi_newton.main.quasi_newton

                'qn_sr1_safe_parameterisation': See quasi_newton.main.quasi_newton

                'qn_sr1_skip_limit': See quasi_newton.main.quasi_newton

                'qn_initial_hessian_scaling': See quasi_newton.main.quasi_newton

                'qn_initial_hessian_fixed': See quasi_newton.main.quasi_newton

                'qn_only_accepted_info': See quasi_newton.main.quasi_newton

    """
    def __init__(self, model, alg_type, new_settings=None):
        self.use_grad_info = False
        self.use_hess_info = False

        self.settings = {'no_iters': 1000,
                         'no_burnin_iters': 250,
                         'step_size': 0.5,
                         'base_hessian': np.eye(3) * 0.10**2,
                         'initial_params': (0.2, 0.5, 1.0),
                         'verbose': False,
                         'verbose_wait_enter': False,
                         'trust_region_size': None,
                         'hessian_estimate': None,
                         'hessian_correction': 'replace',
                         'hessian_correction_verbose': False,
                         'no_iters_between_progress_reports': 100,
                         'qn_memory_length': 20,
                         'qn_initial_hessian': 'fixed',
                         'qn_strategy': None,
                         'qn_bfgs_curvature_cond': 'ignore',
                         'qn_sr1_safe_parameterisation': False,
                         'qn_sr1_skip_limit': 1e-8,
                         'qn_initial_hessian_scaling': 0.10,
                         'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
                         'qn_only_accepted_info': True
                        }


        if new_settings:
            self.settings.update(new_settings)

        if alg_type is 'mh0':
            self.name = "Zero-order Metropolis-Hastings"
        elif alg_type is 'mh1':
            self.name = "First-order Metropolis-Hastings"
            self.use_grad_info = True
        elif alg_type is 'mh2':
            self.name = "Second-order Metropolis-Hastings"
            self.use_grad_info = True
            self.use_hess_info = True
            self.settings['hessian_estimate'] = 'segal_weinstein'
        elif alg_type is 'qmh':
            self.name = "quasi-Newton Metropolis-Hastings"
            self.use_grad_info = True
            self.use_hess_info = True
            self.settings['hessian_estimate'] = 'quasi_newton'
            if self.settings['qn_strategy'] is None:
                raise ValueError("No quasi-Newton strategy selected...")
            elif self.settings['qn_strategy'] is 'sr1':
                print("Hessian estimation using SR1 update.")
            elif self.settings['qn_strategy'] is 'bfgs':
                print("Hessian estimation using BFGS update.")
            else:
                raise ValueError("Unknown quasi-Newton strategy selected...")
        else:
            raise ValueError("Unknown MH variant selected...")

        self.no_hessians_corrected = 0
        self.iter_hessians_corrected = []

        self.model = model
        no_iters = self.settings['no_iters']
        no_burnin_iters = self.settings['no_burnin_iters']
        no_params_to_estimate = self.model.no_params_to_estimate
        no_obs = self.model.no_obs

        if no_burnin_iters >= no_iters:
            raise ValueError("metropolisHastings: no_burnin_iters cannot be " +
                             "larger or equal to no_iters.")

        self.free_params = np.zeros((no_iters, no_params_to_estimate))
        self.params = np.zeros((no_iters, no_params_to_estimate))
        self.prop_free_params = np.zeros((no_iters, no_params_to_estimate))
        self.prop_params = np.zeros((no_iters, no_params_to_estimate))

        self.log_prior = np.zeros((no_iters, 1))
        self.log_like = np.zeros((no_iters, 1))
        self.log_jacobian = np.zeros((no_iters, 1))
        self.states = np.zeros((no_iters, no_obs + 1))
        self.prop_log_prior = np.zeros((no_iters, 1))
        self.prop_log_like = np.zeros((no_iters, 1))
        self.prop_log_jacobian = np.zeros((no_iters, 1))
        self.prop_states = np.zeros((no_iters, no_obs + 1))

        self.accept_prob = np.zeros((no_iters, 1))
        self.accepted = np.zeros((no_iters, 1))
        self.no_samples_hess_est = np.zeros((no_iters, 1))

        self.gradient = np.zeros((no_iters, no_params_to_estimate))
        self.nat_gradient = np.zeros((no_iters, no_params_to_estimate))
        self.hess = np.zeros((no_iters, no_params_to_estimate,
                                 no_params_to_estimate))
        self.prop_grad = np.zeros((no_iters, no_params_to_estimate))
        self.prop_nat_grad = np.zeros((no_iters, no_params_to_estimate))
        self.prop_hess = np.zeros((no_iters, no_params_to_estimate,
                                   no_params_to_estimate))
        self.current_iter = 0

    def run(self, state_estimator):
        """ Runs the Metropolis-Hastings algorithm.

            The settings are given in the attribute settings as a dict. The
            results are stored internally in the object.

            Args:
                state_estimator: a state estimator object

            Returns:
                Nothing.

        """
        print_greeting(self, state_estimator)

        if self.use_grad_info or self.use_hess_info:
            state_estimator.settings['estimate_gradient'] = True

        no_iters = self.settings['no_iters']
        self.current_iter = 0

        self._initialise_params(state_estimator, self.model)

        for i in range(1, no_iters):

            if self.settings['verbose']:
                print("")
                print("#######################################################")
                print("Iteration: {} of {}.".format(i, no_iters))
                print("")

            self.current_iter = i
            self._propose_params(self.model)
            self._compute_accept_prob(state_estimator, self.model)
            if (np.random.random(1) < self.accept_prob[i, :]):
                self._accept_params()
            else:
                self._reject_params()

            if self.settings['verbose_wait_enter']:
                input("Press ENTER to continue...")

            flag = self.settings['no_iters_between_progress_reports']
            flag = np.remainder(i + 1, flag) == 0
            if flag:
                print_progress_report(self)

            if self.settings['verbose']:
                print("")
                print("#######################################################")
                print("")

        print("Run of MH algorithm complete...")

    def _accept_params(self):
        """ Record the accepted parameters. """
        i = self.current_iter
        self.free_params[i, :] = self.prop_free_params[i, :]
        self.params[i, :] = self.prop_params[i, :]
        self.log_jacobian[i] = self.prop_log_jacobian[i]
        self.log_prior[i, :] = self.prop_log_prior[i, :]
        self.log_like[i, :] = self.prop_log_like[i, :]
        self.states[i, :] = self.prop_states[i, :]
        self.gradient[i, :] = self.prop_grad[i, :]
        self.nat_gradient[i, :] = self.prop_nat_grad[i, :]
        self.hess[i, :, :] = self.prop_hess[i, :, :]
        self.accepted[i] = 1.0

    def _reject_params(self):
        """ Record the rejected parameters. """
        offset = 1
        if self.use_hess_info:
            if self.settings['hessian_estimate'] is 'quasi_newton':
                if self.current_iter > self.settings['qn_memory_length']:
                    offset = self.settings['qn_memory_length']
        i = self.current_iter
        self.free_params[i, :] = self.free_params[i - offset, :]
        self.params[i, :] = self.params[i - offset, :]
        self.log_jacobian[i] = self.log_jacobian[i - offset]
        self.log_prior[i, :] = self.log_prior[i - offset, :]
        self.log_like[i, :] = self.log_like[i - offset, :]
        self.states[i, :] = self.states[i - offset, :]
        self.gradient[i, :] = self.gradient[i - offset, :]
        self.nat_gradient[i, :] = self.nat_gradient[i - offset, :]
        self.hess[i, :, :] = self.hess[i - offset, :, :]
        self.accepted[i] = 0.0

    def _initialise_params(self, state, model):
        """ Initialise the Metropolis-Hastings algorithm. """
        model.store_params(self.settings['initial_params'])

        if model.check_parameters():
            self.log_jacobian[0] = model.log_jacobian()
            _, self.log_prior[0] = model.log_prior()

            if self.use_grad_info or self.use_hess_info:
                state.smoother(model)
            else:
                state.filter(model)

            self.log_like[0] = float(state.results['log_like'])
            init_state = state.results['filt_state_est']
            self.states[0, :] = init_state.reshape(model.no_obs+1)

            self.gradient[0, :] = get_gradient(self, state)
            self.hess[0, :, :] = get_hessian(self, state,
                                             self.gradient[0, :])
            self.nat_gradient[0, :] = get_nat_gradient(self,
                                                       self.gradient[0, :],
                                                       self.hess[0, :, :])

            self.free_params[0, :] = model.get_free_params()
            self.params[0, :] = model.get_params()
            self.accept_prob[0] = 1.0
        else:
            raise NameError("The initial values of the parameters does " +
                            "not result in a valid model.")

    def _propose_params(self, model):
        """ Proposes new parameters given the current parameters. """
        offset = 1
        if self.use_hess_info:
            if self.settings['hessian_estimate'] is 'quasi_newton':
                if self.current_iter > self.settings['qn_memory_length']:
                    offset = self.settings['qn_memory_length']

        no_param = self.model.no_params_to_estimate
        cur_params = self.free_params[self.current_iter - offset, :]
        cur_nat_grad = self.nat_gradient[self.current_iter - offset, :]
        cur_hess = self.hess[self.current_iter - offset, :, :]

        if no_param == 1:
            perturbation = np.sqrt(np.abs(cur_hess)) * np.random.normal()
        else:
            try:
                perturbation = np.random.multivariate_normal(np.zeros(no_param),
                                                             cur_hess)
            except RuntimeWarning:
                print("Warning raised in np.random.multivariate_normal " +
                      "so using Cholesky to generate random variables.")
                cur_hess_root = np.linalg.cholesky(cur_hess)
                perturbation = np.random.multivariate_normal(np.zeros(no_param),
                                                             np.eye(no_param))
                perturbation = np.matmul(cur_hess_root, perturbation)

        param_change = cur_nat_grad + perturbation
        prop_params = cur_params + param_change

        if self.settings['verbose']:
            print("cur_params: " + str(["%.3f" % v for v in cur_params]))
            print("prop_free_params: " + str(["%.3f" % v for v in prop_params]))
            print("")

        model.store_free_params(prop_params)
        self.prop_params[self.current_iter, :] = model.get_params()
        self.prop_free_params[self.current_iter, :] = model.get_free_params()

    def _compute_accept_prob(self, state, model):
        """ Computes acceptance probability. """
        offset = 1
        if self.use_hess_info:
            if self.settings['hessian_estimate'] is 'quasi_newton':
                if self.current_iter > self.settings['qn_memory_length']:
                    offset = self.settings['qn_memory_length']

        cur_free_params = self.free_params[self.current_iter - offset, :]
        cur_params = self.params[self.current_iter - offset, :]
        cur_log_jacobian = self.log_jacobian[self.current_iter - offset, 0]
        cur_log_prior = self.log_prior[self.current_iter - offset, :]
        cur_log_like = self.log_like[self.current_iter - offset, 0]
        cur_nat_grad = self.nat_gradient[self.current_iter - offset, :]
        cur_hess = self.hess[self.current_iter - offset, :, :]

        prop_free_params = self.prop_free_params[self.current_iter, :]
        prop_params = self.prop_params[self.current_iter, :]
        model.store_free_params(prop_free_params)

        if model.check_parameters():
            prop_log_jacobian = model.log_jacobian()
            _, prop_log_prior = model.log_prior()

            if self.use_grad_info or self.use_hess_info:
                state.smoother(model)
            else:
                state.filter(model)

            prop_log_like = float(state.results['log_like'])
            prop_states = state.results['filt_state_est'].reshape(model.no_obs+1)

            prop_grad = get_gradient(self, state)
            prop_hess = get_hessian(self, state, prop_grad)
            prop_nat_grad = get_nat_gradient(self, prop_grad, prop_hess)

            if is_valid_covariance_matrix(prop_hess):
                log_prior_diff = float(prop_log_prior - cur_log_prior)
                log_like_diff = float(prop_log_like - cur_log_like)

                cur_mean = cur_free_params + cur_nat_grad
                prop_prop = multivariate_gaussian.logpdf(prop_free_params,
                                                         cur_mean,
                                                         cur_hess)

                prop_mean = prop_free_params + prop_nat_grad
                cur_prop = multivariate_gaussian.logpdf(cur_free_params,
                                                        prop_mean,
                                                        prop_hess)

                log_prop_diff = cur_prop - prop_prop
                log_jacob_diff = prop_log_jacobian - cur_log_jacobian

                try:
                    accept_prob = np.exp(log_prior_diff + log_like_diff +
                                         log_prop_diff + log_jacob_diff)
                except:
                    if self.settings['verbose']:
                        print("Accepted bu overflow occured.")
                    accept_prob = 1.0

                if self.settings['trust_region_size']:
                    abs_param_diff = np.abs(prop_free_params - cur_free_params)
                    max_param_diff = np.max(abs_param_diff)
                    if max_param_diff > self.settings['trust_region_size']:
                        accept_prob = 0.0
                        print("Rejected as parameters violate trust region."
                              + " max_param_diff: " + str(max_param_diff) + ".")
                        print(abs_param_diff)
            else:
                print("iteration: " + str(self.current_iter) +
                      ", estimate of inverse Hessian is not PSD or is" +
                      " singular, so rejecting...")
                #print(prop_hess)
                #print(np.linalg.eig(prop_hess)[0])
                accept_prob = 0.0

            if self.settings['verbose']:
                if is_valid_covariance_matrix(prop_hess):
                    print("")
                    print("prop_params: " + str(["%.3f" % v for v in prop_params]))
                    print("prop_free_params: " + str(["%.3f" % v for v in prop_free_params]))
                    print("prop_prop_mean: " + str(["%.3f" % v for v in cur_mean]))
                    print("prop_prop_stdev: " + str(["%.3f" % v for v in np.sqrt(np.diag(cur_hess))]))
                    print("prop_prop_value: {:.3f}".format(prop_prop))
                    print("")
                    print("cur_params: " + str(["%.3f" % v for v in cur_params]))
                    print("cur_free_params: " + str(["%.3f" % v for v in cur_free_params]))
                    print("cur_prop_mean: " + str(["%.3f" % v for v in prop_mean]))
                    print("cur_prop_stdev: " + str(["%.3f" % v for v in np.sqrt(np.diag(prop_hess))]))
                    print("cur_prop_value: {:.3f}".format(cur_prop))
                    print("")
                    print("prop_log_like: {:.3f}".format(prop_log_like))
                    print("cur_log_like: {:.3f}".format(cur_log_like))
                    print("log_like_diff: {:.3f}".format(log_like_diff))
                    print("")
                    print("log_prior_diff: {:.3f}".format(log_prior_diff))
                    print("log_prop_diff: {:.3f}".format(log_prop_diff))
                    print("log_jacob_diff: {:.3f}".format(log_jacob_diff))
                    print("")
                    print("accept_prob: {:.3f}".format(np.min((1.0,accept_prob))))
                    print("")

            self.accept_prob[self.current_iter] = np.min((1.0, accept_prob))

            self.prop_log_jacobian[self.current_iter] = prop_log_jacobian
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

    # Wrappers
    plot = plot_results
    print_progress_report = print_progress_report
    save_to_db = store_results_to_db
    save_to_file = store_results_to_file
    compute_ess = compute_ess
    compute_iact = compute_iact
    compute_sjd = compute_sjd
    zero_variance_linear = zero_variance_linear
