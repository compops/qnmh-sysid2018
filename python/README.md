# Python code

## Installation
This code was developed using Anaconda 3 and Python 3.6.3. To satisfy these requirements, please download Anaconda from https://www.anaconda.com/ for your OS and some extra libraries by running
``` bash
conda install quandl pymongo palettable cython
```
If you opt to install python yourself some additional libraries are required. These can be installed by executing the following in the root directory:
``` bash
pip install -r requirements.txt
```
see the file for the exact versions used for running the experiments in the paper.

### Cython
Some of the code is written in Cython and requires compilation before it can be run. Please execute the following from the `python` directory
``` bash
python setup.py build_ext --inplace
```
to compile this code. The number of particles and observations are hard-coded into the C-code due to the use of static arrays for speed. To change this, open the file corresponding to the model of interest and change the constants `NoParticles` and `NoObs` in the beginning of the file. Note that `NoObs` is T+1 (as we include the unknown initial state).

### Request Quandl API key
The run of example 3 requires that data is collected from Quandl for each simulation as due to Copyright reasons this data cannot be distributed along the source code. Quandl limits the number of data requests without a API key to 50 per day. Therefore it is advisable to register at Quandl and to enter you own API key in the file `python/scripts_draft1/helper_stochastic_volatility.py`.

## Reproducing the results in the paper
The results in the paper can be reproduced by running the scripts found in the folder `scripts_draft1/`. Here, we discuss each of the three examples in details and provide some additional supplementary details, which are not covered in the paper. The results from each script is saved in the folder `results/` under sub-folders corresponding to the three different examples.

Examples 1 and 2 are repeated using different random seeds 25 times in a Monte Carlo simulation. The simplest way to execute these is to call the script `run_script.sh`, which will run all the experiments (note that this will take at least a day). Another way to execute a single experiment is to call

``` bash
python run_script.py experiment_number
```

where `experiment_number` is 1, 2 or 3. Note that this will still mean that 25 experiments are run for examples 1 and 2. To run only one repetition for a single experiment, change the code in `run_script.py` by removing the for-loop.


### Example 1: Linear Gaussian states-space model using Kalman methods
The script `example1_lgss_kalman.py` reproduces the first example in Section 5.1. The model is a linear Gaussian state-space model given by

``` python
x_{t+1} | x_t ~ N( x_{t+1}; mu + phi * (x_t - mu), sigma_v^2)
y_t     | x_t ~ N( y_t; x_t, 0.5^2)
```

where the unknown parameters are `(mu, phi, sigma_v)`. The parameters are estimated using a synthetic data set generated from the model. The value of the log-posterior and its gradients are computed using Kalman filtering and smoothing. The script makes use of the following settings for the Kalman methods:

``` python
kf_settings = {'initial_state': 0.0,
               'initial_cov': 1e-5,
               'estimate_gradient': True
              }
```
which means that the initial state is zero with the covariance 10^-5 and the gradients of the log-posterior are computed.

The Metropolis-Hastings algorithm makes use of the following settings:

``` python
mh_settings = {'no_iters': 10000,
               'no_burnin_iters': 3000,
               'step_size': 0.5,
               'base_hessian': hessian_estimate,
               'initial_params': (0.2, 0.5, 1.0),
               'verbose': False,
               'verbose_wait_enter': False,
               'trust_region_size': None,
               'hessian_estimate': None,
               'hessian_correction': 'replace',
               'hessian_correction_verbose': False,
               'no_iters_between_progress_reports': 1000,
               'qn_memory_length': 20,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_strategy': None,
               'qn_bfgs_curvature_cond': 'damped',
               'qn_initial_hessian_scaling': 0.01,
               'qn_initial_hessian_fixed': np.eye(3) * 0.01**2,
               'qn_only_accepted_info': True,
               'qn_accept_all_initial': True
               }
```
where the most important settings are `no_iters` (no. iterations), `no_burnin_iters` (no. burn-in iterations), `step_size` (epsilon in the paper), `base_hessian` (pre-conditioning matrix in paper) and `initial_params` (the point in which the Markov chain is initialised). For the quasi-Newton proposal, we have `qn_memory_length` (memory length M in paper), `qn_initial_hessian` (how the Hessian estimate is initialised) and `qn_initial_hessian_scaling` (the size of the initial step),

When running the various experiments, step lengths and similar are adjusted. For the quasi-Newton algorithms `hessian_correction` is used to determined how the negative definite Hessian estimates are corrected, see the paper for details.

### Example 2: Linear Gaussian state-space model using particle methods
The script `example2_lgss_particle.py` reproduce the second example in Section 5.2. The setup is the same as in example 1 but particle filtering and smoothing are used to estimate the log-posterior and its gradients. The settings for the particle methods are:

``` python
pf_settings = {'no_particles': 2000,
                'resampling_method': 'systematic',
                'fixed_lag': 10,
                'initial_state': 0.0,
                'generate_initial_state': True,
                'estimate_gradient': True,
                'estimate_hessian': True,
                }
```

which should be self-explanatory. Remember that the Cython code is used in the script and these settings are overridden by settings written directly as constants in the `.pyx`-file in the directory `state/particle_methods`.

### Example 3: Non-linear state space model using particle methods
The script `example3_stochastic_volatility_particle.py` reproduces the third example in Section 5.3. The model is a stochastic volatility model with leverage given by

``` python
x_{t+1} | x_t ~ N( x_{t+1}; mu + phi * (x_t - mu) + rho * sigma_v * exp(-xt/2) * y_t, sigma_v^2 (1 - rho^2))
y_t     | x_t ~ N( y_t; 0, exp(x_t))
```

where the unknown parameters are `(mu, phi, sigma_v, rho)`. The data is obtained from Quandl and contains the log-return from Bitcoins during a two year period. Settings are similar as for example 2.

## File structure
An overview of the file structure of the code base is found below.

* **data/** contains the datasets used in the paper.
* **helpers/** contains helpers for models, parameterisations, distributions, file management and connection to databases. These should not require any alterations.
* **models/** contains the different models used in the paper. It is here you need to add your own models (see below) if you want to use the code for some other problem.
* **parameter/** contains the MH algorithm and quasi-Newton methods for estimating the Hessian.
* **scripts/** contains helper scripts for the examples in the paper.
* **state/** contains the code for the Kalman filter/smoother, bootstrap particle filter and fixed-lag particle smoother. This code is quite general and should work for most scalar state space models. Cython versions are also available for the linear Gaussian and stochastic volatility models.

## Modifying the code for other models
This code is fairly general and can be used for inference in any state space model expressed by densities and with a scalar state. The main alteration required for using multivariate states is to rewrite the particle vector, propagation step and weighting step in the particle filter and smoother. As well as the standard generalisation of the Kalman filter and smoother.

The models are defined by files in models/. To implement a new model you can alter the existing models and re-define the functions `generate_initial_state`, `generate_state`, `evaluate_state`,  `generate_obs`, `evaluate_obs` and `check_parameters`. The names of these methods and their arguments should be self-explanatory. Furthermore, the gradients of the logarithm of the joint distribution of states and observation need to be computed by hand and entered into the code. The method `log_joint_gradient` is responsible for this computation.

In the paper, all model parameters are unrestricted and can assume any real value in the MH algorithm. This is enabled by reparametersing the model, which is always recommended for MH algorithms. This results in that the reparameterisation must be encoded in the methods `transform_params_to_free` and `transform_params_from_free`, where free parameters are the unrestricted versions. This also introduces a Jacobian factor into the acceptance probability encoded by `log_jacobian` as well as extra terms in the gradients and Hessians of both the log joint distribution of states and observations as well as the log priors. Please take good care when performing this calculations. For an example, see the supplementary material to the, which contains the required computations for the linear Gaussian state space model.

### Calibration of user settings
Furthermore, some alterations are probably required to the settings used in the quasi-Newton algorithm such as initial guess of the Hessian, a standard step length, memory length, etc.

Please, let me know if you need any help with this and I will try my best to sort it out.
