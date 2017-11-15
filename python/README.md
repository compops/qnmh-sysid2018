# Python code

## Installation
This code was developed using Anaconda 3 and Python 3.6.3. To satisfy these requirements, please download Anaconda from https://www.anaconda.com/ for your OS and some extra libraries by running
``` bash
conda install quandl pymongo palettable cython
```
If you opt to install python yourself some additional libraries are required. These can be installed using:
``` bash
pip install quandl pymongo numpy scipy pandas matplotlib palettable cython
```
The exact versions used when running the experiments in the paper are available in the `requirements.txt` file in the root folder of the repository.

### Cython
Some of the code is written in Cython and requires compilation before it can be run. Please execute the following from the root directory
``` bash
python setup.py build_ext --inplace
```
to compile this code. The number of particles and observations are hard-coded into the C-code due to the use of static arrays for speed. To change this, open the file corresponding to the model of interest and change the constants `NoParticles` and `NoObs` in the beginning of the file. Note that `NoObs` is T+1 (as we include the unknown initial state).

### Request Quandl API key
The run of example 3 requires that data is collected from Quandl for each simulation as due to Copyright reasons this data cannot be distributed along the source code. Quandl limits the number of data requests without a API key to 50 per day. Therefore it is advisable to register at Quandl and to enter you own API key in the file `python/scripts_draft1/helper_stochastic_volatility.py`.

## Reproducing the results in the paper

### Example 1: Linear Gaussian state space model using Kalman methods
**example1-lgss-kalman.py**
``` python
mh_settings = {'no_iters': 5000,
               'no_burnin_iters': 1000,
               'step_size': 0.8,
               'base_hessian': np.eye(3) * 0.05**2,
               'initial_params': (0.0, 0.1, 0.2),
               'hessian_correction_verbose': True,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_initial_hessian_scaling': 0.01,
               'verbose': False,
               'trust_region_size': 0.4,
               'qn_only_accepted_info': True,
               'qn_memory_length': 20
               }
```

``` python
mh_settings.update({'qn_strategy': 'bfgs'})
mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'flip'})
mh_settings.update({'qn_strategy': 'sr1', 'hessian_correction': 'replace'})
```

### Example 2: Linear Gaussian state space model using particle methods
**example2-lgss-particle.py**
``` python
pf_settings = {'resampling_method': 'systematic',
               'no_particles': 1000,
               'estimate_gradient': True,
               'estimate_hessian_segalweinstein': True,
               'fixed_lag': 10,
               'generate_initial_state': True
              }
```

### Example 3: Non-linear state space model using particle methods


## File structure
An overview of the file structure of the code base is found below.

* **data/** contains the datasets used in the paper.
* **helpers/** contains helpers for models, parameterisations, distributions, file management and connection to databases. These should not require any alterations.
* **models/** contains the different models used in the paper. It is here you need to add your own models (see below) if you want to use the code for some other problem.
* **parameter/** contains the MH algorithm and quasi-Newton methods for estimating the Hessian.
* **scripts/** contains helper scripts for the examples in the paper.
* **state/** contains the code for the Kalman filter/smoother, bootstrap particle filter and fixed-lag particle smoother. This code is quite general and should work for most scalar state space models. Cython versions are also available for the linear Gaussian and stochastic volatility models.
* **tests/** contains test scripts to validate the implementation. These are largely undocumented.

## Modifying the code for other models
This code is fairly general and can be used for inference in any state space model expressed by densities and with a scalar state. The main alteration required for using multivariate states is to rewrite the particle vector, propagation step and weighting step in the particle filter and smoother. As well as the standard generalisation of the Kalman filter and smoother.

The models are defined by files in models/. To implement a new model you can alter the existing models and re-define the functions `generate_initial_state`, `generate_state`, `evaluate_state`,  `generate_obs`, `evaluate_obs` and `check_parameters`. The names of these methods and their arguments should be self-explanatory. Furthermore, the gradients of the logarithm of the joint distribution of states and observation need to be computed by hand and entered into the code. The method `log_joint_gradient` is responsible for this computation.

In the paper, all model parameters are unrestricted and can assume any real value in the MH algorithm. This is enabled by reparametersing the model, which is always recommended for MH algorithms. This results in that the reparameterisation must be encoded in the methods `transform_params_to_free` and `transform_params_from_free`, where free parameters are the unrestricted versions. This also introduces a Jacobian factor into the acceptance probability encoded by `log_jacobian` as well as extra terms in the gradients and Hessians of both the log joint distribution of states and observations as well as the log priors. Please take good care when performing this calculations. For an example, see the supplementary material to the, which contains the required computations for the linear Gaussian state space model.

### Calibration of user settings

Furthermore, some alterations are probably required to the settings used in the quasi-Newton algorithm such as initial guess of the Hessian, a standard step length, memory length, etc.

Please, let me know if you need any help with this and I will try my best to sort it out.



