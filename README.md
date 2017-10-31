# Python code
This code was downloaded from https://github.com/compops/qnmh-sysid2018 and contains the code and data used to produce the results in the paper:

J. Dahlin, A. Wills and B. Ninness, ***. Pre-print, arXiv:1712:****, 2017.

The tutorial is available as a preprint from http://arxiv.org/pdf/1712****.

## Installation
This code was developed using Anaconda 3 and Python 3.6.3. To satisify these requirements, please download Anaconde from https://www.anaconda.com/ for your OS. In addition, some extra libraries are required. These can be installed using:
``` bash
pip install pymongo
```
Some of the code is written in Cython and requires compilation before it can be run. Please execute the following from the root directory
``` bash
python setup.py build_ext --inplace
```
to compile this code.

An alternative method to reproduce the results is to make use of the Docker image build from this repository. Docker enables you to recreate the computational environment used to create the results in the paper. Hence, it automatically downloads the correct version of Python and all dependencies.

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
**example3-phillips-particle.py**
``` python
mh_settings = {'no_iters': 5000,
               'no_burnin_iters': 1000,
               'step_size': 0.5,
               'base_hessian': np.eye(4) * 0.02**2,
               'initial_params': (0.45, 0.76, 0.0, 0.275),
               'hessian_correction_verbose': True,
               'qn_initial_hessian': 'scaled_gradient',
               'qn_initial_hessian_scaling': 0.001,
               'verbose': False,
               'trust_region_size': 0.4,
               'qn_only_accepted_info': True,
               'qn_memory_length': 20
               }
```


## File structure

* **data/** contains the datasets used in the paper.
* **helpers/** contains helpers for models, parameterisations, distributions, file mangement and connection to databases. These should not require any alterations.
* **models/** contains the different models used in the paper. It is here you need to add your own models (see below) if you want to use the code for some other problem.
* **parameter/** contains the MH algorithm and quasi-Newton methods for estimating the Hessian.
* **scripts/** contains helper scripts for the examples in the paper.
* **state/** contains the code for the Kalman filter/smoother, bootstrap particle filter and fixed-lag particle smoother. This code is quite general and should work for most scalar state space models. Cython versions are also available for the linear Gaussian model.
* **tests/** contains test scripts to validate the implementation. These are largely undocumented.

Results are available as a zip-file in the latest release in the Git repo.

## Modification of inference in other models
This code is fairly general and can be used for inference in any state space model expressed by densities and with a scalar state. The main alteration required for using multivariate states is to rewrite the particle vector, propagation step and weighting step in the particle filter and smoother. As well as the standard generalisation of the Kalman filter and smoother.

The models are defined by files in models/. To implement a new model you can alter the existing models and re-define the functions `generate_initial_state`, `generate_state`, `evaluate_state`,  `generate_obs`, `evaluate_obs` and `check_parameters`. The names of these methods and their arguments should be self explainatory. Furthermore, the gradients of the logarithm of the joint distribution of states and observation need to be computed by hand and entered into the code. The method `log_joint_gradient` is responsible for this computation.

In the paper, all model parameters are unrestricted and can assume any real value in the MH algorithm. This is enabled by reparametersing the model, which is always recommended for MH algorithms. This results in that the reparameterisation must be encoded in the methods `transform_params_to_free` and `transform_params_from_free`, where free parameters are the unrestricted versions. This also introduces a Jacobian factor into the acceptance probability encoded by `log_jacobian` as well as extra terms in the gradients and Hessians of both the log joint distribution of states and observations as well as the log priors. Please take good care when performing this calculations. For an example, see the suplimentary matrial to the, which contains the required computations for the linear Gaussian state space model.

Furthermore, some alterations are probably required to the settings used in the quasi-Newton algorithm such as initial guess of the Hessian, a standard step length, memory length, etc.

Please, let me know if you need any help with this and I will try my best to sort it out.



