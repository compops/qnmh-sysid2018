"""Helpers for displaying and storing results from MCMC algorithms."""
import sys
import copy
import time

import numpy as np
import matplotlib.pylab as plt

from helpers.database import db_insert_results
from helpers.file_system import write_to_json
from palettable.colorbrewer.qualitative import Dark2_8

# Print small progress reports
def print_progress_report(mcmc, max_iact_lag=100):
    """Plots progress report to the screen during a run of an MCMC algorithm."""
    iter = mcmc.current_iter

    print("###################################################################")
    print(" iter: " + str(iter + 1) + " of : "
          + str(mcmc.settings['no_iters']) + " completed.")
    print("")
    print(" Current state of the Markov chain:")
    print(["%.4f" % v for v in mcmc.params[iter - 1, :]])
    print("")
    print(" Proposed next state of the Markov chain:")
    print(["%.4f" % v for v in mcmc.prop_params[iter, :]])
    print("")
    print(" Current posterior mean estimate: ")
    print(["%.4f" % v for v in np.mean(mcmc.params[range(iter), :], axis=0)])
    print("")
    print(" Current acceptance rate:")
    print("%.4f" % np.mean(mcmc.accepted[range(iter)]))
    if (iter > (mcmc.settings['no_burnin_iters'] * 1.5)):
        print("")
        print(" Current IACT values:")
        print(["%.2f" % v for v in mcmc.compute_iact()])
        print("")
        print(" Current log-SJD value:")
        print(str(np.log(mcmc.compute_sjd())))
    if mcmc.settings['hessian_estimate'] is not 'kalman':
        if (iter > mcmc.settings['qn_memory_length']):
            no_samples_hess_est = mcmc.no_samples_hess_est[range(iter)]
            idx = np.where(no_samples_hess_est > 0)[0]
            if len(idx) > 0:
                print("")
                print(" Mean number of samples for Hessian estimate:")
                print("%.4f" % np.mean(no_samples_hess_est[idx]))

    print("###################################################################")

def plot_results(mcmc):
    """Plots results to the screen after a run of an MCMC algorithm."""
    no_iters = mcmc.settings['no_iters']
    no_burnin_iters = mcmc.settings['no_burnin_iters']
    params = mcmc.params[range(no_burnin_iters, no_iters), :]
    prop_params = mcmc.prop_params[range(no_burnin_iters, no_iters), :]
    prop_nat_grad = mcmc.prop_nat_grad[range(no_burnin_iters, no_iters), :]

    no_bins = int(np.floor(np.sqrt(len(params))))
    no_params = mcmc.model.no_params_to_estimate
    param_names = mcmc.model.params_to_estimate

    plt.figure()
    for i in range(no_params):
        col = Dark2_8.mpl_colors[i]
        plt.subplot(no_params, 4, 4 * i + 1)
        plt.hist(params[:, i], bins=no_bins, color = col)
        plt.ylabel("Marginal posterior probability of " + param_names[i])
        plt.xlabel("iter")
        plt.subplot(no_params, 4, 4 * i + 2)
        plt.plot(params[:, i], color = col)
        plt.ylabel("Parameter trace of " + param_names[i])
        plt.xlabel("iter")
        plt.subplot(no_params, 4, 4 * i + 3)
        plt.plot(prop_params[:, i], color = col)
        plt.ylabel("Proposed trace of " + param_names[i])
        plt.xlabel("iter")
        plt.subplot(no_params, 4, 4 * i + 4)
        plt.plot(prop_nat_grad[:, i], color = col)
        plt.ylabel("natural gradient of " + param_names[i])
        plt.xlabel("iter")
    plt.show()

def compile_results(mcmc, sim_name=None, sim_desc=None):
    """Compiles results after a run of an MCMC algorithm."""
    no_iters = mcmc.settings['no_iters']
    no_burnin_iters = mcmc.settings['no_burnin_iters']
    idx = range(no_burnin_iters, no_iters)
    current_time = time.strftime("%c")

    mcmcout = {}
    mcmcout.update({'params': mcmc.params[idx, :]})
    mcmcout.update({'prop_params': mcmc.prop_params[idx, :]})
    mcmcout.update({'states': mcmc.states[idx, :]})
    mcmcout.update({'accept_prob': mcmc.accept_prob[idx, :]})
    mcmcout.update({'no_samples_hess_est': mcmc.no_samples_hess_est[idx, :]})
    mcmcout.update({'nat_gradient': mcmc.nat_gradient[idx, :]})
    mcmcout.update({'hess': mcmc.hess[idx, :]})
    mcmcout.update({'simulation_description': sim_desc})
    mcmcout.update({'simulation_name': sim_name})
    mcmcout.update({'simulation_time': current_time})

    data = {}
    data.update({'observations': mcmc.model.obs})
    if mcmc.model.states is None:
        data.update({'states': mcmc.model.states})
    data.update({'simulation_description': sim_desc})
    data.update({'simulation_name': sim_name})
    data.update({'simulation_time': current_time})

    settings = copy.deepcopy(mcmc.settings)
    settings.update({'sampler_name': mcmc.name})
    settings.update({'simulation_description': sim_desc})
    settings.update({'simulation_name': sim_name})
    settings.update({'simulation_time': current_time})

    return mcmcout, data, settings

def store_results_to_file(mcmc, output_path=None, sim_name=None, sim_desc=None):
    if output_path is None:
        raise ValueError("No output path given...")

    mcout, data, settings = compile_results(mcmc, sim_name=sim_name,
                                            sim_desc=sim_desc)

    desc = {'description': settings['simulation_description'],
            'time': settings['simulation_time']
           }
    write_to_json(mcout, output_path, sim_name, 'mcmc_output.json')
    write_to_json(data, output_path, sim_name, 'data.json')
    write_to_json(settings, output_path, sim_name, 'settings.json')
    write_to_json(desc, output_path, sim_name, 'description.txt')

def store_results_to_db(mcmc, collection=None, sim_name=None, sim_desc=None):
    if collection is None:
        raise ValueError("No Mongo Database collection given...")

    mcout, data, settings = compile_results(mcmc, sim_name=sim_name,
                                            sim_desc=sim_desc)

    db_insert_results(collection, post_name, output=None, data=None, settings=None)