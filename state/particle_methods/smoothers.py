"""Bootstrap particle filter for linear Gaussian model"""
import numpy as np


def flps(particle_method, model):
    """Fixed-lag particle smoother"""

    no_obs = model.no_obs + 1
    no_params = model.no_params
    no_particles = particle_method.settings['no_particles']
    fixed_lag = particle_method.settings['fixed_lag']

    particles = particle_method.particles
    weights = particle_method.weights
    ancestors = particle_method.ancestors

    # Initalise variables
    smo_state_est = np.zeros((no_obs, 1))
    smo_gradient_est = np.zeros((no_params, no_obs))
    log_joint_gradient_estimate = np.zeros(no_params)

    smo_state_est[0] = particle_method.filt_state_est[0]
    smo_state_est[-1] = particle_method.filt_state_est[-1]

    # Run the fixed-lag smoother for the rest
    for i in range(0, no_obs-1):
        particle_indicies = np.arange(0, no_particles)
        lag = int(np.min((i + fixed_lag, no_obs - 1)))

        # Reconstruct particle trajectory
        curr_ancestor = particle_indicies
        for j in range(lag, i, -1):
            curr_ancestor = curr_ancestor.astype(int)
            next_ancestor = curr_ancestor.astype(int)
            curr_ancestor = ancestors[curr_ancestor, j].astype(int)

        # Estimate state
        weighted_particles = particles[curr_ancestor, i] * weights[:, lag]
        smo_state_est[i] = np.nansum(weighted_particles)

        # Estimate gradient
        if particle_method.settings['estimate_gradient']:
            gradient_at_i = model.log_joint_gradient(particles[next_ancestor, i+1],
                                                     particles[curr_ancestor, i],
                                                     i)

            j = 0
            for param in gradient_at_i:
                weighted_gradients = gradient_at_i[param] * weights[:, lag]
                smo_gradient_est[j, i] = np.nansum(weighted_gradients)
                j += 1

    if particle_method.settings['estimate_gradient']:
        log_joint_gradient_estimate = np.sum(smo_gradient_est, axis=1)

    return {'smo_state_est': smo_state_est,
            'log_joint_gradient_estimate': log_joint_gradient_estimate
           }