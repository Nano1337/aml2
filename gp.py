import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import cho_factor,cho_solve # necessary for Cholesky factorization

jax.config.update("jax_enable_x64", True)

prng_key = random.key(0)

'''
Initialize the PRNG with unique `seed`.
'''
def init_prng(seed):
    global prng_key
    prng_key = random.PRNGKey(seed)
    return prng_key
#

'''
Whenever you call random, you need to pass in as the first argument a call to this function.
This will advance the PRNG.
'''
def grab_prng():
    global prng_key
    _,prng_key = random.split(prng_key)
    return prng_key
#

'''
Transform unconstrained hyperparameters to constrained (ensure strictly positive).
'''
def param_transform(unconstrained_hyperparams):
    return np.exp(unconstrained_hyperparams)
#

'''
Transform constrained hyperparameters to unconstrained
'''
def inverse_param_transform(hyperparams):
    return np.log(hyperparams)
#

'''
Evaluate the squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
NOTE: exclude adding the noise variance. This should be added to the covariance when considering just training data.
'''
def sqexp_cov_function(X1, X2, hyperparams):
    signal_var = hyperparams[1]
    length_scale = np.array(hyperparams[2:])  # Shape: (d,)

    # Prevent division by zero
    length_scale = np.maximum(length_scale, 1e-5)

    # Scale the inputs by length scales
    X1_scaled = X1 / length_scale
    X2_scaled = X2 / length_scale

    # Compute the squared Euclidean distance
    sq_dist = np.sum(X1_scaled**2, axis=1, keepdims=True) + \
              np.sum(X2_scaled**2, axis=1, keepdims=True).T - \
              2 * X1_scaled @ X2_scaled.T

    cov = signal_var * np.exp(-1 * sq_dist)  # Included factor of 0.5

    return cov
#
#
#

'''
Evaluate the Mahalanobis-based squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
NOTE: exclude adding the noise variance.
'''
def sqexp_mahalanobis_cov_function(X1, X2, hyperparams):
    signal_var = hyperparams[1]
    length_scale = np.array(hyperparams[2:])  

    inv_length_scale = 1.0 / length_scale  # Shape: (d,)
    
    # Compute the scaled differences
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # Shape: (n1, n2, d)
    diff_scaled = diff * inv_length_scale  # Broadcasting applies scaling per feature
    
    # Compute the squared Mahalanobis distance
    mahalanobis_dist = np.sum(diff_scaled ** 2, axis=-1)  # Shape: (n1, n2)
    
    # Compute the covariance matrix
    cov = signal_var * np.exp(-mahalanobis_dist)
    return cov
#

'''
Compute the log marginal likelihood.
This function should return another function (lml_function) that will be your objective function, passed to JAX for value_and_grad.
It should only require the unconstrained hyperparameters as input. In resppnse, JAX will return gradients for the hyperparameters.
The covariance function, X_train and Y_train will be referenced from within the lml_function.
'''
def log_marginal_likelihood(cov_func, X_train, Y_train):
    def lml_function(unconstrained_hyperparams):
        # transform to constrained space
        constrained_hyperparams = param_transform(unconstrained_hyperparams)
        noise_var = constrained_hyperparams[0]

        # compute covariance, perform Cholesky decomposition, solve linear system, compute log det
        n = len(X_train)
        K = cov_func(X_train, X_train, constrained_hyperparams) + noise_var * np.eye(n)
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y_train)
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))

        y_alpha = np.dot(Y_train, alpha)
        return -0.5 * y_alpha - 0.5 * log_det_K - 0.5 * n * np.log(2 * np.pi)

    return lml_function
#

'''
In the outer function, precompute what is necessary in forming the GP posterior (mean and variance).
The inner function will then actually compute the posterior, given test inputs X_star.
It should return a 2-tuple, consisting of the posterior mean and variance.
'''
def gp_posterior(cov_func, X_train, Y_train, hyperparams):
    # Precompute Cholesky decomposition
    K = cov_func(X_train, X_train, hyperparams) + hyperparams[0] * np.eye(len(X_train))
    L, lower = cho_factor(K)
    alpha = cho_solve((L, lower), Y_train)

    def posterior_predictive(X_star):
        # Compute covariance between X_star and X_train
        K_star = cov_func(X_star, X_train, hyperparams)

        # Compute mean
        mean = K_star @ alpha

        # Compute covariance among X_star
        K_star_star = cov_func(X_star, X_star, hyperparams)

        # Solve for v: L v = K_star.T
        v = cho_solve((L, lower), K_star.T)

        # Compute variance
        var = K_star_star - K_star @ v

        # Add noise variance to the diagonal if necessary
        var = np.diag(var) + hyperparams[0]

        return mean, var

    return posterior_predictive
#

'''
Compute the negative log of the predictive density, given (1) ground-truth labels Y_test, (2) the posterior mean for the test inputs,
(3) the posterior variance for the test inputs, and (4) the noise variance (to be added to posterior variance)
'''
def neg_log_predictive_density(Y_test, posterior_mean, posterior_var, noise_variance):
    # Compute the predictive variance including noise
    predictive_var = posterior_var + noise_variance

    # Compute negative log predictive density
    nll = 0.5 * np.sum((Y_test - posterior_mean)**2 / predictive_var + np.log(predictive_var) + np.log(2 * np.pi))
    return nll
#

'''
Your main optimization loop.
cov_func shoud be either sqexp_cov_function or sqexp_mahalanobis_cov_function.
X_train and Y_train are the training inputs and labels, respectively.
unconstrained_hyperparams_init is the initialization for optimization.
step_size is the gradient ascent step size.
T is the number of steps of gradient ascent to take.
This function should return a 2-tuple, containing (1) the results of optimization (unconstrained hyperparameters), and
(2) the log marginal likelihood at the last step of optimization.
'''
def empirical_bayes(cov_func, X_train, Y_train, unconstrained_hyperparams_init, step_size, T):
    lml_function = log_marginal_likelihood(cov_func, X_train, Y_train)
    grad_lml = jax.grad(lml_function)

    hyperparams = unconstrained_hyperparams_init
    for t in range(T):
        grad_val = grad_lml(hyperparams)
        hyperparams = hyperparams + step_size * grad_val

    final_lml = lml_function(hyperparams)
    return hyperparams, final_lml
#
