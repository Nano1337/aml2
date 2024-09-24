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
    pass
#

'''
Transform constrained hyperparameters to unconstrained
'''
def inverse_param_transform(hyperparams):
    pass
#

'''
Evaluate the squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
NOTE: exclude adding the noise variance. This should be added to the covariance when considering just training data.
'''
def sqexp_cov_function(X1, X2, hyperparams):
    pass
#

'''
Evaluate the Mahalanobis-based squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
NOTE: exclude adding the noise variance.
'''
def sqexp_mahalanobis_cov_function(X1, X2, hyperparams):
    pass
#

'''
Compute the log marginal likelihood.
This function should return another function (lml_function) that will be your objective function, passed to JAX for value_and_grad.
It should only require the unconstrained hyperparameters as input. In resppnse, JAX will return gradients for the hyperparameters.
The covariance function, X_train and Y_train will be referenced from within the lml_function.
'''
def log_marginal_likelihood(cov_func, X_train, Y_train):
    def lml_function(unconstrained_hyperparams):
        pass
    #

    return lml_function
#

'''
In the outer function, precompute what is necessary in forming the GP posterior (mean and variance).
The inner function will then actually compute the posterior, given test inputs X_star.
It should return a 2-tuple, consisting of the posterior mean and variance.
'''
def gp_posterior(cov_func, X_train, Y_train, hyperparams):
    def posterior_predictive(X_star):
        pass
    #

    return posterior_predictive
#

'''
Compute the negative log of the predictive density, given (1) ground-truth labels Y_test, (2) the posterior mean for the test inputs,
(3) the posterior variance for the test inputs, and (4) the noise variance (to be added to posterior variance)
'''
def neg_log_predictive_density(Y_test, posterior_mean, posterior_var, noise_variance):
    pass
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
    pass
#
