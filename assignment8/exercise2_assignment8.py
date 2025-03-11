# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 8 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment8.py
@Main function: METROPOLIS_HASTINGS_HYBRID_KERNELS().
@Date         : 06 November 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the report of the assignment.
"""
import numpy as np              # numpy library.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.
from exercise1_assignment8 import METROPOLIS_HASTINGS_HYBRID_KERNELS # Import the function METROPOLIS_HASTINGS_HYBRID_KERNELS.

# --------------------------------------------------------------------------------------------------------------
# Auxiliar function to plot the histograms of the Markov chains.
# --------------------------------------------------------------------------------------------------------------
def plot_chains(chain: np.ndarray, alpha_t: float, lb_t: float, bins: int = 35, ax: plt.Axes = None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize = (12,4))

    # Histogram of alpha.
    ax[0].hist(chain[:,0], bins = bins, density = True, alpha = 0.8, color = 'blue', label = 'Posterior of $\\alpha$.')
    ax[0].axvline(alpha_t, color = 'red', linestyle = '--', label = r'Real $\alpha$.')
    ax[0].set_title(r'Posterior Distribution of $\alpha$.')
    ax[0].set_xlabel(r'$\alpha$')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Histogram of lb.
    ax[1].hist(chain[:,1], bins = bins, density = True, alpha = 0.8, color = 'green', label = 'Posterior of $\\lambda$.')
    ax[1].axvline(lb_t, color = 'red', linestyle = '--', label = r'Real $\lambda$.')
    ax[1].set_title(r'Posterior Distribution of $\lambda$.')
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel('Density')
    ax[1].legend()

    return ax

# --------------------------------------------------------------------------------------------------------------
# Objective function.
# --------------------------------------------------------------------------------------------------------------
def posterior(x: np.ndarray, t: np.ndarray, c: float, b: float) -> float:
    """
    This function calculates the posterior distribution of the Weibull distribution parameters alpha and lambda.

    Parameters:
    -----------
    x : array_like
        Weibull distribution parameters. x[0] = alpha, x[1] = lambda.
    t : array_like
        Failure times with Weibull distribution.
    c : float
        Parameter of the exponential and gamma distribution.
    b : float
        Parameter of the gamma distribution.

    Returns:
    --------
    float
        Value of the posterior distribution.
    """
    alpha, lb = x                                                      # Weibull distribution parameters.
    likelihood = np.prod(alpha*lb*t**(alpha-1) * np.exp(-lb*t**alpha)) # Likelihood.
    prior_alpha = stats.expon.pdf(alpha, scale = 1/c)                  # Prior of alpha.
    prior_lb_alpha = stats.gamma.pdf(lb, alpha, scale = 1/b)           # Prior of lambda given alpha.
    return likelihood * prior_alpha * prior_lb_alpha                   # Posterior.

# --------------------------------------------------------------------------------------------------------------
# Proposal functions and probability densities.
# --------------------------------------------------------------------------------------------------------------

# Proposal 1 ---------------------------------------------------------------------------------------------------
def prop1_gen(x: np.ndarray, t: np.ndarray, b: float) -> np.ndarray:
    """
    Generates proposal 1 for the given parameter alpha: lambda ~ Gamma(alpha + n, (b + sum(t^alpha)). The parameters
    are the same as the objective function (posterior) and leaves the parameter alpha fixed.
    """
    alpha, lb = x                      # Weibull distribution parameters.
    shape = alpha + len(t)             # Shape parameter.
    scale = 1 / (b + np.sum(t**alpha)) # Scale parameter.
    return np.array([alpha, stats.gamma.rvs(shape, scale = scale)])

def prop1_pdf(x: np.ndarray, y: np.ndarray, t: np.ndarray, b: float) -> float:
    """
    Computes the probability density of proposal 1 for the given parameter lambda alpha. The parameters are the same
    as the objective function (posterior) and returns the probability density value.
    """
    alpha_p, lb_p = y                  # Proposal parameters.
    alpha, lb = x                      # Weibull distribution parameters.
    shape = alpha + len(t)             # Shape parameter.
    scale = 1 / (b + np.sum(t**alpha)) # Scale parameter.
    return stats.gamma.pdf(lb_p, shape, scale = scale)

# Proposal 2 ---------------------------------------------------------------------------------------------------
def prop2_gen(x: np.ndarray, t: np.ndarray, b: float, c: float) -> np.ndarray:
    """
    Generates proposal 2 for the given parameter alpha lambda: alpha ~ Gamma(n+1, 1/(-log(b) - log(prod(t)) + c)).
    The parameters are the same as the objective function (posterior) and leaves the parameter lambda fixed.
    """
    alpha, lb = x                           # Weibull distribution parameters.
    r1 = np.prod(t)                         # Product of the data.
    shape = len(t) + 1                      # Shape parameter.
    scale = 1/(-np.log(b) - np.log(r1) + c) # Scale parameter.
    return np.array([stats.gamma.rvs(shape, scale = scale), lb])

def prop2_pdf(x: np.ndarray, y: np.ndarray, t: np.ndarray, b: float, c: float) -> float:
    """
    Computes the probability density of proposal 2 for the given parameter alpha lambda. The parameters are the same
    as the objective function (posterior) and returns the probability density value.
    """
    alpha_p, lb_p = y                       # Proposal parameters.
    alpha, lb = x                           # Weibull distribution parameters.
    r1 = np.prod(t)                         # Product of the data.
    shape = len(t) + 1                      # Shape parameter.
    scale = 1/(-np.log(b) - np.log(r1) + c) # Scale parameter.
    return stats.gamma.pdf(alpha_p, shape, scale = scale)

# Proposal 3 ---------------------------------------------------------------------------------------------------
def prop3_gen(x: np.ndarray, c: float, b: float) -> np.ndarray:
    """
    Generates proposal 3 for parameters alpha and lambda: alpha ~ Expon(c) and lambda ~ Gamma(alpha, b). The
    parameters are the same as the objective function (posterior) and returns a value for alpha and lambda.
    """
    alpha, lb = x # Weibull distribution parameters.
    return np.array([stats.expon.rvs(scale = 1/c), stats.gamma.rvs(alpha, scale = 1/b)])

def prop3_pdf(x: np.ndarray, y: np.ndarray, c: float, b: float) -> float:
    """
    Computes the probability density of proposal 3 for parameters alpha and lambda. The parameters are the same
    as the objective function (posterior) and returns the probability density value.
    """
    alpha_p, lb_p = y # Proposal parameters.
    alpha, lb = x     # Weibull distribution parameters.
    return stats.expon.pdf(alpha_p, scale = 1/c) * stats.gamma.pdf(lb_p, alpha, scale = 1/b)

# Proposal 4 ---------------------------------------------------------------------------------------------------
def prop4_gen(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Generates proposal 4 for parameters alpha and lambda: alpha ~ N(alpha, sigma) and lambda fixed. The parameters
    are the same as the objective function (posterior) and returns a value of alpha and lambda. Also, sigma is the
    standard deviation of the random error.
    """
    alpha, lb = x                  # Weibull distribution parameters.
    eps = stats.norm.rvs(0, sigma) # Random error.
    return np.array([alpha + eps, lb])

def prop4_pdf(x: np.ndarray, y: np.ndarray, sigma: float = 0.1) -> float:
    """
    Computes the probability density of proposal 4 for parameters alpha and lambda. The parameters are the same as
    the objective function (posterior) and returns the probability density value.
    """
    alpha_p, lb_p = y # Proposal parameters.
    alpha, lb = x     # Weibull distribution parameters.
    return stats.norm.pdf(alpha_p, alpha, sigma)

# --------------------------------------------------------------------------------------------------------------
# Main code.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parameters.
    np.random.seed(13)     # Seed for reproducibility.
    alpha_t, lb_t = 1., 1. # Real parameters of the Weibull distribution.
    c, b          = 1., 1. # Parameters of the exponential and gamma distributions (of the a priori).
    sigma         = 0.1    # Standard deviation of proposal 4.
    n  = 30                # Number of data.
    x0 = np.random.rand(2) # Initial point for the chain.
    N  = 10000             # Number of iterations.
    probs = [0.25, 0.25, 0.25, 0.25] # Probabilidades de seleccionar cada propuesta.

    # Simulation of Weibull distribution data with parameters alpha_t and lb_t.
    t = np.random.weibull(alpha_t, n) / lb_t

    # Generate the proposal functions and probability densities.
    props_gen = [lambda x: prop1_gen(x, t, b),    # Proposal 1.
                 lambda x: prop2_gen(x, t, b, c), # Proposal 2.
                 lambda x: prop3_gen(x, c, b),    # Proposal 3.
                 lambda x: prop4_gen(x, sigma)]   # Proposal 4.

    props_pdf = [lambda x, y: prop1_pdf(x, y, t, b),    # PDF Proposal 1.
                 lambda x, y: prop2_pdf(x, y, t, b, c), # PDF Proposal 2.
                 lambda x, y: prop3_pdf(x, y, c, b),    # PDF Proposal 3.
                 lambda x, y: prop4_pdf(x, y, sigma)]   # PDF Proposal 4.
    
    # Generate the Markov chain.
    chain = METROPOLIS_HASTINGS_HYBRID_KERNELS(f = lambda x: posterior(x, t, c, b),
                                               props_gen = props_gen, props_pdf = props_pdf,
                                               probs = probs, x0 = x0, N = N)

    # Plot the histograms of the Markov chains.
    ax = plot_chains(chain, alpha_t, lb_t)
    plt.savefig("exercise2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()