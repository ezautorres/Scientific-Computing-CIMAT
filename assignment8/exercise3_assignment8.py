# --------------------------------------------------------------------------------------------------------------
# Exercise 3 | Assignment 8 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise3_assignment8.py
@Main function: METROPOLIS_HASTINGS_SIM().
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
from exercise1_assignment8 import METROPOLIS_HASTINGS_HYBRID_KERNELS # Import the Metropolis-Hastings algorithm.

# --------------------------------------------------------------------------------------------------------------
# Auxiliary function for plotting histograms of Markov chains.
# --------------------------------------------------------------------------------------------------------------
def plot_chains(chain: np.ndarray, bins: int = 35, ax: plt.Axes = None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize = (15,6))

    # Histogram of lambdas.
    for j in range(chain.shape[1]-1):
        ax[0].hist(chain[:,j], bins = bins, density = True, alpha = 0.6, label = rf'$\lambda_{{{j+1}}}$.')
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Density")
    ax[0].set_title(r"Distribution of $\lambda$'s.")
    ax[0].legend()

    # Histogram of beta.
    ax[1].hist(chain[:, -1], bins = bins, density = True, alpha = 0.8, label = r'$\beta$.', color = 'blue')
    ax[1].set_xlabel("Value")
    ax[1].set_ylabel("Density")
    ax[1].set_title(r"Distribution of $\beta$.")
    ax[1].legend()

    return ax

# --------------------------------------------------------------------------------------------------------------
# Objective function.
# --------------------------------------------------------------------------------------------------------------
def posterior(x: np.ndarray, t: np.ndarray, p: np.ndarray, alpha: float, gamma: float, delta: float) -> float:
    """
    Posterior distribution of the parameters of the pumps.

    Parameters
    ----------
    x : array_like
        Parameters of the posterior distribution. x[:-1] = lbs, x[-1] = beta.
    t : array_like
        Pump operating times.
    p : array_like
        Number of failures in the time t.
    alpha : float
        Parameter of the gamma distribution.
    gamma : float
        Parameter of the gamma distribution.
    delta : float
        Parameter of the gamma distribution.
    
    Returns
    -------
    float
        Posterior distribution.
    """
    if np.all(x > 0):                                                     # Verify that the parameters are in the distribution support
        lbs, beta = x[:-1], x[-1]                                         # Parameters of the posterior distribution.
        prior_beta = stats.gamma.pdf(beta, gamma, scale=1/delta)          # Prior of beta.
        prior_lambda = np.prod(stats.gamma.pdf(lbs, alpha, scale=1/beta)) # Prior of lambdas.
        likelihood = np.prod(stats.poisson.pmf(p, lbs * t))               # Likelihood.
        return prior_beta * prior_lambda * likelihood                     # Posterior.
    else:
        return 0
    
# --------------------------------------------------------------------------------------------------------------
# Proposed functions and probability densities.
# --------------------------------------------------------------------------------------------------------------

# Proposal lambdas ---------------------------------------------------------------------------------------------
def prop_lambda_i_gen(x: np.ndarray, i: int, t: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    """
    Generates a proposal for the i-th parameter lambda_i ~ Gamma(p_i + alpha, 1/(beta + t_i)). The parameters are
    the same as the objective function (posterior).
    """
    lbs, beta = x[:-1], x[-1]                     # Posterior distribution parameters.
    shape = p[i] + alpha                          # Shape parameter of the gamma distribution.
    rate = beta + t[i]                            # Rate parameter of the gamma distribution.
    propuesta = np.copy(x)                        # Copy the current parameters.
    propuesta[i] = np.random.gamma(shape, 1/rate) # Generate a proposal for the i-th parameter.
    return propuesta 

def prop_lambda_i_pdf(x: np.ndarray, y: np.ndarray, i: int, t: np.ndarray, p: np.ndarray, alpha: float) -> float:
    """
    Calculates the probability density of the proposal for the i-th parameter
        lambda_i ~ Gamma(p_i + alpha, 1/(beta + t_i)).
    The parameters are the same as the objective function (posterior) and returns the probability density value.
    """
    lbs, beta = x[:-1], x[-1]         # Posterior distribution parameters.
    lambdas_p, beta_p = y[:-1], y[-1] # Proposal parameters.
    shape = p[i] + alpha              # Shape parameter of the gamma distribution.
    rate = beta + t[i]                # Rate parameter of the gamma distribution.
    return stats.gamma.pdf(lambdas_p[i], shape, scale=1/rate) 

# Propuesta beta -----------------------------------------------------------------------------------------------
def prop_beta_gen(x: np.ndarray, t: np.ndarray, alpha: float, gamma: float, delta: float) -> np.ndarray:
    """
    Generates a proposal for parameter beta ~ Gamma(alpha*n + gamma, 1/(delta + sum(lambdas))). The parameters
    are the same as the objective function (posterior).
    """
    lbs, beta = x[:-1], x[-1]                      # Posterior distribution parameters.
    shape = len(t) * alpha + gamma                 # Shape parameter of the gamma distribution.
    rate = delta + np.sum(lbs)                     # Rate parameter of the gamma distribution.
    propuesta = np.copy(x)                         # Copy the current parameters.
    propuesta[-1] = np.random.gamma(shape, 1/rate) # Generate a proposal for the beta parameter.
    return propuesta

def prop_beta_pdf(x: np.ndarray, y: np.ndarray, t: np.ndarray, alpha: float, gamma: float, delta: float) -> float:
    """
    Calculates the probability density of the proposal for parameter
        beta ~ Gamma(alpha*n + gamma, 1/(delta + sum(lambdas))).
        The parameters are the same as the objective function (posterior) and returns the probability density value.
    """
    lbs, beta = x[:-1], x[-1]         # Posterior distribution parameters.
    lambdas_p, beta_p = y[:-1], y[-1] # Proposal parameters.
    shape = len(t) * alpha + gamma    # Shape parameter of the gamma distribution.
    rate = delta + np.sum(lbs)        # Rate parameter of the gamma distribution.
    return stats.gamma.pdf(beta_p, shape, scale=1/rate)

# Model parameters and pump data.
t = [94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.1, 10.48] # Operating times.
p = [5, 1, 5, 14, 3, 20, 1, 1, 4, 22]                                  # Number of failures in the time t.
    
# --------------------------------------------------------------------------------------------------------------
# Main code.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    np.random.seed(135) # Seed for reproducibility.
    
    alpha = 1.8                 # Parameter.
    gamma = 0.01                # Parameter.
    delta = 1.                  # Parameter.
    n     = len(t)              # Number of pumps.
    x0    = np.random.rand(n+1) # Initial point.
    N     = 15000               # Number of iterations.
    bn_in = 2000                # Burn-in.
    print(f"Initial Point:\n{x0}")

    # Generate the proposal functions and probability densities.
    props_gen = ([lambda x, i=i: prop_lambda_i_gen(x, i, t, p, alpha) for i in range(10)] +       # Proposals for lambdas.
                 [lambda x: prop_beta_gen(x, t, alpha, gamma, delta)])                            # Proposals for beta.

    props_pdf = ([lambda x, y, i=i: prop_lambda_i_pdf(x, y, i, t, p, alpha) for i in range(10)] + # Density for lambdas.
                 [lambda x, y: prop_beta_pdf(x, y, t, alpha, gamma, delta)])                      # Density for beta.

    # Running the Metropolis-Hastings algorithm with hybrid kernels.
    chain = METROPOLIS_HASTINGS_HYBRID_KERNELS(lambda x: posterior(x, t, p, alpha, gamma, delta),
                                               props_gen, props_pdf, None, x0, N)

    # Plot the chains.
    ax = plot_chains(chain[bn_in:])
    plt.savefig("exercise3.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()