# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Final Project | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_final_project.py
@Main function: 
@Date         : 04 December 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the report of the assignment.

@Description  :
    This file contains the implementation of the Metropolis-Hastings algorithm with hybrid proposal kernels in R^n.
    It is used to estimate the parameters of a posterior distribution. The functions required for the calculation
    of the objective function, the proposals, and the probability densities of the proposals are implemented. In
    addition, the Markov chain is simulated and the results are visualized.
"""
import numpy as np              # numpy library.
from typing import Callable     # type hints.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.
from auxiliary_functions import estim_burn_in_and_modes, chain_histograms, marginal_evolution_burn_in, trayectory2d3d

# --------------------------------------------------------------------------------------------------------------
# 1.- METROPOLIS-HASTINGS FOR R^n WITH HYBRID KERNELS.
# --------------------------------------------------------------------------------------------------------------
def METROPOLIS_HASTINGS_HYBRID_KERNELS(f: Callable, props_gen: list, props_pdf: list, probs: list,
                                       x0: np.ndarray, N: int) -> np.ndarray:
    """
    This function simulates a Markov chain using the Metropolis-Hastings algorithm with hybrid proposal kernels in R^n.

    Parameters
    ----------
    f : Callable
        Objective function.
    props_gen : list
        List of functions that generate proposals.
    props_pdf : list
        List of functions that calculate the probability density of the proposals.
    probs : list
        Probabilities of the proposal kernels. If None, uniform probabilities are assumed.
    x0 : array_like
        Initial point.
    N : int
        Number of iterations.

    Returns
    -------
    chain : np.ndarray
        Simulated Markov chain in R^n.
    """
    K = len(props_gen)            # Number of kernels.
    if probs is None:
        probs = np.ones(K)/K      # Uniform probabilities.
    count = np.zeros(K)           # Acceptance count by kernel.
    chain = np.zeros((N,len(x0))) # Markov chain in R^n.
    chain[0] = x0                 # Initial point.
    accept = 0                    # Number of accepted proposals.
    for i in range(1, N):
        k = np.random.choice(K, p = probs) # Choose a kernel.
        count[k] += 1                      # Count the acceptance.
        xt = chain[i-1]                    # Current point.
        yt = props_gen[k](xt)              # Proposal for the next point.
        try:
            div = f(yt)/f(xt) * props_pdf[k](xt,yt)/props_pdf[k](yt,xt) # f(y)/f(x) * q(x|y) / q(y|x)
        except ZeroDivisionError:
            div = 0
        rho_k = min(1,div) # Acceptance ratio.
        if np.random.uniform(0,1) < rho_k:
            chain[i] = yt  # Aceptance.
            accept += 1
        else:
            chain[i] = xt  # Rejection.

    print(f"Acceptance rate            : {accept / N * 100:.2f}%")
    print("Acceptances by kernel      :", count)
    
    return chain

# --------------------------------------------------------------------------------------------------------------
# 2.- OBJETIVE FUNCTION AND POSTERIOR DISTRIBUTION.
# --------------------------------------------------------------------------------------------------------------
def posterior(params: np.ndarray, alpha: float, beta: float, x: np.ndarray, x_max: int, N_max: int) -> float:
    """
    This function calculates the posterior distribution. Estimation of the probability that the population is of
    size N and the probability of success is p, given the observed data x. It is assumed that x follows a binomial
    distribution. It is assumed that N follows a discrete distribution (which does not explicitly depend on N) and
    p follows a Beta distribution.

    Parameters
    ----------
    params : array_like
        Parameters N and p.
    alpha : float
        Parameter of the Beta distribution.
    beta : float
        Parameter of the Beta distribution.
    x : array_like
        Observed data.
    x_max : int
        Maximum value of the data.
    N_max : int
        Maximum size of the population.
    
    Returns
    -------
    float
        Posterior.
    """
    N, p = params                                                   # Parameters.
    if x_max <= N and N <= N_max and 0 < p and p < 1:               # Support of N and p.
        likelihood = np.prod([stats.binom.pmf(xi,N,p) for xi in x]) # Likelihood.
        #prior_N = 1 / (N_max + 1)                                  # Prior of N.
        prior_p = stats.beta.pdf(p, alpha, beta)                    # Prior of p.
        return likelihood * prior_p                                 # Posterior.
    else:
        return 0

# --------------------------------------------------------------------------------------------------------------
# 3.- PROPOSALS.
# --------------------------------------------------------------------------------------------------------------

# Proposal 1 ---------------------------------------------------------------------------------------------------
def prop1_gen(params: np.ndarray, m: int, alpha: float, beta: float, x_sum: int) -> np.ndarray:
    """
    This function generates a Gibbs proposal for p given N. A new value is proposed for p and N is kept fixed.

    Parameters
    ----------
    params : array_like
        Current parameters.
    m : int
        Number of observations.
    alpha : float
        Parameter of the Beta distribution.
    beta : float
        Parameter of the Beta distribution.
    x_sum : int
        Sum of the data.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    N, p = params                                                  # Current parameters.
    if beta + m * N - x_sum > 0:                                   # The shape parameter must be positive.
        pnew = stats.beta.rvs(alpha + x_sum, beta + m * N - x_sum) # Proposal for p.
    else:
        pnew = stats.beta.rvs(alpha + x_sum, 1)                    # Proposal for p.
    return np.array([N, pnew])                                     # N is kept fixed.

def prop1_pdf(params: np.ndarray, props: np.ndarray, m: int, alpha: float, beta: float, x_sum: int) -> float:
    """
    This function calculates the probability density of the Gibbs proposal for p given N.

    Parameters
    ----------
    params : array_like
        Current parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop1_gen.

    Returns
    -------
    float
        Probability density of the Gibbs proposal for p given N.
    """
    N, p       = params                               # Current parameters.
    Nnew, pnew = props                                # Proposed parameters.
    if beta + m * N - x_sum > 0:                      # N support.
        return stats.beta.pdf(pnew, alpha + x_sum, beta + m * N - x_sum)
    else:
        return stats.beta.pdf(pnew, alpha + x_sum, 1) # N support.
        
# Proposal 2 ---------------------------------------------------------------------------------------------------
def prop2_gen(params: np.ndarray, alpha: float, beta: float, x_max: int, r: int) -> np.ndarray:
    """
    This function generates a uniform proposal for N in the interval [x_max, x_max + r] and a prior proposal for p:
    p ~ Beta(alpha,beta). It turns out to be an independent proposal for N and p.

    Parameters
    ----------
    params : array_like
        Current parameters.
    alpha : float
        Parameter of the Beta distribution.
    beta : float
        Parameter of the Beta distribution.
    x_max : int
        Maximum value of the data.
    r : int
        Parameter for the discrete distribution of N.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    Nnew = np.random.randint(x_max, x_max + r) # Proposal for N.
    pnew = stats.beta.rvs(alpha, beta)         # Proposal for p.
    return np.array([Nnew, pnew])              # N and p proposed.

def prop2_pdf(params: np.ndarray, props: np.ndarray, alpha: float, beta: float, x_max: int, r: int) -> float:
    """
    This function calculates the probability density of the uniform proposal for N in the interval [x_max, x_max + r]
    and the prior proposal for p: p ~ Beta(alpha,beta).
    
    Parameters
    ----------
    params : array_like
        Current parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop2_gen.
    """
    Nnew, pnew = props # Proposed parameters.
    return (1 / r) * stats.beta.pdf(pnew, alpha, beta) 

# Proposal 3 ---------------------------------------------------------------------------------------------------
def prop3_gen(params: np.ndarray, x_sum: int, N_max: int, m: int) -> np.ndarray:
    """
    Hypergeometric proposal for N and p does not change. N_max is the maximum size of the population, m is the size
    of the sample (number of observations), and x_sum are the total observed successes in the data.
    
    Parameters
    ----------
    params : array_like
        Current parameters.
    x_sum : int
        Sum of the data.
    N_max : int
        Maximum size of the population.
    m : int
        Number of observations.
        
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    N, p      = params                                           # Current parameters.
    Nnew = stats.hypergeom.rvs(M = N_max, n = int(x_sum), N = m) # Proposal for N.
    return np.array([Nnew, p])                                   # p does not change.

def prop3_pdf(params: np.ndarray, props: np.ndarray, x_sum: int, N_max: int, m: int) -> float:
    """
    This function calculates the probability density of the hypergeometric proposal for N.

    Parameters
    ----------
    params : array_like
        Current parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop3_gen.
    """
    Nnew, pnew = props # Proposed parameters.
    return stats.hypergeom.pmf(k = m, M = N_max, n = int(x_sum), N = Nnew)

# Proposal 4 ---------------------------------------------------------------------------------------------------
def prop4_gen(params: np.ndarray, x_max: int, lb: int) -> np.ndarray:
    """
    Poisson proposal for N and p does not change. The parameter lb is the parameter of the Poisson distribution.

    Parameters
    ----------
    params : array_like
        Current parameters.
    x_max : int
        Maximum value of the data.
    lb : int
        Parameter of the Poisson distribution.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    N, p = params                        # Current parameters.
    Nnew = x_max + stats.poisson.rvs(lb) # Proposal for N (Poisson).
    return np.array([Nnew, p])           # N proposed and p does not change.

def prop4_pdf(params: np.ndarray, props: np.ndarray, lb: int) -> float:
    """
    This function calculates the probability density of the Poisson proposal for N.

    Parameters
    ----------
    params : array_like
        Current parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop4_gen.
    """
    N, p = params      # Current parameters.
    Nnew, pnew = props # Proposed parameters.
    return stats.poisson.pmf(Nnew, lb)

# Proposal 5 ---------------------------------------------------------------------------------------------------
def prop5_gen(params: np.ndarray) -> np.ndarray:
    """
    Random walk proposal for N and p does not change.

    Parameters
    ----------
    params : array_like
        Current parameters.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    N, p = params                        # Current parameters.
    Nnew = N + np.random.choice([-1, 1]) # Proposal for N (random walk).
    return np.array([Nnew, p])           # N proposed and p does not change.

def prop5_pdf(params: np.ndarray, props: np.ndarray) -> float:
    """
    This function calculates the probability density of the random walk proposal for N.
    """
    return 0.5

# --------------------------------------------------------------------------------------------------------------
# 4.- MAIN FUNCTION.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(1623) # Seed for reproducibility.

    # Initial data and parameters ------------------------------------------------------------------------------
    x       = np.array([7, 8, 6, 5, 2, 8, 6, 6, 7, 4, 8, 8, 6, 4, 8, 8, 10, 5, 4, 2])
    N_samp = 10000     # Number of samples to simulate the chain.
    x_max  = np.max(x) # Maximum value of the data.
    x_sum  = np.sum(x) # Sum of the data.
    m      = len(x)    # Number of observations.
    N_max  = 1000      # Maximum size of the population.
    alpha  = 1         # Parameter of the Beta distribution.
    beta   = 20        # Parameter of the Beta distribution.
    r      = 120       # Parameter for proposal 2 and the initial distribution of N.
    lb     = 400       # Parameter for proposal 4.
    x0     = np.array([np.random.randint(x_max, x_max + r), stats.uniform.rvs()]) # Initial point.
    
    # Definition of kernels: their proposal functions and probability densities -------------------------------
    props_gen = [lambda params: prop1_gen(params, m, alpha, beta, x_sum), # Gibbs proposal.
                 lambda params: prop2_gen(params, alpha, beta, x_max, r), # A prior proposal.
                 lambda params: prop3_gen(params, x_sum, N_max, m),       # Hypergeometric proposal.
                 lambda params: prop4_gen(params, x_max, lb),             # Poisson proposal.
                 lambda params: prop5_gen(params)]                        # Random walk proposal.
    props_pdf = [lambda params, props: prop1_pdf(params, props, m, alpha, beta, x_sum), # PDF proposal 1.
                 lambda params, props: prop2_pdf(params, props, alpha, beta, x_max, r), # PDF proposal 2.
                 lambda params, props: prop3_pdf(params, props, x_sum, N_max, m),       # PDF proposal 3.
                 lambda params, props: prop4_pdf(params, props, lb),                    # PDF proposal 4.
                 lambda params, props: prop5_pdf(params, props)]                        # PDF proposal 5.

    # Markov chain simulation using Metropolis-Hastings with hybrid kernels ----------------------------------
    chain = METROPOLIS_HASTINGS_HYBRID_KERNELS(lambda params: posterior(params, alpha, beta, x, x_max, N_max),
                                               props_gen, props_pdf, None, x0, N_samp)
    
    estim_burn_in_and_modes = estim_burn_in_and_modes(chain) # Estimation of burn-in and modes.
    burn_in = estim_burn_in_and_modes[0]                     # Burn-in.
    N_mode  = estim_burn_in_and_modes[1][0]                  # Mode of N.
    p_mode  = estim_burn_in_and_modes[1][1]                  # Mode of p.

    # Viewing the results -------------------------------------------------------------------------------------
    print("\nResults:" + "-"*50)
    print("The initial distribution is:", x0)
    print( f"Sum of the data            : {x_sum}")
    print( f"Maximum data value         : {x_max}")
    print( f"Burn-in                    : {burn_in}")
    print(rf"Mode of N                  : {N_mode}")
    print(rf"Mode of p                  : {p_mode}")

    # Graphs --------------------------------------------------------------------------------------------------

    # Histograms of the data and estimated binomial distribution.
    ax = chain_histograms(chain = chain, burn_in = burn_in, params = ["$N$", "$p$"],
                          modes = [N_mode, p_mode], bins = [45,45,5], x = x)
    plt.tight_layout()
    plt.savefig("histogram_ex1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # Evolution of the marginal distributions of the chain.
    ax = marginal_evolution_burn_in(chain = chain, burn_in = burn_in, params = ["$N$", "$p$"])
    plt.tight_layout()
    plt.savefig("marginal_evolution_ex1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # 2D and 3D trajectory of the chain.
    ax = trayectory2d3d(chain = chain, params = ["$N$", "$p$"], modes = [N_mode, p_mode])
    plt.tight_layout()
    plt.savefig("trayectory_ex1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()