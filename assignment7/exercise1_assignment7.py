# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 7 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment7.py
@Main function: METROPOLIS_HASTINGS_SIM().
@Date         : 23 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the report of the assignment.

@Description  :
    A Markov chain is simulated using the Metropolis-Hastings algorithm to find the posterior distribution of the
    parameters alpha and beta of a Gamma(alpha, beta) distribution. Gamma(3,100) random variables are simulated
    for n = 5 and n = 40 and the obtained distributions are compared with a histogram.
"""
import numpy as np                                # numpy library.
from typing import Callable                       # type hints.
import scipy.stats as stats                       # scipy library.
import matplotlib.pyplot as plt                   # matplotlib library.
from scipy.special import gamma as gamma_function # gamma function.

# --------------------------------------------------------------------------------------------------------------
# Metropolis-Hastings for simetric proposals.
# --------------------------------------------------------------------------------------------------------------
def METROPOLIS_HASTINGS_SIM(f: Callable, q_gen: Callable, x0: np.ndarray, N: int) -> np.ndarray:
    """
    This function simulates a Markov chain using the Metropolis-Hastings algorithm in R^n. It is assumed that the
    proposal q is symmetric.

    Parameters
    ----------
    f : Callable
        Objective function.
    q_gen : Callable
        Proposal generator.
    x0 : array_like
        Initial point.
    N : int
        Number of iterations.

    Returns
    -------
    chain : np.ndarray
        Markov chain in R^n.
    """
    chain = np.zeros((N,len(x0))) # Markov chain in R^n.
    chain[0] = x0                 # Initial point.
    accept = 0                    # Number of accepted proposals.
    for i in range(1, N):
        xt = chain[i-1] # Current point.
        yt = q_gen(xt)  # Proposal for the next point.
        try:
            div = f(yt)/f(xt) # Ratio of the objective functions.
        except ZeroDivisionError:
            div = 0
        rho = min(1, div) # Acceptance ratio.
        if np.random.uniform(0,1) < rho: 
            chain[i] = yt # Aceptance.
            accept += 1
        else:                            
            chain[i] = xt # Rejection.
            
    print(f"Acceptance rate: {accept / N * 100:.2f}%")
    return chain

# --------------------------------------------------------------------------------------------------------------
# Function to simulate Ga(alpha, beta) random variables.
# --------------------------------------------------------------------------------------------------------------
def SIMULATE_GAMMA(alpha: float, beta: float, n: int = 10000) -> np.ndarray[float]:
    return stats.gamma.rvs(a = alpha, scale = 1/beta, size = n)

# --------------------------------------------------------------------------------------------------------------
# Function to calculate the posterior: f(alpha, beta | n, r1, r2).
# --------------------------------------------------------------------------------------------------------------
def posterior(params: np.ndarray[float,float], x: np.ndarray[float]) -> float:
    """
    This function calculates the posterior using the logarithm, this avoids numerical overflows. The formula is
    derived from the original posterior, but applying the logarithm in each term. At the end the exponential of the
    sum of the logarithms is returned.

    Parameters
    ----------
    params : np.ndarray[float,float]
        alpha and beta parameters.
    x : np.ndarray[float]
        Simulated sample.
    
    Returns
    -------
    float
        Posterior.
    """
    n = len(x)                     # Sample size.
    r1, r2 = np.prod(x), np.sum(x) # r1 and r2.
    alpha, beta = params           # alpha and beta. 
    if 1 <= alpha <= 4 and beta > 1:
        try:
            term1 = n * alpha * np.log(beta) - n*np.log(gamma_function(alpha))
            term2 = (alpha-1) * np.log(r1)
            term3 = -beta * (r2+1)
            return np.exp(term1+term2+term3)
        except:
            return 0
    return 0

# --------------------------------------------------------------------------------------------------------------
# Function to simulate the proposal of alpha and beta using a multivariate normal distribution.
# --------------------------------------------------------------------------------------------------------------
def proposal_gen(params: np.ndarray[float,float], sg1: float, sg2: float) -> np.ndarray[float,float]:
    """
    This function generates a symmetric proposal for alpha and beta using a multivariate normal distribution.

    Parameters
    ----------
    params : np.ndarray[float,float]
        alpha and beta parameters.
    sg1 : float
        Standard deviation for alpha.
    sg2 : float
        Standard deviation for beta.
    
    Returns
    -------
    np.ndarray[float,float]
        Proposal for alpha and beta.
    """
    return params + stats.multivariate_normal.rvs(mean=np.zeros(2), cov=[[sg1**2,0], [0,sg2**2]])

# --------------------------------------------------------------------------------------------------------------
# Main code.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(14623) # Seed for reproducibility.
    
    # Parameters.
    a_t    = 3     # Shape.
    b_t    = 100   # Scale.
    n_05   = 5     # Number of samples.
    n_40   = 40    # Number of samples.
    sg1    = 0.1   # Standard deviation for alpha.
    sg2    = 10    # Standard deviation for beta.
    N      = 10000 # Number of iterations.
    n_bins = 50    # Number of bins for the histograms.

    a_0    = stats.uniform.rvs(1,4) # Initial point for alpha.
    b_0    = stats.expon.rvs(1)     # Initial point for beta.
    x0     = [a_0, b_0]             # Initial point for the chain.
    print("\nInitial Point:", x0)
    
    # Simulación de variables aleatorias.
    x_05 = SIMULATE_GAMMA(a_t, b_t, n_05) # Simulation of Gamma(3,100) random variables for n = 5.
    x_40 = SIMULATE_GAMMA(a_t, b_t, n_40) # Simulation of Gamma(3,100) random variables for n = 40.
    
    # Markov chain simulation for n = 5.
    print("\nMarkov chain simulation for n = 5.")
    chain_05 = METROPOLIS_HASTINGS_SIM(f = lambda x: posterior(x, x_05),
                                       q_gen = lambda x: proposal_gen(x, sg1, sg2),
                                       x0 = x0, N = N)
    alpha_chain_05, beta_chain_05 = chain_05[:,0], chain_05[:,1]
    mean_05 = np.mean(chain_05, axis = 0)
    print(f"Mean for n = {n_05} : {mean_05}")

    # Markov chain simulation for n = 40.
    print("\nMarkov chain simulation for n = 40.")
    chain_50 = METROPOLIS_HASTINGS_SIM(f = lambda x: posterior(x, x_40),
                                       q_gen = lambda x: proposal_gen(x, sg1, sg2),
                                       x0 = x0, N = N)
    alpha_chain_50, beta_chain_50 = chain_50[:,0], chain_50[:,1]
    mean_50 = np.mean(chain_50, axis = 0)
    print(f"Mean for n = {n_40}: {mean_50}")

    # Histograms.
    plt.figure(figsize = (10,8))
    plt.hist(alpha_chain_05, bins = n_bins, density = True, alpha = 0.7, label = f"$n = {n_05}$.")
    plt.hist(alpha_chain_50, bins = n_bins, density = True, alpha = 0.7, label = f"$n = {n_40}$.")
    plt.title(r"Histogram of the random variables $\alpha$.")
    plt.legend()
    plt.grid(True)
    plt.savefig("histogram_n5.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    plt.figure(figsize = (10,8))
    plt.hist(beta_chain_05, bins = n_bins, density = True, alpha = 0.7, label = f"$n = {n_05}$.")
    plt.hist(beta_chain_50, bins = n_bins, density = True, alpha = 0.7, label = f"$n = {n_40}$.")
    plt.legend()
    plt.grid(True)
    plt.title(r"Histogram of the random variables $\beta$.")
    plt.savefig("histogram_n40.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # Scatter plot.
    plt.figure(figsize = (10,8))
    plt.scatter(a_0, b_0, color = "black", label = f"Initial Point: [{x0}]", s = 100)
    plt.plot(alpha_chain_05, beta_chain_05, label = f"Trayectory for $n = {n_05}$.", marker='o', linestyle='-', alpha=0.3)
    plt.plot(alpha_chain_50, beta_chain_50, label = f"Trayectory for $n = {n_40}$.", marker='o', linestyle='-', alpha=0.3)
    plt.scatter(mean_05[0], mean_05[1], color = "blue", label = f"Mean for $n = {n_05}$.", s = 70, zorder = 20)
    plt.scatter(mean_50[0], mean_50[1], color = "red",  label = f"Mean for $n = {n_40}$.", s = 70, zorder = 20)
    plt.legend()
    plt.grid(True)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title("Simulated Markov Chain for the parameters $\\alpha$ and $\\beta$.")
    plt.savefig("trayectory_ex1.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    plt.show()