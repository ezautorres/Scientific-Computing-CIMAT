# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 7 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment7.py
@Main function: METROPOLIS_HASTINGS().
@Date         : 23 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Simulate the distribution $Gamma(\alpha,1)$ with the proposed $Gamma([\alpha],1)$, where $[\alpha]$ denotes the
    integer part of $\alpha$. Also, perform the following experiment: set the initial point $x_0 = 950$ and graph
    the evolution of the chain, i.e., $f(X_t)$ vs $t$.

@Description  :
    A Markov chain is simulated using the Metropolis-Hastings algorithm (the function works for any dimension).
    Gamma(alpha, 1) random variables are simulated with 3 different alphas and the distributions obtained are
    compared with a histogram.
"""
import numpy as np              # numpy library.
from typing import Callable     # type hints.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.

# --------------------------------------------------------------------------------------------------------------
# Auxiliar function for graphics.
# --------------------------------------------------------------------------------------------------------------
def plot_distribution(samples: np.ndarray, x0: float, alpha: float, n_bins: int, filename: str = None) -> None:
    """
    This function plots the histogram of the samples, the objective function and the evolution of the Markov chain.

    Parameters
    ----------
    samples : array_like
        Generated samples.
    x0 : float
        Initial point.
    alpha : float
        Shape parameter of the Gamma distribution.
    n_bins : int
        Number of bins for the histogram.
    filename : str, optional
        Name of the file to save the plot.
    """
    x = np.linspace(min(samples)-1, max(samples), 1000)

    # Visualization of the distributions.
    fig, ax = plt.subplots(2, 1, figsize = (10,8))
    ax[0].scatter(x0, 0, color="red", label=rf"Initial Point $x_0={x0}$.", s=100)
    ax[0].hist(samples, bins=n_bins, density=True, label=f"Histogram.", color="black", alpha=0.5)    
    ax[0].plot(x, stats.gamma.pdf(x, a=alpha, scale=1), color="blue", label="Objetive Distribution.")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$f(x)$")
    ax[0].set_title(rf"Simulation of the random variables $Ga(\alpha={alpha},1)$.") 
    ax[0].legend()
    
    ax[1].plot(samples, '-', label="Evolution of the Markov Chain.", color="blue", alpha=0.7)
    ax[1].scatter(0, x0, color="red", label=rf"Initial Point $x_0={x0}$.", s=100)
    ax[1].set_xlabel(r"Iterations ($r$)")
    ax[1].set_ylabel(r"$X_t$")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------
# Metropolis-Hastings for R^n.
# --------------------------------------------------------------------------------------------------------------
def METROPOLIS_HASTINGS(f: Callable, q_gen: Callable, q_pdf: Callable, x0: np.ndarray, N: int) -> np.ndarray:
    """
    This function simulates a Markov chain using the Metropolis-Hastings algorithm. It is assumed that the proposal q
    is not symmetric.

    Parameters
    ----------
    f : Callable
        Objective function.
    q_gen : Callable
        Proposal generator.
    q_pdf : Callable
        Proposal density function.
    x0 : array_like
        Initial point.
    N : int
        Number of iterations.

    Returns
    -------
    chain : np.ndarray
        Simulated Markov chain.
    """
    chain = np.zeros((N,len(x0))) # Markov chain in R^n.
    chain[0] = x0                 # Initial point.
    accept = 0                    # Number of accepted proposals.
    for i in range(1, N):
        xt = chain[i-1] # Current point.
        yt = q_gen(xt)  # Proposal for the next point.
        try:
            div = f(yt)/f(xt) * q_pdf(xt)/q_pdf(yt) # Ratio of the objective functions and the proposal densities.
        except ZeroDivisionError:
            div = 0
        rho = min(1, div) # Acceptance ratio.
        if np.random.uniform(0,1) < rho: 
            chain[i] = yt # Aceptance.
            accept += 1
        else:                           
            chain[i] = xt # Rejection.
            
    print (f"Acceptance rate: {accept / N * 100:.2f}%")
    return chain

# --------------------------------------------------------------------------------------------------------------
# Objetive Function: f(x) ~ Gamma(alpha, 1)
# --------------------------------------------------------------------------------------------------------------
def f(x: float, alpha: float) -> float:
    """
    Objetive Function: Gamma(alpha, 1).
    """
    if x > 0:
        return stats.gamma.pdf(x = x, a = alpha, scale = 1)
    return 0

# --------------------------------------------------------------------------------------------------------------
# Proposed function: q(x) ~ Gamma([alpha], 1) and its probability density function.
# --------------------------------------------------------------------------------------------------------------
def q_gen(xt: float, alpha: float) -> float:
    """
    Simulates a proposal for the Gamma([alpha], 1) distribution, where [alpha] is the integer part of alpha.
    """
    return stats.gamma.rvs(a = np.floor(alpha), scale = 1)
def q_pdf(x: float, alpha: float) -> float:
    """
    Probability density function of the proposed Gamma([alpha], 1), where [alpha] is the integer part of alpha.
    """
    return stats.gamma.pdf(x = x, a = np.floor(alpha), scale = 1)

# --------------------------------------------------------------------------------------------------------------
# Implementation of the algorithm. Gamma(alpha, 1) random variables with 3 different alphas are simulated.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(145623) # Seed for reproducibility.
    
    # Parameters.
    N      = 10000 # Number of iterations.
    n_bins = 200   # Number of bins for the histogram.
    x0     = 950   # Initial point.
    x0_    = [x0]  # Initial point as a list.

    # --------------------------------------------------------------
    # Example 1: alpha ~ U(x0-200, x0+200)
    # --------------------------------------------------------------
    alpha = np.random.uniform(x0-200, x0+200) # Shape parameter of the Gamma distribution.
    print(f"\nShape parameter for example 1: {alpha}")
    samples = METROPOLIS_HASTINGS(f = lambda x: f(x, alpha),          # Objective function.
                                  q_gen = lambda x: q_gen(x, alpha),  # Proposal function.
                                  q_pdf = lambda x: q_pdf(x, alpha),  # Proposal density function.
                                  x0 = x0_, N = N)                    # Initial point and number of iterations.
    
    plot_distribution(samples, x0, alpha, n_bins, "example1_ex1.pdf") # Visualization of the distributions.

    # --------------------------------------------------------------
    # Example 2: alpha ~ U(x0-500, x0)
    # --------------------------------------------------------------
    alpha = np.random.uniform(x0-500, x0) # Shape parameter of the Gamma distribution.
    print(f"\nShape parameter for example 2: {alpha}")
    samples = METROPOLIS_HASTINGS(f = lambda x: f(x, alpha),          # Objective function.
                                  q_gen = lambda x: q_gen(x, alpha),  # Proposal function.
                                  q_pdf = lambda x: q_pdf(x, alpha),  # Proposal density function.
                                  x0 = x0_, N = N)                    # Initial point and number of iterations.
    
    plot_distribution(samples, x0, alpha, n_bins, "example2_ex1.pdf") # Visualization of the distributions.

    # --------------------------------------------------------------
    # Ejemplo 3: alpha ~ U(x0, x0+500)
    # --------------------------------------------------------------
    alpha = np.random.uniform(x0, x0+500) # Shape parameter of the Gamma distribution.
    print(f"\nShape parameter for example 3: {alpha}")
    samples = METROPOLIS_HASTINGS(f = lambda x: f(x, alpha),          # Objective function.
                                  q_gen = lambda x: q_gen(x, alpha),  # Proposal function.
                                  q_pdf = lambda x: q_pdf(x, alpha),  # Proposal density function.
                                  x0 = x0_, N = N)                    # Initial point and number of iterations.
    
    plot_distribution(samples, x0, alpha, n_bins, "example3_ex1.pdf") # Visualization of the distributions.