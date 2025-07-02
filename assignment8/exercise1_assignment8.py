# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 8 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment8.py
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
from typing import Callable     # type hints.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.

# --------------------------------------------------------------------------------------------------------------
# Function to plot the contours of the density and the evolution of the Markov chain.
# --------------------------------------------------------------------------------------------------------------
def contour_plot(chain: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, ax = None) -> plt.Axes:
    """
    This function plots the contours of the density and the evolution of the Markov chain in R^2.

    Parameters
    ----------
    chain : array_like
        Simulated Markov chain.
    mu : array_like
        Mean of the multivariate normal distribution.
    Sigma : array_like
        Covariance matrix of the multivariate normal distribution.
    ax : plt.Axes
        Axis of the figure.

    Returns
    -------
    ax : plt.Axes
        Axis of the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8)) # Figure and axis.
    x_min, x_max = np.min(chain[:,0])-1, np.max(chain[:,0])+1
    y_min, y_max = np.min(chain[:,1])-1, np.max(chain[:,1])+1
    x, y = np.mgrid[x_min:x_max:0.1, y_min:y_max:0.1]
    pos = np.dstack((x,y))
    rv = stats.multivariate_normal(mu, Sigma)
    contour = ax.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis') # Contours of the density.
    # Plot the initial point and the evolution of the Markov chain.
    ax.scatter(chain[0,0], chain[0,1], color='red', label=f'Initial Point: [{chain[0,0]},{chain[0,1]}].', s = 100, zorder=20)
    ax.plot(chain[:,0], chain[:,1], alpha=0.3, color='yellow', marker='o', linestyle='-', markersize = 1, label='Evolution of the Markov Chain.', zorder=20)
    ax.legend()
    ax.set_title('Bivariate normal contours and Markov chain evolution.')
    ax.set_xlabel('$x_1$.')
    ax.set_ylabel('$x_2$.')
    ax.grid(True)

    return ax

# --------------------------------------------------------------------------------------------------------------
# Metropolis-Hastings for R^n.
# --------------------------------------------------------------------------------------------------------------
def METROPOLIS_HASTINGS_HYBRID_KERNELS(f: Callable, props_gen: list, props_pdf: list, probs: list, x0: np.ndarray, N: int) -> np.ndarray:
    """
    This function simulates a Markov chain using the Metropolis-Hastings algorithm with hybrid proposal kernels in R^n.

    Parameters
    ----------
    f : Callable
        Objective function.
    props_gen : list
        List of functions that generate proposals.
    props_pdf : list
        List of functions that calculate the probability density of proposals.
    probs : list
        List of probabilities of selecting each proposal kernel. They must add up to 1.
    x0 : np.ndarray
        Initial value of the Markov chain.
    N : int
        Number of samples to generate.

    Returns
    -------
    chain : np.ndarray
        Simulated Markov chain.
    """
    n = len(props_gen)            # Number of proposal kernels.
    if probs is None:
        probs = np.ones(n)/n      # Equal probabilities for each kernel.
    count = np.zeros(n)           # Count of acceptances by kernel.
    chain = np.zeros((N,len(x0))) # Markov chain.
    chain[0] = x0                 # Initial value of the chain.
    accept = 0                    # Count of acceptances.
    for i in range(1, N):
        k = np.random.choice(n, p = probs) # Select a kernel.
        count[k] += 1                      # Count the acceptance.
        xt = chain[i-1]                    # Current value of the chain.
        yt = props_gen[k](xt)              # Generate a proposal.
        try:
            div = f(yt)/f(xt) * props_pdf[k](xt, yt) / props_pdf[k](yt, xt) # f(y)/f(x) * q(x|y) / q(y|x)
        except ZeroDivisionError:
            div = 0
        rho_k = min(1, div) # Acceptance ratio.
        if np.random.uniform(0,1) < rho_k:
            chain[i] = yt # Accept the proposal.
            accept += 1
        else:
            chain[i] = xt # Reject the proposal.

    print(f"Acceptance rate: {accept / N * 100:.2f}%")
    print("Acceptance count by kernel: ", count)

    return chain

# --------------------------------------------------------------------------------------------------------------
# Objective function: the bivariate normal distribution.
# --------------------------------------------------------------------------------------------------------------
def objetive(x: np.ndarray, mu1: float, mu2: float, sg1: float, sg2: float, rho: float) -> float:
    """
    This function evaluates the bivariate normal distribution.

    Parameters
    ----------
    x : array_like
        Vector of values in R^2.
    mu1 : float
        Mean of the first variable.
    mu2 : float
        Mean of the second variable.
    sg1 : float
        Standard deviation of the first variable.
    sg2 : float
        Standard deviation of the second variable.
    rho : float
        Correlation coefficient.

    Returns
    -------
    float
        Value of the bivariate normal distribution evaluated in x.
    """
    return stats.multivariate_normal.pdf(x, [mu1, mu2], [[sg1**2, rho*sg1*sg2],[rho*sg1*sg2, sg2**2]])

# --------------------------------------------------------------------------------------------------------------
# Conditional proposals for the bivariate normal distribution and its densities.
# --------------------------------------------------------------------------------------------------------------

# Proposal 1 ---------------------------------------------------------------------------------------------------
def prop1_gen(x: np.ndarray, mu1: float, mu2: float, sg1: float, sg2: float, rho: float) -> np.ndarray:
    """
    This function generates a conditional proposal for the first variable. The parameters are the same as in the
    objective function and it is assumed that the second variable is fixed.

    Returns
    -------
    np.ndarray
        Conditional proposal for the first variable.
    """
    x1, x2 = x                                      # Current values of the chain.
    mean = mu1 + rho * (sg1 / sg2) * (x2 - mu2)     # Mean of the proposal.
    std = sg1 * np.sqrt(1 - rho ** 2)               # Standard deviation of the proposal.

    return np.array([stats.norm.rvs(mean,std), x2]) # Equation 5 of the report.

def prop1_pdf(x: np.ndarray, y: np.ndarray, mu1: float, mu2: float, sg1: float, sg2: float, rho: float) -> float:
    """
    Probability density of the conditional proposal for the first variable. The parameters are the same as in the
    objective function and the second variable is assumed to be fixed, except for y which is the generated proposal.

    Returns
    -------
    float
        Probability density of the conditional proposal for the first variable.
    """
    x1, x2 = x                                # Current values of the chain.
    y1, y2 = y                                # Proposal.
    mean = mu1 + rho * (sg1 / sg2) * (x2-mu2) # Mean of the proposal.
    std = sg1 * np.sqrt(1 - rho**2)           # Standard deviation of the proposal.
    
    return stats.norm.pdf(y1, mean, std)      # Probability density of the proposal evaluated in y1.

# Proposal 2 ---------------------------------------------------------------------------------------------------
def prop2_gen(x: np.ndarray, mu1: float, mu2: float, sg1: float, sg2: float, rho: float) -> np.ndarray:
    """
    Generates the conditional proposal for the second variable. The parameters are the same as in the objective
    function and the first variable is assumed to be fixed.

    Returns
    -------
    np.ndarray
        Conditional proposal for the second variable.
    """
    x1, x2 = x                                      # Current values of the chain.
    mean = mu2 + rho * (sg2 / sg1) * (x1 - mu1)     # Mean of the proposal.
    std = sg2 * np.sqrt(1 - rho ** 2)               # Standard deviation of the proposal.
    
    return np.array([x1, stats.norm.rvs(mean,std)]) # Equation 6 of the report.

def prop2_pdf(x: np.ndarray, y: np.ndarray, mu1: float, mu2: float, sg1: float, sg2: float, rho: float) -> float:
    """
    Probability density of the conditional proposal for the second variable. The parameters are the same as in the
    objective function and the first variable is assumed to be fixed, except for y which is the generated proposal.

    Returns
    -------
    float
        Probability density of the conditional proposal for the second variable.
    """
    x1, x2 = x                              # Current values of the chain.
    y1, y2 = y                              # Proposal.
    mean = mu2 + rho * (sg2/sg1) * (x1-mu1) # Mean of the proposal.
    std = sg2 * np.sqrt(1 - rho**2)         # Standard deviation of the proposal.
    
    return stats.norm.pdf(y2, mean, std)    # Probability density of the proposal evaluated in y2.

# --------------------------------------------------------------------------------------------------------------
# Main code.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(14623) # Seed for reproducibility.
    # Parameters of the bivariate normal distribution.
    mu1, mu2 = np.random.uniform(0,10,2) # Random means in [0,10].
    sg1, sg2 = 1, 1      # Standard deviations.
    # Parameters of the Markov chain.
    rhos  = [0.85, 0.99] # Correlation coefficients.
    x0    = [10, 2.5]    # Initial point.
    N     = 100000       # Number of iterations.
    probs = [0.5, 0.5]   # Probabilities of selecting each proposal kernel.
    print(f"\nParameters of the bivariate normal distribution: \nmu1 = {mu1}, mu2 = {mu2}, sg1 = {sg1}, sg2 = {sg2}.")
    print(f"Initial Point: {x0}\n")

    for rho in rhos:
        print(50*"-", f"\nSimulation for rho = {rho}.")

        # The proposal generation functions and their corresponding densities are defined.
        props_gen = [lambda x: prop1_gen(x, mu1, mu2, sg1, sg2, rho),       # Proposal 1.
                     lambda x: prop2_gen(x, mu1, mu2, sg1, sg2, rho)]       # Proposal 2.
        props_pdf = [lambda x, y: prop1_pdf(x, y, mu1, mu2, sg1, sg2, rho), # Density of proposal 1.
                     lambda x, y: prop2_pdf(x, y, mu1, mu2, sg1, sg2, rho)] # Density of proposal 2.

        # Objective function.
        f = lambda x: objetive(x, mu1, mu2, sg1, sg2, rho)

        # Simulate the Markov chain.
        chain = METROPOLIS_HASTINGS_HYBRID_KERNELS(f, props_gen, props_pdf, probs, x0, N)
        
        # Plot the contours of the density and the evolution of the Markov chain.
        ax = contour_plot(chain, [mu1,mu2], [[sg1**2, rho*sg1*sg2], [rho*sg1*sg2, sg2**2]])
        plt.savefig(f"p{rho}.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
        plt.show()