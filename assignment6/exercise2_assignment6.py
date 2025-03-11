# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 6 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment6.py
@Main function: METROPOLIS_HASTINGS()
@Date         : 16 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the Metropolis-Hastings algorithm to simulate the posterior.
        f(p|x) ~ p^r * (1-p)^(n-r) * cos(pi*p) * I[0 <= p <= 0.5]
    with the two cases of n and r above. To do this, put the proposal
        p'|p= p' ~ Beta(r + 1, n - r + 1) 
    and the initial distribution of the chain µ ~ U(0,1/2).

@Description  :
    Bernoulli random variables Be(1/3) are simulated for n = 5 and n = 50 and the number of successes in each case
    is calculated. The posterior distribution of p|x ~ p^r * (1-p)^(n-r) * cos(pi*p) * I[0<=p<= 0.5] is simulated
    for n = 5 and n = 50 using the Metropolis-Hastings algorithm. Beta(r+1, n-r+1) distributions are proposed as
    transition distributions and the obtained distributions are compared with a histogram.
"""
from exercise1_assignment6 import SIMULATE_BERNOULLI # Import the function to simulate Bernoulli random variables.
import numpy as np                                   # numpy library.
import matplotlib.pyplot as plt                      # matplotlib library.
from scipy.stats import beta, uniform                # beta and uniform distributions.
from typing import Callable                          # Type hinting.

def METROPOLIS_HASTINGS(f: Callable, q_gen: Callable, q_pdf, x0: float, N: int = 10000) -> np.ndarray:
    """
    This function simulates a Markov chain using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    f : Callable
        Objetive function.
    q_gen : Callable
        Proposal distribution.
    q_pdf : Callable
        Proposal distribution density.
    x0 : float
        Initial value.
    N : int, optional
        Number of iterations (default = 10000).
    
    Returns
    -------
    x : np.ndarray
        Simulated Markov chain.
    """
    x = np.zeros(N)         # Markov chain.
    x[0] = x0               # Initial value.
    for i in range(1, N):   # Iterations.
        y_t = q_gen(x[i-1]) # Propose a new value.
        rho = min(1, f(y_t) / f(x[i-1])*q_pdf(x[i-1]) / q_pdf(y_t)) # Acceptance ratio.
        if np.random.uniform(0,1) < rho: # Accept the proposal.
            x[i] = y_t    # Accept the proposal.
        else:
            x[i] = x[i-1] # Reject the proposal.
    return x

if __name__ == "__main__":
# --------------------------------------------------------------------------------------------------------------
# Objetive Function: f(p|x) ~ p^r * (1-p)^(n-r) * cos(pi*p) * I[0 <= p <= 0.5] 
# and transition function: q(x) ~ Beta(r+1, n-r+1).
# --------------------------------------------------------------------------------------------------------------
    # Objetive Function.
    def posterior(p: float, r: int, n: int) -> float: # Equation (1) in the report.
        if p < 0 or p > 0.5:
            return 0
        return (p**r * (1-p)**(n-r) * np.cos(np.pi * p))
    # Transition Function.
    def proposal_gen(p: float, r: int, n: int) -> float: # Independent proposal.
        return beta.rvs(r+1, n-r+1) 
    def proposal_pdf(p: float, r: int, n: int) -> float: # Proposal density.
        return beta.pdf(p,r+1, n-r+1)                      

# --------------------------------------------------------------------------------------------------------------
# Simulation of the posterior with Metropolis-Hastings.
# --------------------------------------------------------------------------------------------------------------

    np.random.seed(14523) # Seed for reproducibility.
    # Parameters.
    p  = 1/3   # Probability of success.
    n1 = 5     # Number of random variables.
    n2 = 50    # Number of random variables.
    N  = 10000 # Number of iterations.

    # Number of successes.
    r1 = SIMULATE_BERNOULLI(n = n1, p = p)
    r2 = SIMULATE_BERNOULLI(n = n2, p = p) 
    x0 = uniform.rvs(0, 0.5) # Initial value in the support of the objective function.
    print("Initial point       :", x0)
    print("Successes for n = 5 :", r1)
    print("Successes for n = 50:", r2)

    # Execute Metropolis-Hastings.
    samples_n1 = METROPOLIS_HASTINGS(lambda p: posterior(p, r1, n1),
                                     lambda p: proposal_gen(p, r = r1, n = n1),
                                     lambda p: proposal_pdf(p, r = r1, n = n1),
                                     x0 = x0, N = N)
    samples_n2 = METROPOLIS_HASTINGS(lambda p: posterior(p, r2, n2),
                                     lambda p: proposal_gen(p, r = r2, n = n2),
                                     lambda p: proposal_pdf(p, r = r2, n = n2),
                                     x0 = x0, N = N)

    # Visualization of the distributions.
    plt.figure(figsize = (10,8))
    plt.hist(samples_n1, bins = 50, density = True, alpha = 0.6, label = f"$n = {n1}$, $r = {r1}$")
    plt.hist(samples_n2, bins = 50, density = True, alpha = 0.6, label = f"$n = {n2}$, $r = {r2}$")
    plt.title("Simulated distributions with Metropolis-Hastings.")
    plt.xlabel("$p$")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("MS_bernoulli.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()