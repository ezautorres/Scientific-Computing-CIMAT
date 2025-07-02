# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 6 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise4_assignment6.py
@Main function: METROPOLIS_HASTINGS()
@Date         : 16 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Same as the previous exercise. but with another proposal distribution.

@Description  :
    Bernoulli random variables Be(1/3) are simulated for n = 5 and n = 50 and the number of successes in each case
    is calculated. The posterior distribution of p|x ~ p^r * (1-p)^(n-r) * cos(pi*p) * I[0<=p<= 0.5] is simulated
    for n = 5 and n = 50 using the Metropolis-Hastings algorithm. Truncated normal distributions at (0, 0.5) are
    proposed as transition distributions and the obtained distributions are compared with a histogram.
"""
from exercise1_assignment6 import SIMULATE_BERNOULLI  # Import the function to simulate Bernoulli random variables.
from exercise2_assignment6 import METROPOLIS_HASTINGS # Import the Metropolis-Hastings algorithm.
import numpy as np                                    # numpy library.
import matplotlib.pyplot as plt                       # matplotlib library.
from scipy.stats import uniform, norm                 # uniform and normal distributions.

# --------------------------------------------------------------------------------------------------------------
# Objetive Function: f(p|x) ~ p^r * (1-p)^(n-r) * cos(pi*p) * I[0 <= p <= 0.5] 
# and transition function: q(x) ~ N(x, sg^2) in (0, 0.5] with sg = 1e-1.
# --------------------------------------------------------------------------------------------------------------
# Objetive function.
def posterior(p: float, r: int, n: int) -> float:
    if p < 0 or p > 0.5:
        return 0
    return (p**r * (1-p)**(n-r) * np.cos(np.pi*p))
# Transition function.
def proposal_gen(p: float, sg: float) -> float: # Normal distribution truncated at (0,0.5).
    y_t = norm.rvs(loc = p, scale = sg)
    y_t = max(1e-7, min(0.5, y_t))              # Truncate the proposal.
    return y_t  
def proposal_pdf(p: float, sg: float) -> float: # Normal distribution truncated at (0,0.5).
    return norm.pdf(p, loc = p, scale = sg)

# --------------------------------------------------------------------------------------------------------------
# Simulation of the posterior distribution using the Metropolis-Hastings algorithm.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(14523) # Seed for reproducibility.
    # Parameters.
    p  = 1/3   # Probability of success.
    n1 = 5     # Number of Bernoulli trials.
    n2 = 50    # Number of Bernoulli trials.
    N  = 10000 # Number of iterations.
    sg = 0.1   # Standard deviation of the proposal distribution.

    # Simulate Bernoulli random variables.
    r1 = SIMULATE_BERNOULLI(n = n1, p = p)
    r2 = SIMULATE_BERNOULLI(n = n2, p = p) 
    x0 = uniform.rvs(0, 0.5) # Initial value in the support of the objetive function.
    print("Initial Point       :", x0)
    print("Successes for n = 5 :", r1)
    print("Successes for n = 50:", r2)

    # Simulate the posterior distribution.
    samples_n1 = METROPOLIS_HASTINGS(lambda p: posterior(p, r1, n1),
                                     lambda p: proposal_gen(p, sg = sg),
                                     lambda p: proposal_pdf(p, sg = sg),
                                     x0 = x0, N = N)
    samples_n2 = METROPOLIS_HASTINGS(lambda p: posterior(p, r2, n2),
                                     lambda p: proposal_gen(p, sg = sg),
                                     lambda p: proposal_pdf(p, sg = sg),
                                     x0 = x0, N = N)

    # Plot the results.
    plt.figure(figsize = (10,8))
    plt.hist(samples_n1, bins = 60, density = True, alpha = 0.6, label = f"$n = {n1}$, $r = {r1}$")
    plt.hist(samples_n2, bins = 60, density = True, alpha = 0.6, label = f"$n = {n2}$, $r = {r2}$")
    plt.legend()
    plt.xlabel("$p$")
    plt.ylabel("Density")
    plt.title("Simulated distributions with Metropolis-Hastings.")
    plt.savefig("MS_normal.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    plt.show()