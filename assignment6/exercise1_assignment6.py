# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 6 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment6.py
@Main function: SIMULATE_BERNOULLI()
@Date         : 16 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Simulate n = 5 and n = 50 Bernoulli random variables Be(1/3); let r be the number of successes in each case.

@Description  :
    n = 5 and n = 50 Bernoulli Be(1/3) random variables are simulated and the number of successes in each case is
    calculated. The np.random.binomial() function of the numpy library is used to simulate the random variables.
"""
import numpy as np # numpy library.

# --------------------------------------------------------------------------------------------------------------
# Function to simulate Bernoulli random variables.
# --------------------------------------------------------------------------------------------------------------
def SIMULATE_BERNOULLI(n: int, p: float = 1/3) -> int:
    """
    This function simulates n Bernoulli Be(p) random variables and returns the number of successes.

    Parameters
    ----------
    n : int
        Number of Bernoulli random variables.
    p : float (default = 1/3)
        Probability of success.

    Returns
    -------
    r : int
        Number of successes.
    """
    samples = np.random.binomial(1, p, n) # Simulate n Bernoulli random variables.
    r = np.sum(samples)                   # Number of successes.
    return r

# --------------------------------------------------------------------------------------------------------------
# Simulation of Bernoulli random variables.
# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    np.random.seed(14523)                  # Seed for reproducibility.
    p  = 1/3                               # Probability of success.
    r1 = SIMULATE_BERNOULLI(n = 5, p = p)  # n = 5
    r2 = SIMULATE_BERNOULLI(n = 50, p = p) # n = 50

    # Results.
    print(f"Succeses for n = 5  : {r1}")
    print(f"Succeses for n = 50 : {r2}")