# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 5 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment5.py
@Main function: GENERATE_UNIFORM().
@Date         : 09 Octubre 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the following algorithm to simulate uniform random variables:
        x_i = 107374182 x_{i-1} + 104420 x_{i-5} mod 2^{31} - 1.
    return x_i and loop through the state, that is x_{j-1} = x_j; j = 1,2,3,4,5; do they look like U(0,1)?

@Description  :
    This script implements a function GENERATE_UNIFORM() that generates a sample of uniform random variables U(0,1)
    using the linear congruence method. The function receives as arguments the initial state of the sequence, the
    coefficients of the recursive relation:
        x_{i} = (107374182 * x_{i-1} + 104420 * x_{i-5}) mod 2^{31} - 1,
    the modulus of the recursive relation and the number of samples to generate. The function returns a list with
    the generated samples.
"""
import numpy as np              # numpy library.
import matplotlib.pyplot as plt # matplotlib library.

def GENERATE_UNIFORM(init_st: np.ndarray, a: int = 107374182, b: int = 104420, mod: int = 2**31 - 1, n: int = 100000):
    """
    The function GENERATE_UNIFORM() generates a sample of uniform random variables U(0,1) using the linear congruence
    method.

    Parameters
    ----------
    init_st : (5,) array_like
        Initial state of the sequence.
    a : int, optional
        Coefficient of x_{i-1} in the recursive relation (default is 107374182).
    b : int, optional
        Coefficient of x_{i-5} in the recursive relation (default is 104420).
    mod : int, optional
        Modulus of the recursive relation (default is 2^{31} - 1).
    n : int, optional
        Number of samples to generate (default is 100000).
    
    Returns
    -------
    results : list
        Sample of uniform random variables U(0,1).
    """
    if len(init_st) != 5:
        raise ValueError("The initial state must contain exactly 5 values.")
    
    state   = np.copy(init_st) # Copy of the initial state.
    results = []               # List to store the generated samples.
    
    for _ in range(n):
        x_i = (a * state[-1] + b * state[0]) % mod # Recursive relation.
        results.append(x_i / mod)                  # Normalized value in [0,1].
        state = np.roll(state, -1)                 # Move the elements one position to the left.
        state[-1] = x_i                            # Update the last element of the state.
    
    return results

# --------------------------------------------------------------------------------------------------------------
# USAGE EXAMPLE
# --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    init_st = np.random.rand(5)                 # Random initial state.
    n = 100000                                  # Number of samples to generate.
    uniform_vals = GENERATE_UNIFORM(init_st, n) # Generate the samples.
    print("Initial State:\n", init_st)

    # Plot the histogram of the generated samples.
    plt.figure(figsize = (10,5))
    plt.hist(uniform_vals, bins = int(n/2000), density = True, color = 'darkblue')
    plt.title("Histogram of Uniform Random Variables U(0,1)")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("ex_2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()