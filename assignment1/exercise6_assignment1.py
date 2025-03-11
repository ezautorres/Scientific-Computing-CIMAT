# --------------------------------------------------------------------------------------------------------------
# Exercise 6 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise6_assignment1.py
@Main function: 
@Date         : 04 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Compare the complexity of your implementation of the Cholesky and LUP factorization algorithms by measuring the
    times they take with respect to the decomposition of a positive definite Hermitian random matrix. Plot the
    comparison.
"""
from exercise5_assignment1 import LU_cholesky # import the LU_cholesky function.
from exercise3_4_assignment1 import LUP       # import the LUP function.
import numpy as np                            # numpy library.
import matplotlib.pyplot as plt               # matplotlib library.
import time                                   # time library.  

def generate_matrix(n: int) -> np.ndarray:
    """
    This function generates a positive definite Hermitian random matrix of dimension nxn.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    A : nxn-ndarray
        Positive definite Hermitian random matrix.
    """
    A = np.random.rand(n,n)     # a random matrix of inputs U(0,1) is generated.
    A = A + A.T + n * np.eye(n) # we make the matrix symmetric and positive definite.
    return A

cholesky_times = []            # list to store cholesky LU execution times.
LUP_times = []                 # list to store LUP execution times.
n_matrix = range(50, 1501, 50) # matrix sizes.

for n in n_matrix: 
    A = generate_matrix(n) # generates a positive definite Hermitian random matrix of size nxn.

    start_time = time.perf_counter()                        # time measurement for LU cholesky.
    LU_cholesky(A)                                          # Cholesky factorization.
    cholesky_times.append(time.perf_counter() - start_time) # runtime.

    start_time = time.perf_counter()                   # time measurement for LUP.
    LUP(A)                                             # LUP factorization.
    LUP_times.append(time.perf_counter() - start_time) # runtime.

# GRAPHING OF RESULTS.
plt.plot(n_matrix, cholesky_times, label = 'LU Cholesky', marker = 'o', c = 'r')
plt.plot(n_matrix, LUP_times, label = 'LUP', marker = 'o', c = 'b')
plt.xlabel(r'Matrix size ($n$).')
plt.ylabel('Execution time (seconds).')
plt.title('Comparison of Execution Times: Cholesky vs LUP.')
plt.legend()
plt.grid(True)
plt.savefig("ex6_as1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
plt.show()