# --------------------------------------------------------------------------------------------------------------
# Exercise 2.2 | Assignment 4 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment4.py
@Main function: QR_iteration()
@Date         : 25 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the assignment's instructions.

@Description  :
    The QR_iteration.py file implements the QR iteration method to calculate the eigenvalues of a matrix. This
    file includes a script that compares the results obtained by the QR_iteration function with those obtained
    by the Scipy linalg.eig function for the matrix in exercise 1 for different values of epsilon and N. It can
    be observed that the results obtained by both functions are equal.
"""
import numpy as np                    # numpy library.            
from scipy import linalg              # scipy library.
from QR_iteration import QR_iteration # QR iteration method.

# --------------------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS.
# --------------------------------------------------------------------------------------------------------------
def create_matrix(eps: float):
    """
    The create_matrix function creates the matrix of exercise 1.

    Parameters
    ----------
    eps : float
        Value of epsilon.
    
    Returns
    -------
    np.array
        Matrix of exercise 1.
    """
    return np.array([[8., 1.,  0.],
                     [1., 4., eps],
                     [0., eps, 1.]], dtype = float)

# --------------------------------------------------------------------------------------------------------------
# Aplication of the QR iteration method.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   
    # Values of epsilon and N.
    N_vals   = [0., 1., 2., 3., 4., 5.]
    epsilons = [10**(-N) for N in N_vals]
    max_iter = 1000 # Maximum number of iterations.

    for N, eps in zip(N_vals, epsilons): # Loop over values of epsilon and N.
        
        A = create_matrix(eps)                   # Matrix of exercise 1.
        qr_eigenvals = QR_iteration(A, max_iter) # Eigenvalues ​​with the QR method.
        scipy_eigenvals = linalg.eigvals(A)      # Eigenvalues ​​with the Scipy function.

        print(f"N = {N}, eps = {eps}")
        print("QR Iteration:", qr_eigenvals)
        print("Scipy       :", scipy_eigenvals)
        print("-" * 50)