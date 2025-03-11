# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 2 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment2.py
@Main function: LEAST_SQUARES_QR().
@Date         : 11 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Described in the assignment's instructions.

@Description  :
    The algorithm that calculates the least squares estimator in a regression using QR decomposition is implemented.
    The BACKWARD_SUBST() function will be used to solve the triangular system. Examples of the algorithm's use are
    included.
"""
import numpy as np                                      # numpy library.
from exercise1_assignment2 import MODIFIED_GRAM_SCHMIDT # import the modified Gram-Schmidt algorithm.

# --------------------------------------------------------------------------------------------------------------
# BACKWARD SUBSTITUTION
# --------------------------------------------------------------------------------------------------------------
def BACKWARD_SUBST(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function solves the system Ux = b by backward substitution for upper triangular matrices.

    Parameters
    ----------
    U : (n,n) array_like
        Upper triangular matrix (system coefficients).
    b : (n,) array_like
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : (n,) ndarray
        Solution to the system Ux = b.
    """
    n = len(U)                       # dimension of the system matrix.
    x = np.zeros(n, dtype = 'float') # initializes the solution vector as a vector of zeros.
    x[n-1] = b[n-1] / U[n-1,n-1]     # calculates the last component of x.
    for i in range(n-1,-1,-1):       # cycle that goes through the rows of the matrix starting from the second to last row.
        s = 0                        # initializes the sum to zero.
        for j in range(i+1,n):       # cycle that iterates through the columns of the matrix.
            s = s + U[i,j] * x[j]    # sums the products of the matrix coefficients and the components of x already calculated.
        x[i] = (b[i] - s) / U[i,i]   # calculates the i component of x.
    return x

# --------------------------------------------------------------------------------------------------------------
# LEAST SQUARES ESTIMATOR.
# --------------------------------------------------------------------------------------------------------------
def LEAST_SQUARES_QR(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    This function calculates the least squares estimator in a regression.

    Parameters
    ----------
    X : (m,n) array_like
        Coefficient matrix.
    y : (m,) array_like
        Response vector.

    Returns
    -------
    beta : (n,) ndarray
        Coefficients of the least squares estimator.
    """
    Q, R = MODIFIED_GRAM_SCHMIDT(X)    # calculates the QR factorization of the matrix X.
    beta = np.linalg.solve(R, Q.T @ y) # solve the triangular system Rβ = Q^{T}y.
    return beta

# --------------------------------------------------------------------------------------------------------------
# EXAMPLES OF USE.
# --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # EXAMPLE 1.
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3],
                  [1, 5]], dtype = 'float')
    y = np.array([1, 2, 1.3, 3.75], dtype = 'float')
    beta = LEAST_SQUARES_QR(X, y) # least squares estimator.
    print('\n-----------------------------------------')
    print('EXAMPLE 1.')
    print('\nMatrix X:')
    print(X)
    print('\nVector y:')
    print(y)
    print('\nLeast Squares Estimator:')
    print(beta)

    # EXAMPLE 2.
    X = np.array([[10,27,7,4],
                  [27,107,18,10],
                  [7,18,7,1],
                  [4,10,1,4]], dtype = 'float')
    y = np.array([60.5, 213.5, 44, 20.5], dtype = 'float')
    beta = LEAST_SQUARES_QR(X, y) # least squares estimator.
    print('\n-----------------------------------------')
    print('Example 2.')
    print('\nMatrix X:')
    print(X)
    print('\nVector y:')
    print(y)
    print('\nLeast Squares Estimator:')
    print(beta)