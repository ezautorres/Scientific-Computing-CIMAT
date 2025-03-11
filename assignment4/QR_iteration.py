# --------------------------------------------------------------------------------------------------------------
# Exercise 2.1 | Assignment 3: Stability | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : QR_iteration.py
@Main function: QR_iteration()
@Date         : 25 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Described in the assignment's instructions.
@Description :
    The QR iteration method is implemented to calculate the eigenvalues of a matrix A. Since the QR factorization
    of the matrix is performed at each iteration, the MODIFIED_GRAM_SCHMIDT function of the past tasks is used to
    perform said task. The QR_iteration function is implemented, which receives as arguments the matrix A, the
    maximum number of iterations, the tolerance for convergence, and a boolean indicating whether the matrix A
    should be overwritten. The function returns an array with the eigenvalues of the matrix A.
    
    Examples of the use of the QR_iteration function are included, comparing the results with those obtained by the
    linalg.eig function of scipy.
"""
import numpy as np       # numpy library.
from scipy import linalg # scipy library.

# --------------------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS.
# --------------------------------------------------------------------------------------------------------------
def MODIFIED_GRAM_SCHMIDT(A: np.ndarray, overwrite_A: bool = False) -> np.ndarray:
    """
    This function calculates the QR factorization of a matrix A by the modified Gram-Schmidt algorithm.
    
    Parameters
    ----------
    A : (n,n) array_like
        Coefficient matrix.
    overwrite_A : bool, optional
        Indicates whether to overwrite matrix A (default is False).

    Returns
    -------
    Q : (n,n) array_like
        Orthogonal matrix.
    R : (n,n) array_like
        Upper triangular matrix.
    """
    m, n = A.shape      # dimensions of matrix A.
    if overwrite_A:
        V = A           # overwrites matrix A.
    else:
        V = np.copy(A)  # copy of matrix A.
    Q = np.zeros((m,n)) # initializes the matrix Q.
    R = np.zeros((n,n)) # initializes the matrix R.
    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i] / R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,i] @ V[:,j]
            V[:,j] = V[:,j] - R[i,j] * Q[:,i]
    return Q, R

# --------------------------------------------------------------------------------------------------------------
# ITERACIÓN QR.
# --------------------------------------------------------------------------------------------------------------
def QR_iteration(A: np.ndarray, iter_: int = 1000, tol: float = 1e-7, overwrite_A: bool = False):
    """
    This function calculates the eigenvalues of a matrix A by the QR iteration method.

    Parameters
    ----------
    A : (n,n) array_like
        Matrix whose eigenvalues are to be calculated.
    iter_ : int, optional
        Maximum number of iterations (default is 1000).
    tol : float, optional
        Tolerance for convergence (default is 1e-7).
    overwrite_A : bool, optional
        Indicates whether to overwrite matrix A (default is False).

    Returns
    -------
    np.array
        Eigenvalues of matrix A.
    """
    if overwrite_A: # If the matrix A should be overwritten.
        A_k = A
    else:           # If the matrix A should not be overwritten.
        A_k = A.copy()
    for _ in range(iter_):
        Q, R = MODIFIED_GRAM_SCHMIDT(A_k)       # QR factorization of matrix A.
        A_k = R @ Q                             # Update matrix A.
        if np.linalg.norm(np.tril(A,-1)) < tol: # Check for convergence.
            break
    return np.diag(A_k)                         # Return the diagonal of matrix A.

# --------------------------------------------------------------------------------------------------------------
# EXAMPLES OF USING THE QR_iteration FUNCTION.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    print("-" * 50)
    print('Example 1: distinct eigenvalues')
    A = np.array([[4, -2, 2],
                  [2, -1, 4],
                  [1, -2, 7]], dtype = float)
    print('Matrix:\n', A)
    print("QR Iteration:", QR_iteration(A))
    print("Scipy       :", linalg.eigvals(A))
    print("-" * 50)

    print('Example 2: equal eigenvalues')
    A = np.array([[1, 0, 0],
                  [2, 2, -1],
                  [0, 1, 0]], dtype = float)
    print('Matrix:\n', A)
    print("QR Iteration:", QR_iteration(A))
    print("Scipy       :", linalg.eigvals(A))
    print("-" * 50)

    print('Example 3: complex eigenvalues')
    A = np.array([[4/5, -3/5, 0],
                  [3/5, 4/5,  0],
                  [1, 2, 2]], dtype = float)
    print('Matrix:\n', A)
    print("QR Iteration:", QR_iteration(A))
    print("Scipy       :", linalg.eigvals(A))
    print("-" * 50)