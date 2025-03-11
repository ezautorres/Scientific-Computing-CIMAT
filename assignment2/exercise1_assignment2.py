# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 2 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment2.py
@Main function: MODIFIED_GRAM_SCHMIDT()
@Date         : 11 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the modified Gram-Schmidt algorithm 8.1 of Trefethen (p. 58) to generate the QR decomposition.

@Description  :
    The modified Gram-Schmidt algorithm is implemented to compute the QR factorization of a matrix A. Examples
    of the algorithm's usage are included.
"""
import numpy as np # numpy library.

# --------------------------------------------------------------------------------------------------------------
# MODIFIED GRAM-SCHMIDT.
# --------------------------------------------------------------------------------------------------------------
def MODIFIED_GRAM_SCHMIDT(A: np.ndarray, overwrite_A: bool = False) -> np.ndarray:
    """
    This function calculates the QR factorization of a matrix A by the modified Gram-Schmidt algorithm.

    Parameters
    ----------
    A : (n,n) array_like
        Coefficient matrix.
    overwrite_A : bool, optional
        Indicates whether to overwrite matrix A (default = False).

    Returns
    -------
    Q : (n,n) ndarray
        Orthogonal matrix.
    R : (n,n) ndarray
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
# EXAMPLES OF USE.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # EXAMPLE 1.
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 7]], dtype = 'float')
    Q, R = MODIFIED_GRAM_SCHMIDT(A)
    print('\n-----------------------------------------')
    print('EXAMPLE 1.')
    print('\nMatrix A:')
    print(A)
    print('\nMatrix Q:')
    print(Q)
    print('\nMatrix R:')
    print(R)
    print('\nProduct QR:')
    print(Q @ R)
    print('\n¿QR = A?', np.linalg.norm(Q @ R - A) < 1e-7)

    # EXAMPLE 2.
    A = np.array([[1, 3, 2, 4],
                  [2, 4, 1, 3],
                  [3, 1, 4, 2]], dtype = 'float')
    Q, R = MODIFIED_GRAM_SCHMIDT(A)
    print('\n-----------------------------------------')
    print('EXAMPLE 2.')
    print('\nMatrix A:')
    print(A)
    print('\nMatrix Q:')
    print(Q)
    print('\nMatrix R:')
    print(R)
    print('\nProduct QR:')
    print(Q @ R)
    print('\n¿QR = A?', np.linalg.norm(Q @ R - A) < 1e-7)