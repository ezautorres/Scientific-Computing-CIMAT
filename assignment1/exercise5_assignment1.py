# --------------------------------------------------------------------------------------------------------------
# Exercise 5 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise5_assignment1.py
@Main function: LU_cholesky().
@Date         : 04 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the Cholesky decomposition algorithm 23.1 from Trefethen (p. 175).

@Description  :
    The Cholesky decomposition algorithm 23.1 from Trefethen (p. 175) is implemented.
    Examples of using the function are included.
"""
import numpy as np # numpy library.

# --------------------------------------------------------------------------------------------------------------
# CHOLESKY DECOMPOSITION.
# --------------------------------------------------------------------------------------------------------------
def LU_cholesky(A: np.ndarray, addzeros: bool = True) -> np.ndarray:
    """
    This function finds the Cholesky decomposition of a symmetric and positive definite matrix A = R^{T}R.

    Parameters
    ----------
    A : (n,n) array_like
        Symmetric and positive definite matrix.
    addzeros : bool, optional
        Boolean indicating whether to add zeros below the diagonal.

    Returns
    -------
    R : (n,n) ndarray
        Upper triangular matrix.
    """
    R = A.copy() # copy matrix A so as not to modify the original.
    n = len(R)   # dimension of matrix A.
    for k in range(n):                         # traverses the columns of the matrix.
        for j in range(k+1,n):                 # traverse the rows of the matrix.
            tmp = R[k,j]/R[k,k] 
            R[j,j:n] = R[j,j:n] - tmp*R[k,j:n] # update row j.
        R[k,k:n] = R[k,k:n]/np.sqrt(R[k,k])    # normalizes row k.
        if addzeros:                           # if you want to add zeros below the diagonal.
            R[k,0:k] = 0
    return R

if __name__ == '__main__':

    print('\n-----------------------------------------')
    print('Example 1:')
    A1 = np.array([[4,-1,0,0],
                   [-1,14,-1,0],
                   [0,-1,1.4,-1],
                   [0,0,-1,3]], dtype = 'float')
    R1 = LU_cholesky(A1)
    print('The matrix A is:\n', A1)
    print('The matrix R is:\n', R1)
    print('¿R^{T}R = A?', np.linalg.norm(R1.T@R1 - A1) < 0.0001)
    
    print('\n-----------------------------------------')
    print('Example 2:')
    A2 = np.array([[4,1,1],
                   [1,2,1],
                   [1,1,3]], dtype = 'float')
    R2 = LU_cholesky(A2)
    print('The matrix A is:\n', A2)
    print('The matrix R is:\n', R2)
    print('¿R^{T}R = A?', np.linalg.norm(R2.T@R2 - A2) < 0.0001)