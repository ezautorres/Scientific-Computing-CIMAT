# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment1.py
@Main function: LUP().
@Date         : 04 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the Gaussian elimination algorithm with partial pivoting LUP, 21.1 of Trefethen (p. 160).

@Description  :
    The Gaussian elimination algorithm with partial pivoting LUP is implemented. 21.1 of Trefethen (p. 160).
    Examples of the use of the LUP function are included.
"""
import numpy as np # numpy library.

# --------------------------------------------------------------------------------------------------------------
# GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING LUP.
# --------------------------------------------------------------------------------------------------------------
def LUP(A: np.ndarray) -> np.ndarray:
    """
    This function finds the factorization PA = LU of A.

    Parameters
    ----------
    A : (n,n) array_like
        System matrix.

    Returns
    -------
    L : (n,n) ndarray
        Lower triangular matrix.
    U : (n,n) ndarray
        Upper triangular matrix.
    P : (n,n) ndarray
        Permutation matrix.
    """
    U = A.copy()  # copy matrix A so as not to modify the original.
    n = len(U)    # dimension of matrix A.
    L = np.eye(n) # initializes the matrix L as the identity matrix.
    P = np.eye(n) # initializes the matrix P as the identity matrix.
    for k in range(n-1):       # cycle that iterates through the columns of the matrix.
        for i in range(k+1,n): # pivot selection to maximize |u_{ik}|.
            if abs(U[i,k]) > abs(U[k,k]):
                tmp1 = U[k,k:n]     # swap the rows of U.
                U[k,k:n] = U[i,k:n]
                U[i,k:n] = tmp1
                tmp2 = L[k,0:k]     # swap the rows of L.
                L[k,0:k] = L[i,0:k]
                L[i,0:k] = tmp2
                tmp3 = P[k]         # swap the rows of P.
                P[k] = P[i]
                P[i] = tmp3
        for j in range(k+1,n):                      # cycle that traverses the rows of the matrix.
            L[j,k] = U[j,k] / U[k,k]                # calculates the coefficients of the matrix L.
            U[j,k:n] = U[j,k:n] - L[j,k] * U[k,k:n] # calculates the coefficients of the matrix U.
    return L, U, P

# --------------------------------------------------------------------------------------------------------------
# EXAMPLES OF USING THE LUP FUNCTION.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    print('\n-----------------------------------------')
    print('Example 1:')
    A1 = np.array([[4,1,3,1],
                  [1,18,2,2],
                  [2,3,12,3],
                  [4,5,6,13]], dtype = 'float')
    L1, U1, P1 = LUP(A1) # find the factorization PA = LU of A.
    print('The matrix A is:\n', A1)
    print('The matrix L is:\n', L1)
    print('The matrix U is:\n', U1)
    print('The matrix P is:\n', P1)
    print('\n¿PA = LU?', np.linalg.norm(P1@A1 - L1@U1) < 0.0001) # verifies that PA = LU.

    print('\n-----------------------------------------')
    print('Example 2:')
    A2 = np.array([[1,1,0,-1,0],
                   [1,0,0,0,-1],
                   [4,4,1,-2,-4],
                   [0,1,0,0,-2],
                   [0,0,2,0,0]], dtype = 'float')
    L2, U2, P2 = LUP(A2) # find the factorization PA = LU of A.
    print('The matrix A is:\n', A2)
    print('The matrix L is:\n', L2)
    print('The matrix U is:\n', U2)
    print('The matrix P is:\n', P2)
    print('\n¿PA = LU?', np.linalg.norm(P2@A2 - L2@U2) < 0.0001) # verifies that PA = LU.

    print('\n-----------------------------------------')
    print('Example 3:')
    A3 = np.array([[3,2,1],
                   [6,6.5,3],
                   [3.9,3,3]], dtype = 'float')
    L3, U3, P3 = LUP(A3) # find the factorization PA = LU of A.
    print('The matrix A is:\n', A3)
    print('The matrix L is:\n', L3)
    print('The matrix U is:\n', U3)
    print('The matrix P is:\n', P3)
    print('\n¿PA = LU?', np.linalg.norm(P3@A3 - L3@U3) < 0.0001) # verifies that PA = LU.