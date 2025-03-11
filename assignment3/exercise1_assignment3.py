# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 3: Stability | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment3.py
@Main function: 
@Date         : 18 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Described in the assignment's instructions.
"""
import numpy as np # numpy library.
import scipy       # scipy library.
import time        # time library.

# --------------------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS.
# --------------------------------------------------------------------------------------------------------------
def LU_cholesky(A: np.ndarray, addzeros: bool = True) -> np.ndarray:
    """
    This function finds the Cholesky decomposition of A = R^{T}R.

    Parameters
    ----------
    A : (n,n) array_like
        Symmetric and positive definite matrix.
    addzeros : bool, optional
        Indicates whether to add zeros below the diagonal (default is True).

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

def generate_Bs(lb1: float, lb20: float = 1, sg: float = 0.025, n: int = 20) -> np.ndarray:
    """
    This function generates matrices B and B_e with eigenvalues in the interval [lb1, lb20 = 1] and random noise.
    
    Parameters
    ----------
    lb1 : float
        Largest eigenvalue.
    lb20 : float, optional
        Smallest eigenvalue (default is 1).
    sg : float, optional
        Standard deviation of the noise (default is 0.025).
    n : int, optional
        Dimension of the matrices (default is 20).

    Returns
    -------
    B : (n,n) ndarray
        Matrix B.
    B_e : (n,n) ndarray
        Noisy matrix B.
    """
    Q, R    = scipy.linalg.qr(np.random.randn(n,n)) # unitary matrix Q.
    lambdas = np.geomspace(lb1, lb20, n)            # generates the eigenvalues ​​in [lb1, lb20 = 1].
    eps_i   = np.random.normal(0, sg, n)            # random noise.
    
    B   = Q.T @ np.diag(lambdas) @ Q         # matrix B.
    B_e = Q.T @ np.diag(lambdas + eps_i) @ Q # noisy matrix B.
    return B, B_e

# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    lb20 = 1     # smallest eigenvalue.
    sg   = 0.025 # standard deviation of the noise.
    n    = 20    # dimension of the matrices.

    # --------------------------------------------------------------------------------------------------------
    print('\n-----------------------------------------')
    print('     a)')
    print('Well conditioned:\n')
    lb1  = 2 # Largest eigenvalue.
    B, B_e = generate_Bs(lb1, lb20, sg, n)
    R   = LU_cholesky(B)   # Cholesky of B.
    R_e = LU_cholesky(B_e) # Cholesky of B_e.
    print('Condition number:', np.linalg.cond(B), np.linalg.cond(B_e))
    print('¿R^{T}R = B?    :', np.linalg.norm(B - R.T @ R) < 1e-6)
    print('¿Re^{T}Re = B?  :', np.linalg.norm(B_e - R_e.T @ R_e) < 1e-6)

    print('\nBad conditioned:\n')
    lb1  = 1e9 # Largest eigenvalue.
    B, B_e = generate_Bs(lb1, lb20, sg, n)
    R   = LU_cholesky(B)   # Cholesky of B.
    R_e = LU_cholesky(B_e) # Cholesky of B_e.
    print('Condition number:', np.linalg.cond(B), np.linalg.cond(B_e))
    print('¿R^{T}R = B?    :', np.linalg.norm(B - R.T @ R) < 1e-6)
    print('¿Re^{T}Re = B?  :', np.linalg.norm(B_e - R_e.T @ R_e) < 1e-6)

    # --------------------------------------------------------------------------------------------------------
    print('\n-----------------------------------------')
    print('     b)')
    R_chol   = scipy.linalg.cholesky(B)   # Cholesky de B.
    R_chol_e = scipy.linalg.cholesky(B_e) # Cholesky de B_e.
    print('Are they the same with Scipy?')
    print(np.linalg.norm(R_chol - R) < 1e-6)
    print(np.linalg.norm(R_chol_e - R_e) < 1e-6)

    # --------------------------------------------------------------------------------------------------------
    print('\n-----------------------------------------')
    print('     c)')

    # Lists to store execution times.
    times_B = []
    times_B_e = []
    times_B_chol = []
    times_B_e_chol = []

    for _ in range(1000): # 1000 iterations.

        B, B_e = generate_Bs(lb1, lb20, sg, n)

        star_time = time.perf_counter()
        LU_cholesky(B)
        times_B.append(time.perf_counter() - star_time)

        star_time = time.perf_counter()
        LU_cholesky(B_e)
        times_B_e.append(time.perf_counter() - star_time)

        star_time = time.perf_counter()
        scipy.linalg.cholesky(B)
        times_B_chol.append(time.perf_counter() - star_time)

        star_time = time.perf_counter()
        scipy.linalg.cholesky(B_e)
        times_B_e_chol.append(time.perf_counter() - star_time)

    print('Average time LU_cholesky(B)     :\n', np.mean(times_B))
    print('Average time LU_cholesky(B_e)   :\n', np.mean(times_B_e))
    print('Average time scipy.cholesky(B)  :\n', np.mean(times_B_chol))
    print('Average time scipy.cholesky(B_e):\n', np.mean(times_B_e_chol))