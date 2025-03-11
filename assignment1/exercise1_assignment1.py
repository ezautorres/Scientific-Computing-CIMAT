# --------------------------------------------------------------------------------------------------------------
# Exercise 1 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise1_assignment1.py
@Main function: FORWARD_SUBST() and BACKWARD_SUBST()
@Date         : 04 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the Backward and Forward substitution algorithms.

@Description  :
    The FORWARD_SUBST() and BACKWARD_SUBST() functions are implemented to solve systems of linear equations for lower
    and upper triangular matrices respectively. Examples of using the functions are included.
"""
import numpy as np # numpy library.

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
    for i in range(n-1,-1,-1):       # cycle that traverses the rows of the matrix starting from the second to last row.
        s = 0                        # initializes the sum to zero.
        for j in range(i+1,n):       # cycle that iterates through the columns of the matrix.
            s = s + U[i,j] * x[j]    # sums the products of the matrix coefficients and the components of x already calculated.
        x[i] = (b[i] - s) / U[i,i]   # calculates the i component of x.
    return x

# --------------------------------------------------------------------------------------------------------------
# FORWARD SUBSTITUTION
# --------------------------------------------------------------------------------------------------------------
def FORWARD_SUBST(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function solves the system Lx = b by forward substitution for lower triangular matrices.
    
    Parameters
    ----------
    L : (n,n) array_like
        Lower triangular matrix (system coefficients).
    b : (n,) array_like
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : (n,) ndarray
        Solution to the system Lx = b.
    """
    n = len(L)                       # dimension of the system matrix.
    x = np.zeros(n, dtype = 'float') # initializes the solution vector as a vector of zeros.
    x[0] = b[0] / L[0,0]             # calculates the first component of x.
    for i in range(1,n):             # cycle that traverses the rows of the matrix starting from the second row.
        s = 0                        # initializes the sum to zero.
        for j in range(0,i):         # cycle that iterates through the columns of the matrix.
            s = s + L[i,j] * x[j]    # sums the products of the matrix coefficients and the components of x already calculated.
        x[i] = (b[i] - s) / L[i,i]   # calculates the i component of x.
    return x

# --------------------------------------------------------------------------------------------------------------
# EXAMPLES OF USING THE FORWARD_SUBST AND BACKWARD_SUBST FUNCTIONS.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # BACKWARD_SUBST.
    print('\n-----------------------------------------')
    print('Backward Substitution Example:')
    U = np.array([[1,4,3,10],
                  [0,1,18,2],
                  [0,0,12,3],
                  [0,0,0,3]], dtype = 'float') # upper triangular matrix.
    b1 = np.array([5,1,3,5],  dtype = 'float') # vector b.
    x1 = BACKWARD_SUBST(U, b1)
    print('U matrix:\n', U)
    print('Vector b:\n', b1)
    print('The solution is:\n', x1)
    print('Exact Solution:\n', np.linalg.solve(U,b1)) # solution with numpy solve function.

    # FORWARD_SUBST.
    print('\n-----------------------------------------')
    print('Forward Substitution Example:')
    L = np.array([[1,0,0,0],
                  [4,1,0,0],
                  [3,18,12,0],
                  [1,2,3,3]],   dtype = 'float') # lower triangular matrix.
    b2 = np.array([10,-15,2,4], dtype = 'float') # vector b.
    x2 = FORWARD_SUBST(L, b2)
    print('L matrix:\n', L)
    print('Vector b:\n', b2)
    print('The solution is:\n', x2)
    print('Exact Solution:\n', np.linalg.solve(L,b2)) # solution with numpy solve function.