"""
@File name    : exercise3_4_assignment1.py
@Main function: 
@Date         : 04 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).
"""
import numpy as np                                              # numpy library.
import scipy.stats as stats                                     # scipy.stats library.
from exercise1_assignment1 import FORWARD_SUBST, BACKWARD_SUBST # forward and backward substitution functions.
from exercise2_assignment1 import LUP                           # LUP function.
np.random.seed(967)                                             # seed for reproducibility.

# --------------------------------------------------------------------------------------------------------------
# Exercise 3 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@Instructions:
    Give the LUP decomposition for a random matrix of inputs U(0,1) of size 5x5, and for the matrix
        A = [[1,0,0,0,1],
            [-1,1,0,0,1],
            [-1,-1,1,0,1],
            [-1,-1,-1,1,1],
            [-1,-1,-1,-1,1]].
    Verify that PA = LU.
"""
if __name__ == '__main__':
# --------------------------------------------------------------------------------------------------------------
    print('\n===================================================\nExercise 3\n===================================================')
    U = np.random.rand(5,5)
    L1, U1, P1 = LUP(U)
    print('Random matrix of inputs U(0,1) of size 5x5.')
    print('- The matrix U(0,1) is:\n', U)
    print('- The matrix L is:\n', L1)
    print('- The matrix U is:\n', U1)
    print('- The matrix P is:\n', P1)
    print('\n- ¿PA = LU?', np.linalg.norm(P1@U - L1@U1) < 0.0001) # verifies that PA = LU.
# --------------------------------------------------------------------------------------------------------------
    print('\n---------------------------------------------------')
    A = np.array([[1,0,0,0,1],
                  [-1,1,0,0,1],
                  [-1,-1,1,0,1],
                  [-1,-1,-1,1,1],
                  [-1,-1,-1,-1,1]], dtype = 'float')
    L2, U2, P2 = LUP(A)
    print('Matrix A:')
    print('- The matrix A is:\n', A)
    print('- The matrix L is:\n', L2)
    print('- The matrix U is:\n', U2)
    print('- The matrix P is:\n', P2)
    print('\n- ¿PA = LU?', np.linalg.norm(P2@A - L2@U2) < 0.0001) # verifies that PA = LU.

# --------------------------------------------------------------------------------------------------------------
# Exercise 4 | Assignment 1 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
    """
@Instructions:
    Using the above LUP decomposition, solve the system in the form Dx = b where D are the matrices of problem 3,
    for 5 different random b with inputs U(0,1). Checking whether or not it is possible to solve the system.
    """
# --------------------------------------------------------------------------------------------------------------
    print('\n===================================================\nExercise 4\n===================================================')
    print('System Dx = b with D the matrix U(0,1) of problem 3 and b random.')
    for i in range(5):
        b1 = stats.uniform.rvs(size = 5)
        z1 = FORWARD_SUBST(L1, P1@b1)
        x1 = BACKWARD_SUBST(U1, z1)
        print(x1)
        print(np.linalg.solve(U, b1))

# --------------------------------------------------------------------------------------------------------------
    print('\n---------------------------------------------------')
    print('System Dx = b with D the matrix A of problem 3 and b random.')
    for i in range(5):
        b2 = stats.uniform.rvs(size = 5)
        z2 = FORWARD_SUBST(L2, P2@b2)
        x2 = BACKWARD_SUBST(U2, z2)
        print('- Solution', x2)
        print('¿Ax = b?', np.linalg.norm(A@x2 - b2) < 0.0001)
        print(np.linalg.solve(A, b2))