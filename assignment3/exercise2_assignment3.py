# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Assignment 3: Stability | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_assignment3.py
@Main function: 
@Date         : 18 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Described in the assignment's instructions.
"""
import numpy as np           # numpy library.
from scipy.linalg import inv # inv function from scipy.linalg.
np.random.seed(9762)         # seed for reproducibility.

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
# Exercise 2.
# --------------------------------------------------------------------------------------------------------------
def fit_QR(X: np.ndarray, beta: np.ndarray, sg: float = 0.12) -> np.ndarray:
    """
    This function adjusts a linear regression model with noise to the data.

    Parameters
    ----------
    X : (n,d) array_like
        Design matrix.
    beta : (d,) array_like
        Model coefficients.
    sg : float, optional
        Standard deviation of the noise (default = 0.12).
    
    Returns
    -------
    beta_hat : (d,) ndarray
        Adjusted coefficients.
    """
    n, d = X.shape
    eps_i = np.random.normal(0, sg, n)    # random noise.
    y = X @ beta + eps_i                  # values ​​of y with noise.
    Q, R = MODIFIED_GRAM_SCHMIDT(X)       # QR factorization of the design matrix.
    beta_hat = BACKWARD_SUBST(R, Q.T @ y) # adjusted coefficients.
    return beta_hat

# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    d, n = 5, 20                      # Design matrix dimensions and number of data.
    sg1  = 0.12                       # Standard deviation of the noise.
    beta = np.array([5.,4.,3.,2.,1.]) # Coefficients of the model.
    X    = np.random.rand(n, d)       # Design matrix.
    
    # --------------------------------------------------------------------------------------------------------
    print('\n-----------------------------------------')
    print('     a)')
    
    # lists to store the rules of the difference between the coefficients.
    dif_beta_hat   = []
    dif_beta_hat_p = []
    dif_beta_hat_c = []
    for _ in range(1000):

        # betha_hat
        beta_hat = fit_QR(X, beta, sg1) # Data adjustment.
        y = X @ beta                    # Observed values.

        # beta_hat_p
        sg2        = 0.01                          # Standard deviation of noise.
        DeltaX     = np.random.normal(0,sg2,(n,d)) # Disturbance of the design matrix.
        X_tilde    = X + DeltaX                    # Disturbed design matrix.
        beta_hat_p = fit_QR(X_tilde, beta, sg1)    # Adjustment of perturbed data.

        # beta_hat_c
        beta_hat_c = inv(X_tilde.T@X_tilde)@X_tilde.T@y # Fitting perturbed data with the least squares formula.
        
        # Difference between the coefficients.
        dif_beta_hat.append(np.linalg.norm(beta_hat - beta))  
        dif_beta_hat_p.append(np.linalg.norm(beta_hat_p - beta))
        dif_beta_hat_c.append(np.linalg.norm(beta_hat_c - beta))

    print('Coefficients of the original model:\n', beta)
    print('QR adjusted coefficients:\n', beta_hat)
    print('QR and perturbation adjusted coefficients:\n', beta_hat_p)
    print('Least squares adjusted coefficients:\n', beta_hat_c)
    print('\nNorms of the difference between the coefficients:')
    print('beta vs beta_hat  :', np.mean(dif_beta_hat))
    print('beta vs beta_hat_p:', np.mean(dif_beta_hat_p))
    print('beta vs beta_hat_c:', np.mean(dif_beta_hat_c))

    # --------------------------------------------------------------------------------------------------------
    print('\n-----------------------------------------')
    print('     b)')

    # We create ill-conditioned X (matrix with almost collinear columns).
    col_1 = np.random.randn(n,1) # Columna 1.
    # We make the other columns 1 with small perturbations.
    X_bad = np.hstack([col_1 + 1e-5 * np.random.randn(n,1) for _ in range(d)])

    # lists to store the rules of the difference between the coefficients.
    dif_beta_hat   = []
    dif_beta_hat_p = []
    dif_beta_hat_c = []
    for _ in range(1000):

        # betha_hat
        beta_hat = fit_QR(X_bad, beta, sg1) # Data adjustment.
        y = X_bad @ beta                    # Observed values.

        # beta_hat_p
        sg2        = 0.01                          # Standard deviation of noise.
        DeltaX     = np.random.normal(0,sg2,(n,d)) # Disturbance of the design matrix.
        X_tilde    = X_bad + DeltaX                # Disturbed design matrix.
        beta_hat_p = fit_QR(X_tilde, beta, sg1)    # Adjustment of perturbed data.

        # beta_hat_c
        beta_hat_c = inv(X_tilde.T@X_tilde)@X_tilde.T@y # Fitting perturbed data with the least squares formula.
        
        # Difference between the coefficients.
        dif_beta_hat.append(np.linalg.norm(beta_hat - beta))  
        dif_beta_hat_p.append(np.linalg.norm(beta_hat_p - beta))
        dif_beta_hat_c.append(np.linalg.norm(beta_hat_c - beta))

    print('Coefficients of the original model:\n', beta)
    print('QR adjusted coefficients:\n', beta_hat)
    print('QR and perturbation adjusted coefficients:\n', beta_hat_p)
    print('Least squares adjusted coefficients:\n', beta_hat_c)
    print('\nX condition number:\n', np.linalg.cond(X_bad))
    print('\nNorms of the difference between the coefficients:')
    print('beta vs beta_hat  :', np.mean(dif_beta_hat))
    print('beta vs beta_hat_p:', np.mean(dif_beta_hat_p))
    print('beta vs beta_hat_c:', np.mean(dif_beta_hat_c))