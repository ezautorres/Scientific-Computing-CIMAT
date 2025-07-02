# --------------------------------------------------------------------------------------------------------------
# Exercise 3 | Assignment 2 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise3_assignment2.py
@Main function: FIT_QR().
@Date         : 11 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Described in the assignment's instructions.

@Description  :
    The algorithm that performs a least squares polynomial fit of a sinusoidal function with noise is implemented.
    The results are plotted for different values of n and p and the execution times are compared.
"""
import numpy as np                                      # import the numpy module.
import matplotlib.pyplot as plt                         # import the matplotlib module.
import time                                             # import the time module.
from scipy import linalg                                # import the scipy linear algebra module.
from exercise1_assignment2 import MODIFIED_GRAM_SCHMIDT # import the modified Gram-Schmidt algorithm.
from exercise2_assignment2 import BACKWARD_SUBST        # import the backward substitution algorithm.
 
def FIT_QR(n: float, sg: float, p: int, spy_: bool = False) -> np.ndarray:
    """
    This function makes a least squares polynomial fit of a sinus function with noise.

    Parameters
    ----------
    n : int
        Number of data.
    sg : float
        Standard deviation of the noise.
    p : int
        Degree of the polynomial.
    spy_ : bool, optional
        Indicates whether scipy's qr function will be used. The default is False.

    Returns
    -------
    x_i : 1D-ndarray
        x_i values.
    y_i : 1D-ndarray
        y_i values.
    beta : 1D-ndarray
        Coefficients of the polynomial fit.
    y_hat : 1D-ndarray
        Adjusted values.
    """
    i = np.arange(1, n + 1)            # data indexes.
    x_i = 4 * np.pi * i / n            # x_i values.
    eps_i = np.random.normal(0, sg, n) # random noise.
    y_i = np.sin(x_i) + eps_i          # noisy y_i values.
    X = np.vander(x_i, p)              # design matrix.

    if spy_:
        Q, R = linalg.qr(X, mode="economic") # QR factorization of the design matrix with scipy.
    else:
        Q, R = MODIFIED_GRAM_SCHMIDT(X)      # QR factorization of the design matrix.

    beta = BACKWARD_SUBST(R, Q.T @ y_i)      # coefficients of the polynomial fit.
    y_hat = X @ beta                         # prediction of fitted values.

    return x_i, y_i, beta, y_hat

# --------------------------------------------------------------------------------------------------------------
# CASES p = 2, 4, 6, 100 and n = 100, 1000, 10000.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    qr_times_Gram_Schmidt = [] # list to store the running times of QR factorization with Gram-Schmidt.
    qr_times_Scipy = []        # list to store QR factorization run times with Scipy.

    # Parámetros del experimento.
    n_vals = [100, 1000, 10000] # n values.
    p_vals = [2, 4, 6, 100]     # degrees of polynomials.
    sigma  = 0.11               # standard deviation of noise.

    # ----------------------------------------------------------------------------------------------------------
    # QR OF GRAM-SCHMIDT.
    # ----------------------------------------------------------------------------------------------------------

    # A figure with 12 subfigures is created.
    fig, axs = plt.subplots(len(p_vals), len(n_vals), figsize = (18, 12))
    fig.suptitle('Polynomial fitting for different values of $p$ and $n$ with QR Gram-Schmidt')

    # It is graphed for each combination of n and p.
    for i, p in enumerate(p_vals):
        for j, n in enumerate(n_vals):
            
            # Perform the fitting.
            s_time = time.perf_counter()                               # start of runtime.
            x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)                # polynomial fit.
            qr_times_Gram_Schmidt.append(time.perf_counter() - s_time) # end of runtime.

            # Graph the results.
            axs[i,j].scatter(x_i, y_i, c = 'r', alpha = 0.6, s = 10, label = 'Noisy data')
            axs[i,j].plot(x_i, y_hat, c = 'b', label = f'Polynomial fit of degree {p-1}')
            axs[i,j].set_title(f'$n = {n}$, $p = {p}$')
            axs[i,j].grid(True)

            # Axis labels.
            axs[3,j].set_xlabel('$x_i$')
            axs[i,0].set_ylabel('$y_i$')

    # Adjust the layout and display the graph.
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.savefig("gram_schmidt.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    plt.show()

    # ----------------------------------------------------------------------------------------------------------
    # SCIPY QR.
    # ----------------------------------------------------------------------------------------------------------

    # A figure with 12 subfigures is created
    fig, axs = plt.subplots(len(p_vals), len(n_vals), figsize = (18, 12))
    fig.suptitle('Polynomial fitting for different values of $p$ and $n$ with Scipy QR')

    # It is graphed for each combination of n and p.
    for i, p in enumerate(p_vals):
        for j, n in enumerate(n_vals):
            
            # Perform the fitting.
            s_time = time.perf_counter()                        # start of runtime.
            x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)         # polynomial fit.
            qr_times_Scipy.append(time.perf_counter() - s_time) # end of runtime.

            # Graph the results.
            axs[i,j].scatter(x_i, y_i, c = 'r', alpha = 0.6, s = 10, label = 'Noisy data')
            axs[i,j].plot(x_i, y_hat, c = 'b', label = f'Polynomial fit of degree {p-1}')
            axs[i,j].set_title(f'$n = {n}$, $p = {p}$')
            axs[i,j].grid(True)

            # Axis labels.
            axs[3,j].set_xlabel('$x_i$')
            axs[i,0].set_ylabel('$y_i$')

    # Adjust the layout and display the graph.
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.savefig("scipy.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # ----------------------------------------------------------------------------------------------------------
    # COMPARISON OF EXECUTION TIMES.
    # ----------------------------------------------------------------------------------------------------------
    print('\nRunning times of QR factorization with Gram-Schmidt:\n')
    print(qr_times_Gram_Schmidt)
    print('\nQR factorization runtimes with Scipy:\n')
    print(qr_times_Scipy)
    print('\nDifference in execution times:\n')
    print(np.array(qr_times_Scipy) - np.array(qr_times_Gram_Schmidt))