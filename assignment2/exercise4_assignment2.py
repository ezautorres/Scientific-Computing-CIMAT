# --------------------------------------------------------------------------------------------------------------
# Exercise 4 | Assignment 2 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise4_assignment2.py
@Main function: 
@Date         : 11 September 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Let p = 0.1n, or ten times as many observations as there are coefficients in the regression. What is the
    maximum n that your computer can handle?

@Description  :
    The algorithm that performs a least squares polynomial fit of a sinusoidal function is implemented.
"""
import numpy as np                       # import the numpy module.
import matplotlib.pyplot as plt          # import the matplotlib module.
from exercise3_assignment2 import FIT_QR # import the FIT_QR function.

def graph(n: int, p: int, x_i: np.ndarray, y_i: np.ndarray, y_hat: np.ndarray, filename: str) -> None:
    """
    This function plots the noisy data and the polynomial fit of degree p.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Degree of the polynomial.
    x_i : (n,) array_like
        x values.
    y_i : (n,) array_like
        y values.
    y_hat : (n,) array_like
        Estimated values of y.
    filename : str
        Name of the file to save the plot.
    """
    plt.scatter(x_i, y_i, c = 'r', alpha = 0.6, s = 10, label = 'Noisy data.')
    plt.plot(x_i, y_hat, c = 'b', label = f'Polynomial fit of degree {p}')
    plt.title(f'$n = {n}$, $p = {p}$')
    plt.xlabel('$x_i$')
    plt.ylabel('$y_i$')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

if __name__ == '__main__':

    sigma = 0.11 # standard deviation of the noise.

    # Example for n = 100 and p = 10 ---------------------------------------------------------------------------
    n = 100
    p = int(0.1*n)
    x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)
    graph(n, p, x_i, y_i, y_hat, 'n_100_p_10.pdf')

    # Example for n = 1000 and p = 100 ---------------------------------------------------------------------------
    n = 1000
    p = int(0.1*n)
    x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)
    graph(n, p, x_i, y_i, y_hat, 'n_1000_p_100.pdf')

    # Example for n = 2810 and p = 281 ---------------------------------------------------------------------------
    n = 2810
    p = int(0.1*n)
    x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)
    graph(n, p, x_i, y_i, y_hat, 'n_2810_p_281.pdf')

    # Example for n = 2820 and p = 282 ---------------------------------------------------------------------------
    n = 2820
    p = int(0.1*n)
    x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)
    graph(n, p, x_i, y_i, y_hat, 'n_2820_p_282.pdf')

    # Example for n = 28200 and p = 2820 -------------------------------------------------------------------------
    n = 28200
    p = int(0.1*n)
    x_i, y_i, beta, y_hat = FIT_QR(n, sigma, p)
    graph(n, p, x_i, y_i, y_hat, 'n_28200_p_2820.pdf')