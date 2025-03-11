# --------------------------------------------------------------------------------------------------------------
# Exercise 3 | Assignment 7 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise3_assignment7.py
@Main function: METROPOLIS_HASTINGS().
@Date         : 23 October 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement Random Walk Metropolis Hasting (RWMH) where the target distribution is $N_2 (mu,Sigma)$, with
        mu = (3,5), Sigma = ((1,0.9),(0.9,1)).
    Use as a proposal $e_t ~ N_2 (0,sg*I)$. How to choose $sigma$ to make the chain efficient? What are the
    consequences of choosing $\sigma$? As an experiment, choose as initial point
        x_0 = (1000,1)
    and comment on the results.

@Description  :
    The Random Walk Metropolis-Hastings algorithm is implemented in R^n. The objective function
        f(x) ~ N(mu, Sigma) = N([3, 5], [[1, 0.9], [0.9, 1]])
    is considered. Four Markov chains are simulated in R^2 with different initial points and covariance
    matrices for the proposal. The density contours and the evolution of the Markov chain in R^2, the
    marginal histograms and the evolution of the marginal Markov chains are plotted.
"""
import numpy as np              # numpy library.
from typing import Callable     # type hints.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.

# --------------------------------------------------------------------------------------------------------------
# Auxiliar functions for graphics.
# --------------------------------------------------------------------------------------------------------------

# Contour plot of density.
def contour_plot(chain: np.ndarray, mean: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, ax = None) -> plt.Axes:
    """
    This function plots the density contours and the evolution of the Markov chain in R^2.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,8)) # Figure and axis.
    x_max = np.max(chain[:,0])+1
    x_min = np.min(chain[:,0])-1
    y_max = np.max(chain[:,1])+1
    y_min = np.min(chain[:,1])-1
    x, y = np.mgrid[x_min:x_max:0.1, y_min:y_max:0.1]
    pos = np.dstack((x,y))
    rv = stats.multivariate_normal(mu, Sigma)
    contour = ax.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis') # Density Contours.
    # Chain Points
    ax.scatter(chain[0,0], chain[0,1], color='red', label=f'Initial point: [{chain[0,0]},{chain[0,1]}].', s = 100, zorder=20)
    ax.plot(chain[:,0], chain[:,1], alpha=0.3, color='yellow', marker='o', linestyle='-', markersize = 1, label='Evolution of the Markov Chain.', zorder=20)
    ax.scatter(mean[0], mean[1], color='blue', label=f'Chain Mean: [{mean[0]},{mean[1]}].', s = 70, zorder=20)
    ax.legend()
    ax.set_title('Density Contours and Evolution of the Markov Chain.')
    ax.set_xlabel('$x_1$.')
    ax.set_ylabel('$x_2$.')
    ax.grid(True)
    
    return ax

# Marginal histograms.
def marginal_histograms(chain: np.ndarray, n_bins: int, ax = None) -> plt.Axes:
    """
    This function plots the marginal histograms of the Markov chain in R^2.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,8))
    # Histograms.
    ax.hist(chain[:,0], bins=n_bins, density = True, alpha = 0.7, label = 'Histogram of $x_1$.')
    ax.hist(chain[:,1], bins=n_bins, density = True, alpha = 0.7, label = 'Histogram of $x_2$.')
    # Density functions.
    x_1 = np.linspace(np.min(chain[:,0])-1, np.max(chain[:,0])+1, 1000)
    x_2 = np.linspace(np.min(chain[:,1])-1, np.max(chain[:,1])+1, 1000)
    ax.plot(x_1, stats.norm.pdf(x_1, loc = 3, scale = 1), color='blue', label='Density of $x_1$.')
    ax.plot(x_2, stats.norm.pdf(x_2, loc = 5, scale = 1), color='red', label ='Density of $x_2$.')
    ax.legend()
    ax.set_title('Marginal Histograms of the Markov Chain.')
    ax.set_xlabel('$x$.')
    ax.set_ylabel('Frequency.')
    
    return ax

# Evolution of the Marginals Markov chains.
def plot_evolution_burn_in(chain: np.ndarray, burn_in: int, ax = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize = (10,8))
    ax[0].plot(chain[:,0], '-', label='Evolution of the chain $x_1$.', color='blue')
    ax[0].axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in} iterations).')
    ax[0].scatter(0, chain[0,0], color='green', label=rf"Initial point $x_1(0) = {chain[0,0]:.2f}$.", s=100)
    ax[0].set_ylabel(r"$X_t$")
    ax[0].set_title('Evolution of $x_1$ in the chain.')
    ax[0].legend()

    ax[1].plot(chain[:,1], '-', label='Evolution of the chain $x_2$.', color='blue')
    ax[1].axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in} iteraciones).')
    ax[1].scatter(0, chain[0,1], color='green', label=rf"Initial point $x_2(0) = {chain[0,1]:.2f}$.", s=100)
    ax[1].set_title('Evolution of $x_2$ in the chain.')
    ax[1].set_ylabel(r"$X_t$")
    ax[1].legend()
    
    return ax

# Function to calculate the moving average.
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Function to estimate the burn-in.
def estimate_burn_in(chain, window = 100):
    mean_x1 = moving_average(chain[:,0], window)
    mean_x2 = moving_average(chain[:,1], window)

    # Estimate burn-in as the first point where the moving average stabilizes.
    burn_in_x1 = np.argmax(np.abs(np.diff(mean_x1)) < 1e-3) + window
    burn_in_x2 = np.argmax(np.abs(np.diff(mean_x2)) < 1e-3) + window
    
    # Take the maximum of the two burn-ins.
    return max(burn_in_x1, burn_in_x2)

# --------------------------------------------------------------------------------------------------------------
# Random Walk Metropolis-Hastings en R^n.
# --------------------------------------------------------------------------------------------------------------
def RANDOM_WALK_METROPOLIS_HASTINGS(f: Callable, x0: np.ndarray, cov: np.ndarray, N: int) -> np.ndarray:
    """
    Random Walk Metropolis-Hastings Algorithm in R^n.

    Parameters
    ----------
    f : Callable
        Objective function.
    x0 : array_like
        Initial point.
    cov : array_like
        Covariance matrix for the proposal.
    N : int
        Number of iterations.

    Returns
    -------
    chain : np.ndarray
        Simulated Markov chain in R^n. Each row is an iteration and each column is a dimension.
    """
    n = len(x0)             # Space dimension.
    chain = np.zeros((N,n)) # Markov chain in R^n.
    chain[0] = x0           # Initial point.
    accept = 0              # Number of accepted proposals.
    media0 = np.zeros(n)    # Mean of the proposal.
    for i in range(1,N):
        xt = chain[i-1]     # Current point.
        yt = xt + stats.multivariate_normal.rvs(mean=media0, cov=cov) # Proposal.
        try:
            div = f(yt)/f(xt) # Ratio of the objective functions. 
        except ZeroDivisionError:
            div = 0
        rho = min(1, div) # Acceptance ratio.
        if np.random.rand() < rho:
            chain[i] = yt # Aceptance.
            accept += 1
        else:
            chain[i] = xt # Rejection.

    print(f"Acceptance rate       : {accept / N * 100:.2f}%")
    return chain

# --------------------------------------------------------------------------------------------------------------
# Objetive Function: f(x) ~ N(mu, Sigma) = N([3, 5], [[1, 0.9], [0.9, 1]]).
# --------------------------------------------------------------------------------------------------------------
def f_pdf(x: np.ndarray, mu: np.ndarray = [3,5], Sigma: np.ndarray = [[1,0.9],[0.9,1]]) -> float:
    """
    Objective function: multivariate normal distribution.
    """
    return stats.multivariate_normal.pdf(x, mean = mu, cov = Sigma)

# --------------------------------------------------------------------------------------------------------------
# Implementation of the algorithm.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    np.random.seed(315)                  # Seed for reproducibility.
    mu     = np.array([3,5])             # Mean of the distribution.
    Sigma  = np.array([[1,0.9],[0.9,1]]) # Covariance matrix of the distribution.
    n_bins = 50                          # Number of bins for the histograms.
    N      = 10000                       # Number of iterations.

# --------------------------------------------------------------------------------------------------------------
# EXAMPLE 1.
    print("\nExample 1:")
    x0       = np.array([1,1]) # Initial point.
    sg       = 1               # Covariance matrix for the proposal.
    cov_prop = sg*np.eye(2)    # Covariance of the proposal.
    chain    = RANDOM_WALK_METROPOLIS_HASTINGS(f = lambda x: f_pdf(x, mu, Sigma),
                                               x0 = x0, cov = cov_prop, N = N)
    burn_in_estimated = estimate_burn_in(chain) # Estimation of the burn-in.
    print(f"Initial Distribution  : {x0}")
    print(f"Proposal variance     : {sg}")
    print(f"Estimated burn-in     : {burn_in_estimated} iterations")
    print(f"Promedio de la cadena : {np.mean(chain, axis = 0)}")
    print(f"Chain covariance      : \n{np.cov(chain.T)}\n")
    # Graphics.
    ax = contour_plot(chain,np.mean(chain,axis = 0),mu,Sigma) # Density contours.
    plt.savefig("contour_example1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = marginal_histograms(chain, n_bins) # Marginal histograms.
    plt.savefig("histograms_example1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = plot_evolution_burn_in(chain, burn_in_estimated) # Evolution of the marginal Markov chains.
    plt.tight_layout()
    plt.savefig("evolution_example1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------
# EXAMPLE 2.
    print("\nExample 2:")
    x0       = np.array([20,15]) # Initial point.
    sg       = 20                # Covariance matrix for the proposal.
    cov_prop = sg*np.eye(2)      # Covariance of the proposal.
    chain    = RANDOM_WALK_METROPOLIS_HASTINGS(f = lambda x: f_pdf(x, mu, Sigma),
                                               x0 = x0, cov = cov_prop, N = N)
    burn_in_estimated = estimate_burn_in(chain) # Estimation of the burn-in.
    print(f"Initial Distribution  : {x0}")
    print(f"Proposal variance     : {sg}")
    print(f"Estimated burn-in     : {burn_in_estimated} iterations")
    print(f"Promedio de la cadena : {np.mean(chain, axis = 0)}")
    print(f"Chain covariance      : \n{np.cov(chain.T)}\n")
    # Graphics.
    ax = contour_plot(chain,np.mean(chain,axis = 0),mu,Sigma) # Density contours.
    plt.savefig("contour_example2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = marginal_histograms(chain, n_bins) # Marginal histograms.
    plt.savefig("histograms_example2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = plot_evolution_burn_in(chain, burn_in_estimated) # Evolution of the marginal Markov chains.
    plt.tight_layout()
    plt.savefig("evolution_example2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------
# EXAMPLE 3.
    print("\nExample 3:")
    x0       = np.array([-20,-10]) # Initial point.
    sg       = 25                  # Covariance matrix for the proposal.
    cov_prop = sg*np.eye(2)        # Covariance of the proposal.
    chain    = RANDOM_WALK_METROPOLIS_HASTINGS(f = lambda x: f_pdf(x, mu, Sigma),
                                               x0 = x0, cov = cov_prop, N = N)
    burn_in_estimated = estimate_burn_in(chain) # Estimation of the burn-in.
    print(f"Initial Distribution  : {x0}")
    print(f"Proposal variance     : {sg}")
    print(f"Estimated burn-in     : {burn_in_estimated} iterations")
    print(f"Promedio de la cadena : {np.mean(chain, axis = 0)}")
    print(f"Chain covariance      : \n{np.cov(chain.T)}\n")
    # Graphics.
    ax = contour_plot(chain,np.mean(chain,axis = 0),mu,Sigma) # Density contours.
    plt.savefig("contour_example3.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = marginal_histograms(chain, n_bins) # Marginal histograms.
    plt.savefig("histograms_example3.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = plot_evolution_burn_in(chain, burn_in_estimated) # Evolution of the marginal Markov chains.
    plt.tight_layout()
    plt.savefig("evolution_example3.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------
# EXAMPLE 4.
    print("\nExample 4: [1000,1]")
    x0       = np.array([1000,1]) # Initial point.
    print(r"Probability that $x_0$ is on the support:")
    print(stats.multivariate_normal.pdf(x0, mean = mu, cov = Sigma))
    sg       = 100                # Covariance matrix for the proposal.
    cov_prop = sg*np.eye(2)       # Covariance of the proposal.
    chain    = RANDOM_WALK_METROPOLIS_HASTINGS(f = lambda x: f_pdf(x, mu, Sigma),
                                               x0 = x0, cov = cov_prop, N = N)
    burn_in_estimated = estimate_burn_in(chain) # Estimation of the burn-in.
    print(f"Initial Distribution  : {x0}")
    print(f"Proposal variance     : {sg}")
    print(f"Estimated burn-in     : {burn_in_estimated} iterations")
    print(f"Promedio de la cadena : {np.mean(chain, axis = 0)}")
    print(f"Chain covariance      : \n{np.cov(chain.T)}\n")
    # Graphics.
    ax = contour_plot(chain,np.mean(chain,axis = 0),mu,Sigma) # Density contours.
    plt.savefig("contour_example4.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()
    ax = plot_evolution_burn_in(chain, burn_in_estimated) # Evolution of the marginal Markov chains.
    plt.tight_layout()
    plt.savefig("evolution_example4.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()