# --------------------------------------------------------------------------------------------------------------
# Auxilary Functions | Final Project | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : auxiliary_functions.py
@Main function: 
@Date         : 04 December 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the report of the assignment.

@Description  :
    This file contains auxiliary functions for Markov chain analysis and results visualization.
    The implemented functions are:
        - moving_average: Calculates the moving average of a one-dimensional array.
        - estim_burn_in_and_modes: Estimates the burn-in of a Markov chain and the mode of each parameter.
        - marginal_evolution_burn_in: Plots the evolution of marginal Markov chains with respect to the estimated burn-in.
        - histograms_chain: Plots the histograms of the simulated Markov chains.
        - trajectory2d3d: Plots the trajectory of the Markov chain in parameter space.
"""
import matplotlib.pyplot as plt # matplotlib library.
import numpy as np              # numpy library.
import scipy.stats as stats     # scipy library.

# --------------------------------------------------------------------------------------------------------------
# Auxilary Functions.
# --------------------------------------------------------------------------------------------------------------
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    This function calculates the moving average of a one-dimensional array.
    
    Parameters
    ----------
    x : array_like
        One-dimensional array.
    w : int
        Window size.
    
    Returns
    -------
    np.ndarray
        Moving average of the input array.
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def estim_burn_in_and_modes(chain: np.ndarray, window: int = 100) -> int:
    """
    This function estimates the burn-in of a Markov chain from the evolution of the moving average. In addition, it
    estimates the mode of the chain for each parameter.
    
    Parameters
    ----------
    chain : array_like
        Simulated Markov chain.
    window : int
        Window size for the moving average (default = 100).
    
    Returns
    -------
    burn_in : int
        Estimated burn-in of the chain.
    modes : np.ndarray
        Modes of the chain parameters.
    """
    n_params = chain.shape[1]     # Number of parameters.
    burn_in  = np.zeros(n_params) # Array to store the burn-in of each parameter.
    modes    = np.zeros(n_params) # Array to store the mode of each parameter.
    
    for i in range(n_params):
        # Estimate burn-in as the first point where the moving average stabilizes
        burn_in[i] = np.argmax(np.abs(np.diff(moving_average(chain[:,i], window))) < 1e-3) + window

        # Estimate the mode of the chain in dimensions N and p.
        hist, bin_edges = np.histogram(chain[int(burn_in[i]):,i], bins = 200, density = True)
        modes[i] = bin_edges[np.argmax(hist) + 1]

    return int(max(burn_in)), modes

# --------------------------------------------------------------------------------------------------------------
# Functions for results visualization.
# --------------------------------------------------------------------------------------------------------------
def marginal_evolution_burn_in(chain: np.ndarray, burn_in: int, params: list, ax = None) -> plt.Axes:
    """
    This function plots the evolution of the marginal Markov chains with respect to the estimated burn-in.

    Parameters
    ----------
    chain : array_like
        Simulated Markov chain.
    params : list
        Names of the chain parameters.
    ax : plt.Axes, optional
        Figure axes (default = None).
    burn_in : int
        Estimated burn-in of the chain.
    """
    n_params = len(params) # Number of parameters.
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow']

    if ax is None: # Figure and axes creation.
        fig, ax = plt.subplots(n_params, 1, figsize = (10,8))

    for i in range(n_params): # Plotting the evolution of the chain for each parameter.
        ax[i].plot(chain[:,i], '-', label = rf"Evolution of the chain {params[i]}", color = colors[i])
        ax[i].axvline(x = burn_in, color = 'r', linestyle = '--', label = f"Burn-in ({burn_in} iterations)")
        ax[i].scatter(0, chain[0,i], color = 'g', label = rf"Initial point {params[i]}(0)={chain[0,i]:.2f}", s = 100)
        ax[i].set_title(f"Evolution of {params[i]} in the chain.")
        ax[i].set_ylabel(r"$X_t$")
        ax[i].legend()

    return ax

def chain_histograms(chain: np.ndarray, burn_in: int, params: list, modes: list,
                     bins: list, x: np.ndarray = None, ax: plt.Axes = None) -> None:
    """
    This function plots the histograms of the simulated Markov chains. It also plots the histogram of the data and 
    the estimated binomial distribution if the x array is provided (this applies only to assignment 1).

    Parameters
    ----------
    chain : array_like
        Simulated Markov chain.
    burn_in : int
        Estimated burn-in of the chain.
    params : list
        Names of the chain parameters.
    modes : list
        Modes of string parameters (same length as params).
    bins : list
        Number of bins for each histogram (same length as params).
    x : np.ndarray, optional
        Data array (default = None).
    ax : plt.Axes, optional
        Figure axes (default = None).
    """
    n_params = len(params) # Number of parameters.

    # Figure and axes creation.
    if ax is None:
        if x is not None: # If the x array is provided, a figure with n_params + 1 subplots is created.
            fig, ax = plt.subplots(n_params + 1, 1, figsize = (9,9))
        else:             # If the x array is not provided, a figure with n_params subplots is created.
            fig, ax = plt.subplots(n_params, 1, figsize = (10,8))

    # Histograms of the chain parameters.
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow']
    for i in range(n_params):
        ax[i].hist(chain[burn_in:,i], bins = bins[i], density = True, alpha = 1,
                   label = rf"Posterior of {params[i]}", color = colors[i], edgecolor = 'black')
        ax[i].axvline(modes[i], color = 'r', label = rf"Mode of {params[i]}")
        ax[i].set_title(rf"Posterior Distribution of {params[i]}.")
        ax[i].set_xlabel(params[i])
        ax[i].set_ylabel('Density')
        ax[i].legend()

    # Histogram of the data and estimated binomial distribution if the array x is given.
    if x is not None:
        x_vals = np.arange(min(min(x),0), 20)
        binom_vals = stats.binom.pmf(x_vals, n = int(modes[0]), p = modes[1])
        ax[2].hist(x, bins = bins[-1], density = True, alpha = 1, color = 'gray',
                   label = "Data", edgecolor = 'black')
        ax[2].plot(x_vals, binom_vals, 'r', label = 'Estimated Binomial Distribution.')
        ax[2].set_title('Estimated Binomial Distribution and Data.')
        ax[2].set_xlabel('Number of Successes')
        ax[2].set_ylabel('Density')
        ax[2].legend()

    return ax

def trayectory2d3d(chain: np.ndarray, params: list, modes: list, ax = None) -> plt.Axes:
    """
    This function plots the trajectory of the Markov chain in parameter space. If params has length 2, the trajectory
    is plotted in 2D. If params has length 3, the trajectory is plotted in 3D and in the 2D planes of each pair of
    parameters.
    
    Parameters
    ----------
    chain : array_like
        Simulated Markov chain.
    params : list
        Names of the chain parameters.
    modes : list
        Modes of the chain parameters (same length as params).
    ax : plt.Axes, optional
        Figure axes (default = None).
    """
    n_params = len(params) # Number of parameters.

    # Case 2D --------------------------------------------------------------
    if n_params == 2:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = (10,8))
        ax.plot(chain[:,0], chain[:,1], label = "Chain trajectory", marker = 'o', linestyle = '-', alpha = 0.3, color = 'blue')
        ax.scatter(chain[0,0], chain[0,1], color = "k", label = f"Initial point: [{chain[0,0]:.2f}, {chain[0,1]:.2f}]", s = 100, zorder = 20)
        ax.scatter(modes[0], modes[1], color = "r", label = f"Chain mode: [{modes[0]:.2f}, {modes[1]:.2f}]", s = 70, zorder = 20)
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        ax.set_title("Trajectory of the Markov Chain.")

    # Case 3D --------------------------------------------------------------
    if n_params == 3:
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize = (14,9))
        colors = ['blue', 'green', 'orange']
        k = 0  # Counter for the axes.
        for i in range(1,4):
            for j in range(i + 1, 4):
                ax[k].plot(chain[:, i-1], chain[:, j-1], label = "Chain trajectory", marker = 'o', linestyle = '-',
                           alpha = 0.3, color = colors[k])
                ax[k].scatter(chain[0, i-1], chain[0, j-1], color = "k", s = 100, zorder = 20,
                              label = f"Initial point: [{chain[0, i-1]:.2f}, {chain[0, j-1]:.2f}]")
                ax[k].scatter(modes[i-1], modes[j-1], color = "r", s = 70, zorder = 20,
                              label=f"Chain mode: [{modes[i-1]:.2f}, {modes[j-1]:.2f}]")
                ax[k].legend()
                ax[k].grid(True)
                ax[k].set_xlabel(params[i-1])
                ax[k].set_ylabel(params[j-1])
                ax[k].set_title(f"Trajectory in the plane {params[i-1]} vs. {params[j-1]}.")
                k += 1      
    return ax