# --------------------------------------------------------------------------------------------------------------
# Exercise 5.1 | Assignment 5 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : ARS.py
@Main function: INITIAL_DICT, sample, insert_points, acceptance_rejection.
@Date         : 09 Octubre 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Description  :
    Due to the amount of code, it is proposed to create a dictionary that stores the parameters of the ARS method
    and the auxiliary functions. In this way, the values of the parameters and functions can be accessed more
    easily. In addition, it is proposed to create a function that initializes the parameters of the ARS method and
    saves them in the dictionary. Finally, it is proposed to create a function that generates samples from the
    log-concave distribution using the ARS method and updates the dictionary parameters as samples are generated.

    ARS is a sampling method that uses an upper envelope of the objective function to propose samples and then decide
    whether to accept them or not using the acceptance/rejection criterion.

    The code implements the Adaptive Rejection Sampling (ARS) method to sample from a log-concave distribution. This
    method creates a tangential approximation of the logarithm function of the target density \log(h(xi)), and
    generates samples based on this upper envelope. As samples are generated, the tangents are adjusted to improve
    the approximation of the function.
"""
import numpy as np              # numpy library.
import matplotlib.pyplot as plt # matplotlib library.
from typing import Callable     # typing module.
import scipy.stats as stats     # scipy library.

# --------------------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS.
# --------------------------------------------------------------------------------------------------------------
def intersections_and_areas(hi: np.ndarray, xi: np.ndarray, Dhi: np.ndarray, lb: float, ub: float) -> tuple: 
    """
    This function calculates the intersection points of the tangents and the accumulated area under the upper
    envelope. The accumulated area up to the last point is also calculated.

    Parameters
    ----------
    hi : (n,) array_like
        Values of h(xi).
    xi : (n,) array_like
        Ordered vector of points where h(x) is defined.
    Dhi : (n,) array_like
        Values of Dh(xi).
    lb : float
        Lower limit of the domain.
    ub : float
        Upper limit of the domain.

    Returns
    -------
    z : (n+1,) ndarray
        Intersection points of the tangents.
    u : (n,) ndarray
        Upper envelope, values of u(xi) = h(xi) + Dh(xi) * (z - xi).
    s : (n+1,) ndarray
        Accumulated area under the upper envelope, s(zi) = exp(u(zi)) / Dh(zi).
    cui : float
        Accumulated area up to the last point, cui = s(zi[-1]).
    """
    N = len(xi) # Number of points.

    # Calculation of the intersection points of the tangents.
    zi       = np.zeros(N+1)                                      # Intersection points.
    zi[1:-1] = - (np.diff(hi) - np.diff(xi * Dhi)) / np.diff(Dhi) # Expression (7) of the PDF.
    zi[0]    = lb                                                 # Lower limit.
    zi[-1]   = ub                                                 # Upper limit.

    # Calculation of the upper envelope and the accumulated area under the upper envelope.
    iss = np.append([0], np.arange(N))                         # Indexes for the points.
    ui  = hi[iss] + Dhi[iss] * (zi - xi[iss])                  # Upper envelope. Expression (8) of the PDF.
    si  = np.append([0], np.cumsum(np.diff(np.exp(ui)) / Dhi)) # Accumulated area under the upper envelope. Expression (10) of the PDF.
    cui = si[-1]                                               # Accumulated area up to the last point. Expression (11) of the PDF.

    return zi, ui, si, cui

# --------------------------------------------------------------------------------------------------------------
# ADAPTIVE REJECTION SAMPLING (ARS).
# --------------------------------------------------------------------------------------------------------------
def INITIAL_DICT(h: Callable, Dh: Callable, xi: np.ndarray, lb: float, ub: float, n_max: int = 100, **fargs) -> dict:
    """
    Creates a dictionary with initial parameters for the Adaptive Rejection Sampling (ARS) sampling method. Initializes
    the points xi, the values h(xi), their slopes Dh(xi), and calculates the intersection points zi of the tangents.
    Additionally, it calculates the upper envelope u(x) = h(xi) + Dh(xi) * (x - xi) and the accumulated area s under
    the tangents. The accumulated area up to the last point is stored in cui.

    Parameters
    ----------
    h : Callable
        Function that calculates log(f(x)) where f(x) is proportional to the density to be sampled.
    Dh : Callable
        Derivative of h(x): d/dx log(f(x)) to calculate the slope of the tangents (h(x)=f'(x)/f(x)).
    xi : (n,) array_like
        Ordered vector of initial points where h(x) is defined to initialize the tangents.
    lb : float
        Lower limit of the domain.
    ub : float
        Upper limit of the domain.
    n_max : int, optional
        Maximum number of points defining the tangents. (default is 100).
    fargs : dict
        Arguments for h and Dh.

    Returns
    -------
    dict : dict
        Dictionary with the initial parameters.
    
    Notes
    -----
    Inside the dictionary, the previous arguments are stored and also:
        - dict['hi'] : values of h(xi).
        - dict['Dhi']: values of Dh(xi).
        - dict['zi'] : intersection points of the tangents.
        - dict['ui'] : value of the upper envelope at the intersections zi: u(xi) = h(xi) + Dh(xi) * (zi - xi).
        - dict['si'] : accumulated area under the upper envelope, s(zi) = exp(u(zi)) / Dh(zi).
        - dict['cui']: accumulated area up to the last point, cu = s(zi[-1]).
    which are calculated in the function intersections_and_areas().
    """
    hi  = h(xi,**fargs)  # values of h(xi).
    Dhi = Dh(xi,**fargs) # values of Dh(xi).
    dict = {
        'h'    : h,     # function that calculates log(f(x)).
        'Dh'   : Dh,    # derivative of h(x).
        'xi'   : xi,    # ordered vector of initial points where h(x) is defined.
        'fargs': fargs, # arguments for h and Dh.
        'lb'   : lb,    # lower limit of the domain.
        'ub'   : ub,    # upper limit of the domain.
        'n_max': n_max, # maximum number of points defining the envelope.
        'hi'   : hi,    # values of h(xi).
        'Dhi'  : Dhi    # values of Dh(xi).
    }
    # The intersection points of the tangents are calculated and stored in the dictionary.
    dict['zi'], dict['ui'], dict['si'], dict['cui'] = intersections_and_areas(hi, xi, Dhi, lb, ub)
    
    return dict

def sample(dict: dict) -> tuple:
    """
    This function returns a single value randomly sampled from the upper envelope of the log-concave function being
    sampled and the index of the segment in which the sampled value falls.

    Parameters
    ----------
    dict : dict
        Dictionary with the parameters of the ARS method.

    Returns
    -------
    xt : float
        Randomly sampled value from the upper envelope.
    i : int
        Index of the segment in which the sampled value falls.
    """
    u = np.random.rand()                                # random number in [0,1].
    i = np.nonzero(dict['si'] / dict['cui'] < u)[0][-1] # index of the segment where the sample falls. Expression (12) of the PDF.
    xt = dict['xi'][i] + (-dict['hi'][i] + np.log(dict['Dhi'][i] * (dict['cui'] * u - dict['si'][i]) + np.exp(dict['ui'][i]))) / dict['Dhi'][i] # Expression (13) of the PDF.
    return xt, i

def insert_points(dict: dict, xnew: np.ndarray = [], hnew: np.ndarray = [], Dhnew: np.ndarray = []):
    """
    This function updates the envelopes with new points. If no points are provided, it only recalculates the envelope
    from the existing points. If new points are provided, they are concatenated with the existing ones and the
    envelope is recalculated. In addition, the intersection points of the tangents and the accumulated area under
    the envelope are recalculated.

    Parameters
    ----------
    dict : dict
        Dictionary with the parameters of the ARS method.
    xnew : (m,) array_like, optional
        New points to insert in the envelope. (default is []).
    hnew : (m,) array_like, optional
        Values of h(x) in the new points. (default is []).
    Dhnew : (m,) array_like, optional
        Values of Dh(x) in the new points. (default is []).
    """
    if len(xnew) > 0:
        xi = np.hstack([dict['xi'], xnew])                 # existing points are concatenated with new ones.
        idx = np.argsort(xi)                               # indexes of the ordered points.
        dict['xi'] = xi[idx]                               # points are updated.
        dict['hi'] = np.hstack([dict['hi'], hnew])[idx]    # values of h(x) are updated.
        dict['Dhi'] = np.hstack([dict['Dhi'], Dhnew])[idx] # values of Dh(x) are updated.

    # The intersection points of the tangents and the accumulated area under the upper envelope are calculated.
    dict['zi'], dict['ui'], dict['si'], dict['cui'] = intersections_and_areas(dict['hi'], dict['xi'], dict['Dhi'], dict['lb'], dict['ub'])

def acceptance_rejection(dict: dict, N: int) -> np.ndarray:
    """
    This function generates N samples from the log-concave distribution using the Adaptive Rejection Sampling (ARS)

    Parameters
    ----------
    dict : dict
        Dictionary with the parameters of the ARS method.
    N : int
        Number of samples to generate.

    Returns
    -------
    samples : (N,) ndarray
        Samples generated from the log-concave distribution.
    """
    samples = np.zeros(N) # array to store the samples.
    n = 0
    while n < N:
        xt, i = sample(dict)                                       # a random value is sampled from the upper envelope.
        ht = dict['h'](xt, **dict['fargs'])                        # the value of h(xt) is calculated.
        hprimet = dict['Dh'](xt, **dict['fargs'])                  # the value of Dh(xt) is calculated.
        ut = dict['hi'][i] + (xt - dict['xi'][i]) * dict['Dhi'][i] # the value of the upper envelope at the sampled point is calculated.

        u = np.random.rand()  # random number in [0,1].
        if u < np.exp(ht-ut): # acceptance/rejection criterion. Expression (15) of the PDF.
            samples[n] = xt   # the sample is stored.
            n += 1

        # New points are inserted if the number of points is less than the maximum allowed.
        if len(dict['ui']) < dict['n_max']:
            insert_points(dict, [xt], [ht], [hprimet])
    
    return samples   

# --------------------------------------------------------------------------------------------------------------
# USAGE EXAMPLES.
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
# --------------------------------------------------------------------------------------------------------------
# Example 1: Normal Distribution N(2,3).
    # Probability density function of the LogNormal distribution and its derivative.
    def h(x: float, mu: float, sg: float) -> float:
        return -1/(2*sg**2)*(x-mu)**2
    def Dh(x: float, mu: float, sg: float) -> float:
        return -1/sg**2*(x-mu)

    N       = 10000                                                      # Number of samples.
    xi      = np.array([-4,-3,-2, 1, 4])                                 # Initial points.
    dict    = INITIAL_DICT(h=h, Dh=Dh, xi=xi, lb=-50, ub=50, mu=2, sg=3) # Initial dictionary.
    samples = acceptance_rejection(dict, N)                              # Samples generated.

    # Graph of the generated samples and the theoretical normal distribution N(2,3).
    plt.figure(figsize = (10,6))
    plt.hist(samples, bins = int(N/100), density = True, alpha = 1, color = 'b', label = 'Samples generated.')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    real_dist = stats.norm.pdf(x, 2, 3) # PDF of the normal distribution N(2,3).
    plt.plot(x, real_dist, 'k', label = r'PDF $N(2,3)$')
    plt.legend() 
    plt.title(r"Normal Distribution $N(2,3)$ - ARS vs PDF.")
    plt.savefig("ARS_example1.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

# --------------------------------------------------------------------------------------------------------------
# Example 2: Beta Distribution N(1.3, 2.7).
    # Probability density function of the Log Beta distribution and its derivative.
    def h(x: float, a: float, b: float) -> float:
        return (a-1) * np.log(x) + (b-1) * np.log(1-x)
    def Dh(x: float, a: float, b: float) -> float:
        return (a-1)/x - (b-1)/(1-x)
    N       = 10000                                                     # Number of samples.
    xi      = np.array([0.1, 0.6])                                      # Initial points.
    dict    = INITIAL_DICT(h=h, Dh=Dh, xi=xi, lb=0, ub=1, a=1.3, b=2.7) # Initial dictionary.
    samples = acceptance_rejection(dict, N)                             # Samples generated.

    # Graph of the generated samples and the theoretical beta distribution Beta(1.3,2.7).
    plt.figure(figsize = (10,6))
    plt.hist(samples, bins = int(N/100), density = True, alpha = 1, color = 'b', label = 'Samples generated.')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    real_dist = stats.beta.pdf(x,1.3,2.7) # PDF de la distribución beta(1.3,2.7)
    plt.plot(x, real_dist, 'k', label = r'PDF $Beta(1.3,2.7)$.')
    plt.legend()
    plt.title(r"$Beta(1.3,2.7)$ Distribution - ARS vs PDF.")
    plt.savefig("ARS_example2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()