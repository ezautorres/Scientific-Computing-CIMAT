# --------------------------------------------------------------------------------------------------------------
# Exercise 5.2 | Assignment 5 | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise5_assignment5.py
@Main function: 
@Date         : 09 Octubre 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    Implement the Adaptive Rejection Sampling algorithm and simulate 10,000 samples of a Gamma(2,1).
    When is it appropriate to stop adapting the envelope?

@Description  :
    It is desired to generate samples from a Gamma(2,1) distribution using the acceptance-rejection method.
    To do this, the acceptance-rejection method is implemented in the ARS.py file and samples from the Gamma(2,1)
    distribution are generated using this method. The generated samples are compared with the theoretical Gamma(2,1)
    distribution using a histogram.
"""
import numpy as np                                 # numpy library.
import matplotlib.pyplot as plt                    # matplotlib library.
import scipy.stats as stats                        # scipy library.
from ARS import INITIAL_DICT, acceptance_rejection # functions implemented in ARS.py. 

# Generate samples from a Gamma(2,1) distribution using the acceptance-rejection method:
def h(x: float, shape: float, scale: float):
    return (shape-1)*np.log(x)-x/scale # h(x,k,\theta) = (k-1)*log(x)-x/\theta (Expression 18 of the document).
def Dh(x: float, shape: float, scale: float):
    return (shape-1)/x-1/scale         # h'(x,k,\theta) = (k-1)/x-1/\theta (Expression 19 of the document).
N       = 10000                                                              # Samples to generate. 
xi      = np.array([0.1, 0.5, 2.0, 5.0, 10.0])                               # Initial points for the envelope.
dict    = INITIAL_DICT(h=h, Dh=Dh, xi=xi, lb=0.001, ub=20, shape=2, scale=1) # Initial dictionary.
samples = acceptance_rejection(dict, N)                                      # Generate samples using ARS.

# Graph of the generated samples and the theoretical Gamma(2,1) distribution:
plt.figure(figsize = (10,6))
plt.hist(samples, bins = int(N/100), density = True, alpha = 1, color = 'b', label = 'Samples generated.')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, N)
real_dist = stats.gamma.pdf(x, 2, 0, 1) # PDF of the Gamma(2,1) distribution.
plt.plot(x, real_dist, 'k', label = r'PDF $Gamma(2,1)$')
plt.legend() 
plt.title(r"$Gamma(2,1)$ distribution - ARS vs PDF")
plt.savefig("ARS_gamma.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
plt.show()