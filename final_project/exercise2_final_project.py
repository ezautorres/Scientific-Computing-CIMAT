# --------------------------------------------------------------------------------------------------------------
# Exercise 2 | Final Project | Scientific Computing for Probability, Statistics and Data Science.
# --------------------------------------------------------------------------------------------------------------
"""
@File name    : exercise2_final_project.py
@Main function: 
@Date         : 04 December 2024.
@Author       : Ezau Faridh Torres Torres.
@Email        : ezau.torres@cimat.mx
@Course       : Scientific Computing for Probability, Statistics and Data Science.
@Institution  : Mathematics Research Center (CIMAT).

@Instructions :
    The instructions are described in the report of the assignment.

@Description  :
    This uses the Metropolis-Hastings algorithm with hybrid proposal kernels in R^n to estimate the parameters of
    a Poisson regression model. The parameters to be estimated are a, b, and c.
"""
import numpy as np              # numpy library.
import scipy.stats as stats     # scipy library.
import matplotlib.pyplot as plt # matplotlib library.
from exercise1_final_project import METROPOLIS_HASTINGS_HYBRID_KERNELS # Import METROPOLIS_HASTINGS_HYBRID_KERNELS.
from auxiliary_functions import estim_burn_in_and_modes, chain_histograms, marginal_evolution_burn_in, trayectory2d3d

# --------------------------------------------------------------------------------------------------------------
# 1.- EXTRA FUNCTIONS: Functions (15) and (16) of the PDF.
# --------------------------------------------------------------------------------------------------------------
def g_b(x: np.ndarray, b: float) -> np.ndarray:
    return np.exp(-(x**2)/(2*b**2))

def lbda(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c * g_b(x - a, b)

# --------------------------------------------------------------------------------------------------------------
# 2.- OBJECTIVE FUNCTION: POSTERIOR (Function (20) of the PDF).
# --------------------------------------------------------------------------------------------------------------
def posterior(params: np.ndarray, x: np.ndarray, y: np.ndarray, mu_a: float, sigma_a: float,
              alpha_b: float, beta_b: float, alpha_c: float, beta_c: float) -> float:
    """
    Objective function: Posterior distribution of the parameters.
    
    Parameters
    ----------
    params : array_like
        Parameters of the distribution.
    x : array_like
        Independent variable data.
    y : array_like
        Independent variable data.
    mu_a : float
        Mean of the prior distribution of a.
    sigma_a : float
        Standard deviation of the prior distribution of a.
    alpha_b : float
        Alpha parameter of the prior distribution of b.
    beta_b : float
        Beta parameter of the prior distribution of b.
    alpha_c : float
        Parameter alpha of the prior distribution of c.
    beta_c : float
        Parameter beta of the prior distribution of c.
        
    Returns
    -------
    float
        Posterior of the parameters.
    """
    a, b, c = params                                                # Parameters.
    if b > 0 and c > 0:                                             # Restrictions for b and c.
        lambdas = lbda(x, a, b, c)                                  # Estimate lambdas with the parameters.
        likelihood = np.prod(stats.poisson.pmf(y, lambdas))         # Likelihood.
        prior_a = stats.norm.pdf(a, loc = mu_a, scale = sigma_a)    # Prior of a.
        prior_b = stats.gamma.pdf(b, a = alpha_b, scale = 1/beta_b) # Prior of b.
        prior_c = stats.gamma.pdf(c, a = alpha_c, scale = 1/beta_c) # Prior of c.
        return likelihood * prior_a * prior_b * prior_c             # Posterior.
    else:
        return 0

# --------------------------------------------------------------------------------------------------------------
# 3.- PROPOSALS.
# --------------------------------------------------------------------------------------------------------------

# Proposal 1 ---------------------------------------------------------------------------------------------------
def prop_gibbs_c_gen(params: np.ndarray, X: np.ndarray, y_sum: float, beta_c: float) -> np.ndarray:
    """
    Gibbs proposal for parameter c, given parameters a, b and data.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters.
    X : array_like
        Independent variable data.
    y_sum : float
        Sum of the data of the dependent variable.
    beta_c : float
        Beta parameter of the prior distribution of c.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    a, b, c = params                                                   # Current parameters.
    alpha_gibbs = 3 + y_sum                                            # Parameter alpha of the proposal.
    beta_gibbs = beta_c + np.sum(np.exp(-(X - a) ** 2 / (2 * b ** 2))) # Parameter beta of the proposal.
    c_new = stats.gamma.rvs(alpha_gibbs, scale = 1/beta_gibbs)         # Proposal of c.
    return np.array([a, b, c_new])

def prop_gibbs_c_pdf(params: np.ndarray, props: np.ndarray, X: np.ndarray, y_sum: float, beta_c: float) -> float:
    """
    Probability density of the Gibbs proposal for parameter c.

    Parameters
    ----------
    params : array_like
        Distribution parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop_gibbs_c_gen.
    """
    a, b, c = params                                                  # Current parameters.
    a_new, b_new, c_new = props                                       # Proposed parameters.
    alpha_gibbs = 3 + y_sum                                           # Parameter alpha of the proposal.
    beta_gibs = beta_c + np.sum(np.exp(-(X - a) ** 2 / (2 * b ** 2))) # Parameter beta of the proposal.
    return stats.gamma.pdf(c_new, alpha_gibbs, scale = 1/beta_gibs)   # PDF of the proposal.

# Proposal 2 ---------------------------------------------------------------------------------------------------
def prop_joint_normal_gen(params: np.ndarray, sg_a: float, sg_b: float, sg_c: float) -> np.ndarray:
    """
    Generates a joint normal proposal for the parameters.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters.
    sg_a : float
        Standard deviation of the proposal for a.
    sg_b : float
        Standard deviation of the proposal for b.
    sg_c : float
        Standard deviation of the proposal for c.
        
    Returns
    -------
    np.ndarray
        Joint normal proposal for the parameters.
    """
    a, b, c = params                       # Parameters.
    a_new = stats.norm.rvs(a, sg_a)        # Proposal for a.
    b_new = stats.norm.rvs(b, sg_b)        # Proposal for b.
    c_new = stats.norm.rvs(c, sg_c)        # Proposal for c.
    return np.array([a_new, b_new, c_new]) # Joint proposal.

def prop_joint_normal_pdf(params: np.ndarray, props: np.ndarray, sg_a: float, sg_b: float, sg_c: float) -> float:
    """
    Probability density of the joint normal proposal for the parameters.

    Parameters
    ----------
    params : array_like
        Distribution parameters.
    props : array_like
        Joint normal proposal for the parameters.
    The rest of the parameters are the same as in prop_joint_normal_gen.
    """
    a, b, c = params            # Current parameters.
    a_new, b_new, c_new = props # Proposed parameters.
    return stats.norm.pdf(a_new, a, sg_a) * stats.norm.pdf(b_new, b, sg_b) * stats.norm.pdf(c_new, c, sg_c)

# Proposal 3 ---------------------------------------------------------------------------------------------------
def prop_a_apriori_gen(params: np.ndarray, mu_a: float, sigma_a: float) -> np.ndarray:
    """
    Generates the a priori proposal for parameter a leaving parameters b and c fixed.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters.
    mu_a : float
        Mean of the a priori distribution of a.
    sigma_a : float
        Standard deviation of the a priori distribution of a.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    a, b, c = params                      # Current parameters.
    a_new = stats.norm.rvs(mu_a, sigma_a) # Proposal for a.
    return np.array([a_new, b, c])        # Joint proposal.

def prop_a_apriori_pdf(params: np.ndarray, props: np.ndarray, mu_a: float, sigma_a: float) -> float:
    """
    Probability density of the a priori proposal for parameter a.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters.
    props : array_like
        Proposed parameters.
    The rest of the parameters are the same as in prop_a_apriori_gen.
    """
    a, b, c = params    # Current parameters.
    a_new, b, c = props # Proposed parameters.
    return stats.norm.pdf(a_new, mu_a, sigma_a)

# Proposal 4 ---------------------------------------------------------------------------------------------------
def prop_b_apriori_gen(params: np.ndarray, alpha_b: float, beta_b: float) -> np.ndarray:
    """
    Generates the a priori proposal for parameter b leaving parameters a and c fixed.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters
    alpha_b : float
        Parameter alpha of the prior distribution of b.
    beta_b : float
        Beta parameter of the prior distribution of b.
    
    Returns
    -------
    np.ndarray
        Proposed parameters.
    """
    a, b, c = params                                   # Current parameters.
    b_new = stats.gamma.rvs(alpha_b, scale = 1/beta_b) # Proposal for b.
    return np.array([a, b_new, c])                     # Joint proposal.

def prop_b_apriori_pdf(params: np.ndarray, props: np.ndarray, alpha_b: float, beta_b: float) -> float:
    """
    Probability density of the a priori proposal for parameter b.
    
    Parameters
    ----------
    params : array_like
        Distribution parameters.
    props : array_like
        A priori proposal for parameter a.
    The rest of the parameters are the same as in prop_b_apriori_gen.
    """ 
    a, b, c = params    # Current parameters.
    a, b_new, c = props # Proposed parameters.
    return stats.gamma.pdf(b_new, alpha_b, scale = 1/beta_b)

# --------------------------------------------------------------------------------------------------------------
# 4.- Main Code.
# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(11111) # Seed for reproducibility.

    # Initial data and parameters ------------------------------------------------------------------------------
    X = np.array([27, 19, 21, 51, 16, 59, 16, 54, 52, 16, 31, 31, 54, 26, 19, 13, 59, 48, 54, 23, 50, 59, 55, 37, 61, 53, 56, 31, 34, 15, 41, 14, 13, 13, 32, 46, 17, 52, 54, 25, 61, 15, 53, 39, 33, 52, 65, 35, 65, 26, 54, 16, 47, 14, 42, 47, 48, 25, 15, 46, 31, 50, 42, 23, 17, 47, 32, 65, 45, 28, 12, 22, 30, 36, 33, 16, 39, 50, 13, 23, 50, 34, 19, 46, 43, 56, 52, 42, 48, 55, 37, 21, 45, 64, 53, 16, 62, 16, 25, 62])
    Y = np.array([1275, 325, 517, 0, 86, 0, 101, 0, 0, 89, 78, 83, 0, 1074, 508, 5, 0, 0, 0, 1447, 0, 0, 0, 0, 0, 0, 0, 87, 7, 37, 0, 15, 5, 6, 35, 0, 158, 0, 0, 1349, 0, 35, 0, 0, 12, 0, 0, 2, 0, 1117, 0, 79, 0, 13, 0, 0, 0, 1334, 56, 0, 81, 0, 0, 1480, 177, 0, 29, 0, 0, 551, 0, 1338, 196, 0, 9, 104, 0, 0, 3, 1430, 0, 2, 492, 0, 0, 0, 0, 0, 0, 0, 0, 1057, 0, 0, 0, 68,0, 87, 1362, 0])
    
    N_samp  = 2000  # Number of samples.
    mu_a    = 35    # Mean of the prior distribution of a.
    sigma_a = 5     # Standard deviation of the prior distribution of a.
    alpha_b = 2     # Parameter alpha of the prior distribution of b.
    beta_b  = 2/5   # Beta parameter of the prior distribution of b.
    alpha_c = 3     # Parameter alpha of the prior distribution of c.
    beta_c  = 3/950 # Beta parameter of the prior distribution of c.
    y_sum   = np.sum(Y) # Sum of the dependent variable data for proposal 1.
    sg_a    = sigma_a   # Standard deviation of the proposal for a in proposal 2.
    sg_b    = 1/beta_b  # Standard deviation of the proposal for b in proposal 2.
    sg_c    = 1/beta_c  # Standard deviation of the proposal for c in proposal 2.
    a0      = stats.norm.rvs(mu_a, sigma_a)              # Inicialization of a.
    b0      = stats.gamma.rvs(alpha_b, scale = 1/beta_b) # Inicialization of b.
    c0      = stats.gamma.rvs(alpha_c, scale = 1/beta_c) # Inicialization of c.
    x0      = np.array([a0, b0, c0])                     # Initial point for the chain.
    
    # Definition of kernels: their proposal functions and probability densities -------------------------------
    props_gen = [lambda params: prop_gibbs_c_gen(params, X, y_sum, beta_c),      # Gibbs proposal for c.
                 lambda params: prop_joint_normal_gen(params, sg_a, sg_b, sg_c), # Joint normal proposal.
                 lambda params: prop_a_apriori_gen(params, mu_a, sigma_a),       # A priori proposal for a.
                 lambda params: prop_b_apriori_gen(params, alpha_b, beta_b)]     # A priori proposal for b.
    props_pdf = [lambda params, props: prop_gibbs_c_pdf(params, props, X, y_sum, beta_c),      # PDF proposal 1.
                 lambda params, props: prop_joint_normal_pdf(params, props, sg_a, sg_b, sg_c), # PDF proposal 2.
                 lambda params, props: prop_a_apriori_pdf(params, props, mu_a, sigma_a),       # PDF proposal 3.
                 lambda params, props: prop_b_apriori_pdf(params, props, alpha_b, beta_b)]     # PDF proposal 4.

    # Markov chain simulation using Metropolis-Hastings with hybrid kernels ------------------------------------
    chain = METROPOLIS_HASTINGS_HYBRID_KERNELS(
            lambda params: posterior(params, X, Y, mu_a, sigma_a, alpha_b, beta_b, alpha_c, beta_c),
            props_gen, props_pdf, None, x0, N_samp)

    estim_burn_in_and_modes = estim_burn_in_and_modes(chain) # Estimation of burn-in and modes.
    burn_in = estim_burn_in_and_modes[0]                     # Burn-in.
    a_mode  = estim_burn_in_and_modes[1][0]                  # Mode of a.
    b_mode  = estim_burn_in_and_modes[1][1]                  # Mode of b.
    c_mode  = estim_burn_in_and_modes[1][2]                  # Mode of c.
    lambdas_est = lbda(X, a_mode, b_mode, c_mode)            # Estimate lambdas with the modes.

    # Viewing the results --------------------------------------------------------------------------------------
    print("\nResults:" + "-"*50)
    print("The initial distribution is:", x0)
    print( f"Burn-in                    : {burn_in}")
    print(rf"Mode of a                  : {a_mode}")
    print(rf"Mode of b                  : {b_mode}")
    print(rf"Mode of c                  : {c_mode}")

    # Graphs ---------------------------------------------------------------------------------------------------

    # Scatter plot of the data and the estimated model.
    plt.figure(figsize = (10,6))
    plt.scatter(X, Y, label = "Real Data.", color = "blue", lw = 2)
    plt.scatter(X, lambdas_est, label = rf"Estimated $\lambda$.", color = "red", lw = 2)
    plt.xlabel("Age (X)")
    plt.ylabel("Purchase amount (Y)")
    plt.title("Comparison between real data and estimated model.")
    plt.legend()
    plt.grid()
    plt.savefig("comparison_ex2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # Histograms of the parameters.
    ax = chain_histograms(chain = chain, burn_in = burn_in, params = ["$a$", "$b$", "$c$"],
                          modes = [a_mode, b_mode, c_mode], bins = [50,50,50])
    plt.tight_layout()
    plt.savefig("histogram_ex2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # Evolution of the marginal distributions of the parameters.
    ax = marginal_evolution_burn_in(chain = chain, burn_in = burn_in, params = ["$a$", "$b$", "$c$"])
    plt.tight_layout()
    plt.savefig("marginal_evolution_ex2.pdf", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500, format = 'pdf', transparent = True)
    plt.show()

    # Trayectory of the parameters.
    ax = trayectory2d3d(chain = chain, params = ["$a$", "$b$", "$c$"], modes = [a_mode, b_mode, c_mode])
    plt.tight_layout()
    plt.savefig("trayectory_ex2.png", bbox_inches = 'tight', pad_inches = 0.4, dpi = 500)
    plt.show()