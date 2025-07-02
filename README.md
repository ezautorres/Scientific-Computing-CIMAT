# Scientific Computing – CIMAT (Fall 2024)

**Author:** Ezau Faridh Torres Torres  
**Advisor:** Dr. José Andrés Christen Gracia  
**Course:** Scientific Computing for Probability, Statistics, and Data Science  
**Institution:** CIMAT – Centro de Investigación en Matemáticas  
**Term:** Fall 2024 

This repository contains all course assignments and the final project from the graduate-level class *Scientific Computing for Probability, Statistics, and Data Science* at CIMAT (Fall 2024). The course was taught by my thesis advisor and significantly influenced the computational direction of my master's thesis, which focuses on solving inverse problems using Physics-Informed Neural Networks (PINNs).

## 📄 Table of Contents

- [Repository Structure](#repository-structure)
- [Technical Stack](#technical-stack)
- [Overview of Assignments](#overview-of-assignments)
  - [Assignment 1 – LU and Cholesky Decomposition](#assignment-1--lu-and-cholesky-decomposition)
  - [Assignment 2 – QR Decomposition and Least Squares](#assignment-2--qr-decomposition-and-least-squares)
  - [Assignment 3 – Numerical Stability](#assignment-3--numerical-stability)
  - [Assignment 4 – Eigenvalue Computation](#assignment-4--eigenvalue-computation)
  - [Assignment 5 – Stochastic Simulation](#assignment-5--stochastic-simulation)
  - [Assignment 6 – MCMC: Metropolis-Hastings](#assignment-6--mcmc-metropolis-hastings)
  - [Assignment 7 – Metropolis-Hastings in Multivariate Settings](#assignment-7--metropolis-hastings-in-multivariate-settings)
  - [Assignment 8 – MCMC with Hybrid Kernels and Gibbs Sampling](#assignment-8--mcmc-with-hybrid-kernels-and-gibbs-sampling)
  - [Final Project – Bayesian Inference for Weibull Parameters](#final-project--bayesian-inference-for-weibull-parameters)
- [Contact](#-contact)

---

## Repository Structure

Each assignment comprises the following elements:

- Python scripts with modular implementations of the required models and methods.
- A `report.pdf` that explains the methodology and findings.
- A `results/` directory with visual representations of the results.  

---

## Technical Stack

This project was developed in Python 3.11 using:

- **Core libraries:** `numpy`, `scipy`, `matplotlib`, `pandas`
- **Symbolic computation:** `sympy`
- **Statistical modeling & distributions:** `scipy.stats`
- **Plotting & visualization:** `seaborn`, `matplotlib`
- **Jupyter Notebooks** (for prototyping)

> Note: Each assignment may include additional libraries specified in the corresponding script headers.

---

## Overview of Assignments

The following section presents a concise overview of each task, highlighting its primary objective:

- **Assignment 1 – LU and Cholesky Decomposition**  
  Implementation of forward/backward substitution, LUP decomposition with pivoting, and Cholesky factorization. Includes performance comparison over increasing matrix sizes.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment1/results/ex6_as1.png" alt="Execution time comparison – Cholesky vs LUP" width="500"/>
  </div>

- **Assignment 2 – QR Decomposition and Least Squares**  
  Implementation of the modified Gram-Schmidt algorithm and its application to solve linear regression problems via QR decomposition. Includes polynomial fitting with varying degrees and sample sizes, and a performance comparison between the custom implementation and SciPy's QR routine.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment2/results/gram_schmidt.png" alt="Polynomial fitting using QR decomposition" width="500"/>
  </div>

- **Assignment 3 – Numerical Stability**  
  Analysis of numerical stability in Cholesky decomposition under perturbations. The task explores how matrix conditioning affects the results of QR-based least squares solutions. Includes timing comparisons and estimator sensitivity under both well-conditioned and ill-conditioned scenarios.

- **Assignment 4 – Eigenvalue Computation**  
  Application of Gershgorin’s theorem to estimate eigenvalue locations and implementation of the QR iteration algorithm to numerically compute eigenvalues. Includes comparisons with SciPy’s `eig` function across perturbation levels, confirming the accuracy and limitations of the custom QR method.

- **Assignment 5 – Stochastic Simulation**  
  Exploration of methods for sampling from distributions, including inverse transform sampling, linear congruential generators, and SciPy’s discrete random utilities. The assignment culminates with a full implementation of Adaptive Rejection Sampling (ARS), applied to simulate from Gamma(2,1), Normal, and Beta distributions with high accuracy.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment5/results/ARS_gamma.png" alt="Gamma(2,1) distribution sampled via ARS" width="500"/>
  </div>

- **Assignment 6 – MCMC: Metropolis-Hastings**  
  Simulation of Bernoulli data and posterior inference for the parameter \( 𝞺 \) using Metropolis-Hastings. Two proposal distributions were implemented: a Beta prior-informed proposal and a truncated Normal centered at the current state. The task includes analysis of irreducibility and ergodicity, along with convergence behavior as sample size increases.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment6/results/MS_normal.png" alt="Posterior sampling with Metropolis-Hastings" width="500"/>
  </div>

- **Assignment 7 – Metropolis-Hastings in Multivariate Settings**  
  Implementation of Metropolis-Hastings for bivariate and Gamma distributions, including random walk proposals and convergence diagnostics under different sample sizes and proposal variances.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment7/results/ex1/trayectory_ex1.png" alt="Posterior over alpha and beta" width="500"/>
  </div>

- **Assignment 8 – MCMC with Hybrid Kernels and Gibbs Sampling**  
  Simulation from complex posteriors using hybrid Metropolis-Hastings and Gibbs samplers. Includes examples with bivariate normals, Weibull likelihoods, and hierarchical Poisson-Gamma models for nuclear pump failure data.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/assignment8/results/p0.85.png" alt="Posterior" width="500"/>
  </div>

---

- **Final Project – Bayesian Inference for Weibull Parameters**  
  Full Bayesian treatment of a Weibull likelihood using MCMC. Implements both standard Metropolis-Hastings and adaptive proposals for posterior sampling of \(𝛼\) and \(λ\), with convergence diagnostics and posterior summaries for simulated datasets.

  <div align="center">
    <img src="https://github.com/ezautorres/Scientific-Computing-CIMAT/raw/main/final_project/results/trayectory_ex2.png" alt="Trayectory of alpha and lambda for Weibull model" width="500"/>
  </div>
---

## Learning Outcomes

Throughout the course, I gained practical experience in:

- Implementing numerical linear algebra algorithms from scratch
- Performing polynomial and spline interpolation
- Solving ordinary differential equations using numerical schemes
- Designing and evaluating stochastic simulation pipelines (e.g., ARS, MCMC)
- Analyzing convergence and stability in numerical methods
- Applying Bayesian inference via MCMC techniques to real data
- Writing clear scientific reports with integrated visualizations

## 📫 Contact

- 📧 Email: ezau.torres@cimat.mx  
- 💼 LinkedIn: [linkedin.com/in/ezautorres](https://linkedin.com/in/ezautorres)