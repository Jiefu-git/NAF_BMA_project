import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itr

import torch
import torch.distributions as dist

# Set the default device to CPU
device = torch.device('cpu')

# log transformation of variables in the crime data dataset
def log_transform(crime_data, center=False, scale=False):
    crime_data.loc[:, crime_data.columns != 'So'] = np.log(crime_data.loc[:, crime_data.columns != 'So'])
    if center:
        crime_data.loc[:, crime_data.columns != 'y'] = (crime_data.loc[:, crime_data.columns != 'y'] -
                                                        crime_data.loc[:, crime_data.columns != 'y'].mean())
    if scale:
        crime_data.loc[:, crime_data.columns != 'y'] = (crime_data.loc[:, crime_data.columns != 'y'] /
                                                        crime_data.loc[:, crime_data.columns != 'y'].std())
    return crime_data


# Create model space index
def power_set_index(n_predictors):
    """Generates a matrix of variable indicators defining the space of
    all the models

    Args:
        n_predictors: Number of predictors to be considered
    Returns:
        A matrix with variable indicators"""
    power_set = itr.product([0, 1], repeat=n_predictors)
    array_index = []
    for i in list(power_set):
        array_index = array_index + [np.array(i)]
    array_index = np.array(array_index)
    ids = np.array([i for i in range(len(array_index))])
    return np.append(np.append(ids[:, None], np.ones(len(array_index))[:, None], axis=1), array_index, axis=1)


# posterior statistics of bayesian linear regression
def BayesReg(X, y):
    """Compute the marginal posterior distribution mean and std of beta and sigma^2, X is the
    design matrix without intercept"""

    g = len(X)
    n = len(X)

    # MLE
    XTX_inv = torch.inverse(torch.matmul(X.T, X))
    U = torch.matmul(XTX_inv, X.T)
    alphaml = torch.mean(y)
    betaml = torch.matmul(U, y)
    residuals = y - alphaml - torch.matmul(X, betaml)
    s2 = torch.matmul(residuals.t(), residuals)

    # Calculate kappa
    betatilde_minus_betaml = - betaml
    kappa = s2+torch.matmul(torch.matmul(betatilde_minus_betaml.t(), X.t()),torch.matmul(X, betatilde_minus_betaml))/(g + 1)

    # Calculate malphabayes and mbetabayes
    malphabayes = alphaml
    mbetabayes = g / (g + 1) * (betaml)

    # Calculate msigma2bayes
    msigma2bayes = kappa / (n - 3)

    # Calculate valphabayes and vbetabayes
    valphabayes = kappa / (n * (n - 3))
    vbetabayes = torch.diag(kappa * g / ((g + 1) * (n - 3)) * torch.inverse(torch.matmul(X.t(), X)))

    # Calculate vsigma2bayes
    vsigma2bayes = 2 * kappa**2 / ((n - 3) * (n - 4))

    return malphabayes, mbetabayes, msigma2bayes, valphabayes, vbetabayes, vsigma2bayes, kappa, s2, betaml


# Compare the true posterior distibution and the distribution of the generated beta's
def plot_beta_marginal_post(m, df, loc_beta, scale_beta, flow_sample):
    """Plot marginal posterior samples of beta"""
    # Set the degrees of freedom (shape parameter) for the t-distribution
    degrees_beta = df

    # Location and scale parameters
    loc_beta = loc_beta[m-1]
    scale_beta = np.sqrt(scale_beta[m-1,m-1])

    # Generate samples from the t-distribution
    beta_t_samples = loc_beta + scale_beta * np.random.standard_t(df=degrees_beta, size=1000)

    # Samples from flow model (alpha)
    beta_flow_samples = flow_sample[:,m+1]

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the KDE histogram for the first distribution
    sns.distplot(beta_t_samples, hist=False, kde=True, color='blue',
                 hist_kws={'edgecolor': 'black'}, label='True Posterior')

    # Plot the KDE histogram for the second distribution
    sns.distplot(beta_flow_samples, hist=False, kde=True, color='red',
                 hist_kws={'edgecolor': 'black'}, label='Flow Samples')

    # Set plot title and labels
    plt.title(f"Generated Samples and Posterior Distribution (Beta {m})")
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()


# Log-likelihood
# Here we transform the \phi by \phi = log(1+exp(\rho))
def regression_log_likelihood_torch(Y, X, beta, rho):
    """Computes log-likelihood of a standard regression model """
    n = len(X)
    mu = torch.matmul(beta, X.t())  # mean of the data likelihood
    # Y.permute(1,0) is [1, n]
    log_likelihood = (0.5*n*torch.log(torch.log(1+torch.exp(rho))) - 0.5*n*torch.log(2*torch.tensor(np.pi)) -
                      0.5*torch.log(1+torch.exp(rho))*((mu-Y.permute(1, 0))**2).sum(1))
    return log_likelihood


# Zellner's g Prior
# Normal(0, g * (\sigma^2) *(X'X)^{-1})
def log_pdf_zg_prior_1(theta, rho, X):
    num_observations = len(rho)
    g = len(X)
    p = theta.shape[1]  # number of covariates

    # precision matrix
    XTX = torch.matmul(X.T, X)
    XTX_inv = torch.inverse(XTX)

    # g/phi
    g_phi = g/torch.log(1+torch.exp(rho))
    # Thus, the precision matrix is g_phi * XTX_inv

    # Compute the log likelihood for the prior on Beta
    log_zg_prior = (-0.5*(1/g_phi)*(theta @ XTX @ theta.t()).diagonal() -
                    0.5 * torch.logdet(XTX_inv) - 0.5*p*torch.log(g_phi) - 0.5*p*torch.log(2 * torch.tensor([np.pi])))

    return log_zg_prior


# Normal(0, g * (\sigma^2) *(X'X)^{-1})
def log_pdf_zg_prior_2(theta, rho, X):
    S_G = len(rho)
    g = len(X)
    k = theta.shape[1]

    # Calculate the precision matrix for the i-th observation in PyTorch
    XTX = torch.matmul(X.T, X)
    XTX_inv = torch.inverse(XTX)

    # Initialize a tensor to store log likelihoods
    log_zg_prior = torch.empty(S_G)

    for i in range(S_G):
        precision_matrix = (g/torch.log(1+torch.exp(rho[i]))) * XTX_inv

        # Compute the log likelihood for the i-th observation using PyTorch distributions
        multivariate_normal = dist.MultivariateNormal(torch.zeros(k), precision_matrix)
        log_zg_prior[i] = multivariate_normal.log_prob(theta[i, :])

    return log_zg_prior


# log-likelihood of the base distribution
def log_prob_z0(z0):
    """Compute the log-likelihood of the initial base distribution of the flow"""
    # initial distribution is Normal(0,1)
    mean = 0.0
    stddev = 1.0
    log_likelihoods = torch.sum(-0.5 * ((z0 - mean) / stddev)**2 -
                                0.5 * torch.log(2 * torch.tensor([np.pi]).log() * stddev**2), dim=1)
    return log_likelihoods


# ELBO function for a fixed linear regression model
def elbo(param, X, Y):
    rho = param[:, 0]
    beta = param[:, 1:]

    # prior
    g = len(X)
    log_p_rho = torch.log((1/torch.log(1+torch.exp(rho)))*(torch.exp(rho)/(1+torch.exp(rho))))
    log_p_beta = log_pdf_zg_prior_1(theta=beta, rho=rho, X=X)

    # Data likelihood
    log_L = regression_log_likelihood_torch(Y=Y, X=X, beta=beta, rho=rho)

    out = log_p_rho + log_L + log_p_beta

    return out


def normal_density(theta, mu, sd):
    density = sp.stats.norm(loc=mu, scale=sd)
    pdf = density.pdf(theta)
    return pdf

