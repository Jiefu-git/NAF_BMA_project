import torch
import numpy as np
import pandas as pd
import time
import seaborn as sns
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal

import flows
import iaf_modules

delta = 1e-7
pi_tensor = torch.tensor(np.pi)
c = - 0.5 * torch.log(2*pi_tensor)

def softplus(x, delta=1e-7):
    softplus_ = torch.nn.Softplus()
    return softplus_(x) + delta


def log_normal(x, mean=torch.Tensor([0.0]), log_var=torch.Tensor([0.0]), eps=0.00001):
    return - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var/2. + c

log_var_prior = torch.log(softplus(torch.sqrt(torch.log(torch.Tensor([2.0])))) ** 2)

def log_logNormal(x, mean=torch.Tensor([0.0]), log_var=log_var_prior, eps=0.000001):
    return - (torch.log(x)-mean) ** 2 / (2. * torch.exp(log_var) + eps) - torch.log(x) - log_var/2. + c


def log_prior_GP(param):
    """return log prior for n_mc MC samples
    param [n_mc, 1+1+1+K]: sigma2, beta, eta, ls[1], ls[2]. MC samples of the NAF last iteration """
    n_mc = param.size(0)
    sigma2_k = torch.exp(param[:, 0])
    beta_k = param[:, 1]
    eta_k = torch.exp(param[:, 2])
    ls1_k = torch.exp(param[:, 3])
    ls2_k = torch.exp(param[:, 4])

    log_prior = log_normal(x=beta_k) + log_logNormal(x=sigma2_k) + log_logNormal(x=eta_k) + \
                log_logNormal(x=ls1_k) + log_logNormal(x=ls2_k)
    
    return log_prior


# Gaussian Process covariance matrices list
def GP_cov(x, ls, eta, sigma2):
    """Return a list of covariance matrices
    Args:
        x: [N, K]
        ls: [n_mc, K]
        eta: [n_mc]
        sigma2: [n_mc]
        """
    n_mc = ls.size(0)
    N = x.size(0)
    up_ind, low_ind = np.triu_indices(N, 1)
    x_broadcasted = x.unsqueeze(0).expand(n_mc, -1, -1)
    ls_broadcasted = ls.unsqueeze(1).expand(-1, N, -1)
    x_ls = x_broadcasted/ls_broadcasted
    # Empty list
    cov_list = []

    for i in range(n_mc):
        dist_mat = torch.exp( -0.5 * (torch.nn.functional.pdist(x_ls[i], p = 2).pow(2)))
        W_ = torch.ones((N, N))
        W_[up_ind, low_ind] = dist_mat
        W_[low_ind, up_ind] = dist_mat
        noise_cov = torch.diag(torch.ones(N)) * sigma2[i]
        cov_mat = W_.mul(eta[i]) + noise_cov
        cov_list.append(cov_mat)

    return cov_list

# Gaussian Process log likelihood
def GP_log_likelihood(x, y, param):
    """
    data log-likelihood for Gaussian Process: y ~ GP(beta, RBFkernel(x))

    Input:
    x: [N, K]
    y: [N]
    param [n_mc, 1+1+1+K]: sigma2, beta, eta, ls[1], ls[2]. MC samples of the NAF last iteration 
    
    Output:
    log_likelihood: [n_mc]
    """
    N = x.size(0)
    K = x.size(1)
    n_mc = param.size(0)
    sigma2_k = torch.exp(param[:, 0])
    beta_k = param[:, 1]
    eta_k = torch.exp(param[:, 2])
    ls_k = torch.exp(param[:, 3:(3+K)])

    # re-form tensors
    y_expanded = y.expand(n_mc, -1) # [n_mc, N]
    beta_expanded = torch.tile(beta_k.unsqueeze(1), (1, N)) # [n_mc, N]
    cov_list = GP_cov(x=x, ls=ls_k, eta=eta_k, sigma2=sigma2_k)
    cov_stacked = torch.stack(cov_list)
    mvn = MultivariateNormal(beta_expanded, covariance_matrix=cov_stacked)
    log_prob = mvn.log_prob(y_expanded)

    return log_prob


