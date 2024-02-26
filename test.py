import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

# Assuming 'y' is a tensor with size [N]
# 'beta' is a tensor with size [N] and all elements equal to 1
# 'cov' is a covariance matrix with size [N, N]
# 先弄清楚loglikelihood function里的维度问题
y = torch.Tensor([1., 2., 3., 4., 5.])
N = y.size(0)
beta = torch.Tensor([[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]])
b = torch.Tensor([1., 1.])
mat1 =torch.mul(torch.diag(torch.ones(5)), b.unsqueeze(1).unsqueeze(1))
n_mc = beta.size(0)
y_expanded = y.expand(n_mc, -1)
y_expanded
# Create a MultivariateNormal distribution
mvn = MultivariateNormal(beta, covariance_matrix=mat1)
log_prob_list = mvn.log_prob(y_expanded)
log_prob_list.shape
print(log_prob_list)
beta.shape
mat1.size(2)
# Calculate the log probability of 'y'
log_prob_y = mvn.log_prob(y)

# Display the result
print(log_prob_y.item())  # Convert to a Python float if needed


# 假设有了mc sample z_k [n_mc, 5]: beta, sigma2, eta, ls[1], ls[2]
z_k = torch.rand(10, 5)
beta_k = z_k[:, 0]
beta_k.shape

t = torch.tensor([1, 2, 3, 4, 5])
torch.tile(t.unsqueeze(1), (1, 3))
y_expanded.shape

# 解决cov matrix的生成问题
# 先生成list of kernel matrices ([n_mc, N, N]), then multiply with eta ([n_mc]) 
x = torch.rand(5, 2)
ls = torch.Tensor([[1., 1.], [2., 2.], [3., 3.]])
eta = torch.rand(3)
nmc = ls.size(0)
N = x.size(0)
x_broadcasted = x.unsqueeze(0).expand(nmc, -1, -1)
ls_broadcasted = ls.unsqueeze(1).expand(-1, N, -1)
x_ls = x_broadcasted/ls_broadcasted
x_ls.shape
W_ = torch.ones((N,N))
up_ind, low_ind = np.triu_indices(N, 1)

mat_list = []
for i in range(nmc):
    mat_list.append(torch.exp( -0.5 * (torch.nn.functional.pdist(x_ls[i], p = 2).pow(2))).mul(eta[i]))

dist_mat = torch.exp( -0.5 * (torch.nn.functional.pdist(x_ls, p = 2).pow(2)))
mat_list[0].shape


# Create example matrices
A = torch.rand(5, 2)
B = torch.rand(3, 2)

# Broadcast 'A' and 'B' to have the same shape
A_broadcasted = A.unsqueeze(0).expand(B.size(0), -1, -1)
B_broadcasted = B.unsqueeze(1).expand(-1, A.size(0), -1)

B_broadcasted
# Perform element-wise division
result = A_broadcasted / B_broadcasted

# Display the result
print(result.shape)


def GP_cov(x, ls, eta):
    """Return a list of covariance matrices for each MC sample
    Args:
        x: [N, K]
        ls: [n_mc, K]
        eta: [n_mc]
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
        cov_mat = W_.mul(eta[i])
        cov_list.append(cov_mat)

    return cov_list

mat_list1 = GP_cov(x=torch.rand(5, 2), ls=torch.Tensor([[1., 1.], [2., 2.], [3., 3.]]), eta=torch.rand(3))
mat_list1[0]