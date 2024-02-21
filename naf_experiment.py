# ~ 5 itr/s
# If you encounter nan output, try to reduce the learning rate
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

import torch
import iaf_modules as iaf_modules
import naf_helper
import flows
from naf_helper import log_transform
from naf_helper import power_set_index


# Set the default device to CPU
device = torch.device('cpu')

# data
# Loading the crime data
UScrime = pd.read_csv("UScrime.csv", index_col=0)  # Full dataset of crime data
# UScrime = pd.read_csv("UScrime_train.csv", index_col= 0) # Training dataset of crime data

UScrime_log = log_transform(UScrime.copy(), center=False, scale=False)
UScrime_log_center = log_transform(UScrime.copy(), center=True, scale=False)
UScrime_log_center_scale = log_transform(UScrime.copy(), center=True, scale=True)


# Choices of the size of the variable set (and which variables)
vars_n = "3"
vars_test = ["M", "Ed", "Prob"]
if vars_n == "4":
    vars_test = ["M", "So", "Ed", "Prob"]
elif vars_n == "3":
    vars_test = ["M", "Ed", "Prob"]
elif vars_n == "2":
    vars_test = ["M", "Prob"]

# Full model design matrix and variable transformation choice
design_type = "logCS"
X_design = UScrime_log_center_scale.loc[:, vars_test].values
if design_type == "logC":
    X_design = UScrime_log_center.loc[:, vars_test].values
elif design_type == "log":
    X_design = UScrime_log.loc[:, vars_test].values
elif design_type == "logCS":
    X_design = UScrime_log_center_scale.loc[:, vars_test].values

X_design_intercept = np.append(np.ones((len(X_design), 1)), X_design, axis=1)
Y = UScrime_log_center.loc[:, "y"].values

# Model space indicator matrix generation
K = len(vars_test)  # Number of predictors
model_space = power_set_index(K).astype(int)


# Convert X_design and y to PyTorch tensors
X_design_tensor = torch.tensor(X_design_intercept, dtype=torch.float32)  # center + scale
y_tensor = torch.tensor(Y, dtype=torch.float32)
y_tensor = y_tensor.unsqueeze(1)
y_tensor_C_scale = (y_tensor-torch.mean(y_tensor))/torch.std(y_tensor)  # center + scale

# Check the dimensions of the tensors
print("X_design_tensor shape:", X_design_tensor.shape)
print("y_tensor_C_scale shape:", y_tensor_C_scale.shape)

# check the first few elements of the tensors
print(X_design_tensor[0:5, :])
print(y_tensor_C_scale[0:5])


# Model Initialization
random.seed(12)
iter_num = 300
n_mc = 10
critical_i = iter_num * (3/4)

X_list = []  # each model has its own design matrix
for model in model_space:
    X_list.append(X_design_tensor[:, model[1:].astype(bool)])

Y = y_tensor_C_scale

# q(M)
q_A = np.ones(len(model_space)) / len(model_space)
# p(M)
pi_A = np.ones(len(model_space)) / len(model_space)
# q(M) container
q_A_array = np.zeros((iter_num, len(model_space)))


# Initialization
dim_list = []
for model in model_space:
    dim = int(sum(model[1:]))
    dim_list.append(dim)

model_list = []
optimizer_list = []
dim_list = []

for model in model_space:
    dim = int(sum(model[1:]))
    mdl = flows.IAF_DSF(dim+1, 128, 1, 8, activation=iaf_modules.softplus, num_ds_dim=16, num_ds_layers=8)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=0.001, betas=(0.9, 0.999))
    dim_list = []
    model_list.append(mdl)
    optimizer_list.append(optimizer)

ELBO = torch.empty(iter_num, len(model_list), 1)


time_counter = np.array(time.monotonic())
for i in tqdm(range(iter_num)):
    q_A_star = np.ones(len(model_list))/len(model_list)
    for j in range(len(model_list)):
        optimizer_list[j].zero_grad()
        zk, logdet, logPz0, context = model_list[j].sample(n_mc)
        losses = logPz0-naf_helper.elbo(zk, X_list[j], Y) - logdet
        ELBO[i, j] = losses.mean().item()
        # update q_A
        # We are minimizing the -ELBO, so we need the '-' sign here
        q_A_star[j] = np.exp(-losses.mean().item() + np.log(pi_A[j]))

        # update inference network parameters
        if i > critical_i:
            loss = q_A[j]*(losses.mean())
            loss.backward()
            optimizer_list[j].step()
        else:
            loss = losses.mean()
            loss.backward()
            optimizer_list[j].step()
    # Averaged version
    if i > critical_i:
        # Averaged version
        q_A = q_A_star.copy() / np.sum(q_A_star)
    q_A_array[i, :] = q_A.copy()
    time_counter = np.append(time_counter, time.monotonic())

print(np.sort(np.round(q_A, 2))[::-1])

ELBO_np = ELBO.numpy()
elbo = np.squeeze(ELBO_np, axis=-1)
#np.savetxt("D:/Jeff/MSU/Research/NAF/naf_cpu/elbo_result/elbo_naf_epoch1000_mc1.csv", elbo, delimiter=',', comments='')

