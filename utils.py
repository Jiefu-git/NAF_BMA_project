import torch
import numpy as np
import torch.nn as nn

# Set the default device to CPU
device = torch.device('cpu')

delta = 1e-7
pi_tensor = torch.tensor(np.pi)
sigmoid = lambda x: torch.nn.functional.sigmoid(x) * (1-delta) + 0.5 * delta


def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


c = - 0.5 * torch.log(2*pi_tensor)


def log_normal(x, mean, log_var, eps=0.00001):
    return - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var/2. + c


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    maximum = lambda x: x.max(axis)[0]
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max
    return B


def log_mean_exp(A, axis=-1):
    return log_sum_exp(A, axis, sum_op=torch.mean)


def log_sum_exp_np(A, axis=-1, sum_op=np.sum):
    A_max = np.max(A, axis, keepdims=True)
    B = np.log(sum_op(np.exp(A-A_max),axis,keepdims=True)) + A_max
    return B


def log_mean_exp_np(A, axis=-1):
    return log_sum_exp_np(A, axis, sum_op=np.mean)



def softplus(x, delta=1e-7):
    softplus_ = nn.Softplus()
    return softplus_(x) + delta


def sigmoid(x, delta=1e-7):
    sigmoid_ = nn.Sigmoid()
    return sigmoid_(x) * (1 - delta) + 0.5 * delta


def sigmoid2(x):
    return sigmoid(x) * 2.0


def logsigmoid(x):
    return -softplus(-x)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


def logit(x):
    return log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


def sum1(x):
    return x.sum(1)


def sum_from_one(x):
    if len(x.size()) > 2:
        return sum_from_one(sum1(x))
    else:
        return sum1(x)



