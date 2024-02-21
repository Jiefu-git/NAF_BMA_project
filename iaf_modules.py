import numpy as np
import torch
import math

import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from functools import reduce

# Set the default device to CPU
device = torch.device('cpu')

# aliasing
N_ = None

delta = 1e-7


# %------------ Useful Functions ------------%

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



class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


class Lambda(nn.Module):

    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, input):
        return self.function(input)


class WNlinear(Module):

    def __init__(self, in_features, out_features,
                 bias=True, mask=N_, norm=True):
        super(WNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mask', mask)
        self.norm = norm
        self.direction = Parameter(torch.Tensor(out_features, in_features))
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', N_)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, N_])
            weight = self.scale[:, N_].mul(direction)
        else:
            weight = self.scale[:, N_].mul(self.direction)
        if self.mask is not N_:
            weight = weight * Variable(self.mask)
            weight = weight.to(input.dtype)
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


def tile(x, r):
    return np.tile(x, r).reshape(x.shape[0], x.shape[1] * r)


class CWNlinear(Module):

    def __init__(self, in_features, out_features, context_features,
                 mask=N_, norm=True):
        super(CWNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.register_buffer('mask', mask)
        self.norm = norm
        self.direction = Parameter(torch.zeros(int(out_features), in_features))
        self.cscale = nn.Linear(context_features, int(out_features))
        self.cbias = nn.Linear(context_features, int(out_features))
        self.reset_parameters()
        self.cscale.weight.data.normal_(0, 0.001)
        self.cbias.weight.data.normal_(0, 0.001)

    def reset_parameters(self):
        self.direction.data.normal_(0, 0.001)

    def forward(self, inputs):
        input, context = inputs
        scale = self.cscale(context)
        bias = self.cbias(context)
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, N_])
            weight = direction
        else:
            weight = self.direction
        if self.mask is not N_:
            weight = weight * Variable(self.mask)
        return scale * F.linear(input, weight.type('torch.FloatTensor'), None) + bias, context

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


# %------------ MADE ------------%

def get_rank(max_rank, num_out):
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = np.random.choice(max_rank, excess, False)
    rank_out = np.delete(rank_out, remove_ind)
    np.random.shuffle(rank_out)
    return rank_out.astype('float32')


def get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype('float32')


def get_masks_all(ds, fixed_order=False, derank=1):
    # ds: list of dimensions dx, d1, d2, ... dh, dx,
    #                       (2 in/output + h hidden layers)
    # derank only used for self connection, dim > 1
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx-derank, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - derank
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [np.zeros([ds[i+1],ds[i]]).astype('float32') for i in range(len(ds)-1)]
    if derank == 1:
        assert np.all(np.diag(reduce(np.dot,ms[::-1])) == 0), 'wrong masks'

    return ms, rx


def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False, derank=1):
    ms, rx = get_masks_all([dim,]+[dh for i in range(num_layers-1)]+[dim,],
                           fixed_order, derank)
    ml = ms[-1]
    ml_ = (ml.transpose(1,0)[:, :, None]*(np.ones(int(num_outlayers)))).reshape(
                           dh, int(dim*num_outlayers)).transpose(1,0)
    # ml_ = (ml.transpose(1,0)[:,:,None]*([np.cast['float32'](1),] *\
    #                        num_outlayers)).reshape(
    #                        dh, dim*num_outlayers).transpose(1,0)
    ms[-1] = ml_
    return ms, rx


class MADE(Module):

    def __init__(self, dim, hid_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU(), fixed_order=False,
                 derank=1):
        super(MADE, self).__init__()

        oper = WNlinear

        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_outlayers = num_outlayers
        self.activation = activation

        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                           fixed_order, derank)
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx

        sequels = list()
        for i in range(num_layers-1):
            if i == 0:
                sequels.append(oper(dim, hid_dim, True, ms[i], False))
                sequels.append(activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, True, ms[i], False))
                sequels.append(activation)

        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                hid_dim, dim*num_outlayers, True, ms[-1])

    def forward(self, input):
        hid = self.input_to_hidden(input)
        return self.hidden_to_output(hid).view(
                -1, self.dim, self.num_outlayers)

    def randomize(self):
        ms, rx = get_masks(self.dim, self.hid_dim,
                           self.num_layers, self.num_outlayers)
        for i in range(self.num_layers-1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i*2].mask.is_cuda:
                mask = mask
            self.input_to_hidden[i*2].mask.data.zero_().add_(mask)
        self.rx = rx


class cMADE(Module):

    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU(), fixed_order=False,
                 derank=1):
        super(cMADE, self).__init__()

        oper = CWNlinear

        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = Lambda(lambda x: (activation(x[0]), x[1]))

        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                           fixed_order, derank)
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx

        sequels = list()
        for i in range(num_layers-1):
            if i == 0:
                sequels.append(oper(dim, hid_dim, context_dim,
                                    ms[i], False))
                sequels.append(self.activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, context_dim,
                                    ms[i], False))
                sequels.append(self.activation)

        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                hid_dim, dim*num_outlayers, context_dim, ms[-1])

    def forward(self, inputs):
        input, context = inputs
        hid, _ = self.input_to_hidden((input, context))
        out, _ = self.hidden_to_output((hid, context))
        return out.view(-1, self.dim, int(self.num_outlayers)), context

    def randomize(self):
        ms, rx = get_masks(self.dim, self.hid_dim,
                           self.num_layers, self.num_outlayers)
        for i in range(self.num_layers-1):
            mask = torch.from_numpy(ms[i])
            self.input_to_hidden[i*2].mask.zero_().add_(mask)
        self.rx = rx



