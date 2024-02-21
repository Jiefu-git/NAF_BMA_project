import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
import iaf_modules as iaf_modules
from iaf_modules import log
from torch.autograd import Variable
import utils as utils
import numpy as np


sum_from_one = iaf_modules.sum_from_one

# Set the default device to CPU
device = torch.device('cpu')


# Flows
class BaseFlow(Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]
        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(np.ones((n, self.context_dim)).astype('float32')))
        return self.forward((spl, lgd, context))


class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

        self.act_a = lambda x: utils.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: utils.softmax(x, dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=utils.delta):

        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0*ndim:1*ndim])
        b_ = self.act_b(dsparams[:, :, 1*ndim:2*ndim])
        w = self.act_w(dsparams[:, :, 2*ndim:3*ndim])

        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2*ndim:3*ndim], dim=2) + \
            utils.logsigmoid(pre_sigm) + \
            utils.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj, 2).sum(2)
        logdet_ = logj + np.log(1-delta) - (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet

        return xnew, logdet


class IAF_DSF(BaseFlow):
    mollify = 0.0

    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DSF, self).__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers,
                    num_ds_multiplier*(hid_dim//dim)*num_ds_layers,
                    activation, fixed_order)
            self.out_to_dsparams = nn.Conv1d(num_ds_multiplier*(hid_dim//dim)*num_ds_layers,
                                             3*num_ds_layers*num_ds_dim, 1)
            self.reset_parameters()

        self.sf = SigmoidFlow(num_ds_dim)

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)

        inv = np.log(np.exp(1-utils.delta)-1)
        for i in range(self.num_ds_layers):
            nc = self.num_ds_dim
            nparams = nc * 3
            s = i*nparams
            self.out_to_dsparams.bias.data[s:s+nc].uniform_(inv, inv)

    def forward(self, inputs):
        x, logdet, context = inputs
        log_prob_z0 = torch.sum(-0.5 * x**2 - 0.5 * torch.log(2 * torch.tensor([np.pi])), dim=1)
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, iaf_modules.cMADE):
            out = out.permute(0, 2, 1)
            dsparams = self.out_to_dsparams(out).permute(0, 2, 1)
            nparams = self.num_ds_dim*3

        mollify = self.mollify
        h = x.view(x.size(0), -1)
        for i in range(self.num_ds_layers):
            params = dsparams[:, :, i*nparams:(i+1)*nparams]
            h, logdet = self.sf(h, logdet, params, mollify)

        return h, logdet, log_prob_z0, context

