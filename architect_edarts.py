import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def normalize(x, dim, min_v=1e-5):
    x = torch.clamp(x, min=min_v)
    normed = x / x.sum(dim=dim, keepdim=True)
    return normed

class Architect(object):

  def __init__(self, model, args):
    self.model = model
    self.lr = args.arch_learning_rate
    self.arch_params = model._arch_parameters
    self.n_edges = sum(1 for i in range(self._steps) for n in range(2+i))
    edge_scaling = np.zeros(k)
    n_inputs = 2
    ind = 0
    for n in range(self._steps):
        edge_scaling[ind:ind+n_inputs] = 1./n_inputs
        ind += n_inputs
        n_inputs += 1
    edge_scaling = torch.from_numpy(edge_scaling).cuda()
    self.edge_scaling = edge_scaling

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    for p in self.arch_params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    self._backward_step(input_valid, target_valid)
    for i, p in enumerate(self.arch_params):
        p.data.mul_(torch.exp(-lr * p.grad.data))
        if i < 2:
            p.data = normalize(p.data, -1)
        else:
            p.data = p.data * edge_scaling

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

