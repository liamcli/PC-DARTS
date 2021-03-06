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

  def __init__(self, model, criterion, args):
    self.model = model
    self.lr = args.arch_learning_rate
    self.arch_params = model._modules['module']._arch_parameters
    self.n_edges = sum(1 for i in range(4) for n in range(2+i))
    self.n_inputs = 2
    self.n_nodes = 4

    #edge_scaling = np.zeros(self.n_edges)
    #n_inputs = 2
    #ind = 0
    #for n in range(4):
    #    edge_scaling[ind:ind+n_inputs] = 1./n_inputs
    #    ind += n_inputs
    #    n_inputs += 1
    #edge_scaling = torch.from_numpy(edge_scaling).cuda()
    #self.edge_scaling = edge_scaling
    self.criterion = criterion
    self.grad_clip = args.grad_clip

  def step(self, input_valid, target_valid):
    self._backward_step(input_valid, target_valid)
    nn.utils.clip_grad_norm_(self.arch_params, self.grad_clip)

    for i, p in enumerate(self.arch_params):
        p.data.mul_(torch.exp(-self.lr * p.grad.data))
        print(torch.norm(p.grad.data, p=float('inf')))
        if i < 2:
            p.data.clamp_(min=1e-5)
            p.data.div_(p.data.sum(dim=-1, keepdim=True))
        else:
            node_weights = torch.zeros([self.n_edges]).cuda()
            offset = 0
            n_inputs = self.n_inputs
            for i in range(self.n_nodes):
                node_weights[offset : offset + n_inputs] = sum(
                    p.data[offset : offset + n_inputs]
                )
                offset += n_inputs
                n_inputs += 1
            p.data = p.data / node_weights

    for p in self.arch_params:
        if p.grad is not None:
            p.grad.zero_()

  def _backward_step(self, input_valid, target_valid):
    logits = self.model(input_valid)
    loss = self.criterion(logits, target_valid)
    loss.backward()

