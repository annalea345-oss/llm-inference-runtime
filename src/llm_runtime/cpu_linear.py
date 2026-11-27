"""Opt-in two-row CPU inference projections; imported only with local extras."""

import torch
from torch import nn
from torch.nn import functional as F


class RowwiseDecodeLinear(nn.Linear):
    """Keep weights/parameter names, using two GEMVs for a two-request decode.

    Attention and KV updates still run as one model batch. Other shapes, devices,
    dtypes, and training/autograd use the ordinary Linear implementation.
    """

    def __init__(self, original: nn.Linear):
        # Meta initialization avoids allocating another full copy of the weights.
        super().__init__(original.in_features, original.out_features,
                         original.bias is not None, device='meta', dtype=original.weight.dtype)
        self.weight = original.weight
        self.bias = original.bias
        self.train(original.training)
        self.enabled = True
        self.rowwise_calls = 0

    def forward(self, inputs):
        if (self.enabled and not self.training and not torch.is_grad_enabled()
                and inputs.ndim == 3 and inputs.shape[:2] == (2, 1)
                and inputs.device.type == self.weight.device.type == 'cpu'
                and inputs.dtype == self.weight.dtype == torch.float32):
            self.rowwise_calls += 1
            return torch.cat([F.linear(row, self.weight, self.bias)
                              for row in inputs.split(1, dim=0)], dim=0)
        return F.linear(inputs, self.weight, self.bias)


def install_rowwise_decode(model):
    """Replace plain Linear modules in a newly loaded, uninstrumented model.

    No global class patch, bound-method closures, or additional weight storage.
    Custom Linear subclasses are left intact. Load before serving any requests.
    """
    installed = []
    for name, child in list(model.named_children()):
        if type(child) is nn.Linear:
            replacement = RowwiseDecodeLinear(child)
            setattr(model, name, replacement)
            installed.append(replacement)
        else:
            installed.extend(install_rowwise_decode(child))
    return installed
