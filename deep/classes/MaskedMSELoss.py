import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self, exclude_value=9.0, size_average=True, reduce=True):
        super(MaskedMSELoss, self).__init__()
        self.exclude_value = exclude_value
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        m = input.lt(self.exclude_value)
        input = torch.masked_select(input, m)
        target = torch.masked_select(target, m)

        return F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)

