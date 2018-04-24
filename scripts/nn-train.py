import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision.utils as vutils
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

class FirstNet(nn.Module):
    def __init__(self, layers = [64]):
        super(FirstNet, self).__init__()

        _layers = []
        input_size = 1
        for i, layer_size in enumerate(layers):
            _layers.append(nn.Linear(input_size, layer_size))
            _layers.append(nn.ReLU())
            input_size = layer_size
        _layers.append(nn.Linear(input_size, 1))

        self.model = nn.Sequential(*_layers)


    def forward(self, x):
        return self.model(x.view(-1,1))

net = FirstNet([64])

print(net)

tbwriter = SummaryWriter()
X = Variable(torch.FloatTensor([0.0]))
tbwriter.add_graph_onnx(net)

learning_rate = 0.1
num_epochs = 2000

use_cuda = torch.cuda.is_available()
if use_cuda:
    net = net.cuda()

data_iter = iter(dataloader)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
net.train()

for epoch in range(num_epochs):
    for X, Y in dataloader:
        if use_cuda:
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
        else:
            X = Variable(X)
            Y = Variable(Y)
        pred = net.forward(X)
        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbwriter.add_scalar("Loss", loss.data[0])

    if (epoch % 100 == 99):
        print("Epoch: {:>4} Loss: {}".format(epoch, loss.data[0]))
        for name, param in net.named_parameters():
            tbwriter.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        logFig(epoch)
