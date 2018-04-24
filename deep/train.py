import numpy as np
from collections import OrderedDict
import time
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision.utils as vutils
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from classes import JesterDataset, MaskedMSELoss, ConsumerClassifier

# check for GPU
usecuda = torch.cuda.is_available()
if usecuda:
	print('training on GPU')
else:
	print('training on CPU')

# prep dataset
dataset = JesterDataset()
dataloader = DataLoader(dataset, batch_size=100000000, shuffle=False, num_workers=1)

# prep model
net = ConsumerClassifier()

if os.path.isfile('runs/recent_training.pt') and os.path.isfile('runs/recent_training.pkl'):
    net.load_state_dict(torch.load('runs/recent_training.pt'))
    with open ('runs/recent_training.pkl','rb') as f:
        info = pickle.load(f)
    print("Resuming training of previously discovered training run")
else:
    print("Starting new training run")
    info = {'dir':'runs/' + net.description(), 'epoch':0}
print(net)

# prep tensorboardx writer
writer = SummaryWriter(info['dir'])
writer.add_graph(net, Variable(dataset[0]))

# start training maybe
print("begin training")
start = time.time()

learning_rate = 0.1
momentum = 0.9
# num_epochs = 200000
num_epochs = 18800
critereon = MaskedMSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = torch.optim.Adam(net.parameters, lr=learning_rate)

if usecuda:
    net = net.cuda()

#see if we can fit everything in memory, otherwise, we must use batches.
x_train = Variable(dataset.getTrainBatch())
x_test = Variable(dataset.getTestBatch())
if usecuda:
    x_train = x_train.cuda()
    x_test = x_test.cuda()
usebatch = False

try:
    for epoch in range(info['epoch'], info['epoch'] + num_epochs):
        # forward
        y_train = net(x_train)
        loss = critereon(x_train, y_train)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        writer.add_scalar("Loss", loss.data[0], epoch)
        if (epoch % 100) == 0:
            print("Epoch: {:>4} Loss: {:6f}  Time: {:5.2f}".format(epoch, loss.data[0], time.time() - start))

            dat = []
            for z in net.parameters():
                dat.append(z.cpu().data.numpy())
            writer.add_histogram("parameters", z, epoch)
            
            #swap dataset to test mode and check loss
            y_test = net(x_test)
            loss = critereon(x_test, y_test)
            writer.add_scalar("Test Loss", loss.data[0], epoch)
            print("  Test Loss: {:6f}".format(loss.data[0]))

except KeyboardInterrupt:
    print("Exiting early due to keyboard interrupt")

#save to disk
info['epoch'] = epoch
net = net.cpu()
torch.save(net.state_dict(), 'runs/recent_training.pt')
torch.save(net.state_dict(), writer.file_writer.get_logdir() + '/training.pt')
with open('runs/recent_training.pkl', 'wb') as f:
    pickle.dump(info, f)

#clean up
writer.close()

print("Main thread exited OK")