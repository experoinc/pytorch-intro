#!/bin/env python3

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from tqdm import trange, tqdm

dat = pd.read_csv('rankings.csv')
cids = dat['CustomerID']
dat.drop('CustomerID', inplace=True, axis=1)

class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=200):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=False)
        
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
    
    def user(self, user):
        return self.user_factors(user)
    
    def item(self, item):
        return self.item_factors(item)

# no need to train on all data, since it's sparse
dat = dat.as_matrix()
#dat = dat[:500,:500]
rows, cols = dat.nonzero()

# check for GPU
usecuda = torch.cuda.is_available()
if usecuda:
	print('training on GPU')
else:
	print('training on CPU')

# instantiate model, loss, and optim
n_users = dat.shape[0]
n_items = dat.shape[1]

model = MatrixFactorization(n_users, n_items, n_factors=400)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

if usecuda:
	model.cuda()
	loss_func.cuda()

# prep data for factorizer ingestion
inpt1 = []
inpt2 = []
labl = []

for row, col in zip(*(rows, cols)):
	inpt1.append(np.long(row))
	inpt2.append(np.long(col))
	labl.append(dat[row, col])

inpt1 = Variable(torch.LongTensor(inpt1))
inpt2 = Variable(torch.LongTensor(inpt2))
labl = Variable(torch.FloatTensor(np.array(labl)))

if usecuda:
    inpt1 = inpt1.cuda()
    inpt2 = inpt2.cuda()
    labl = labl.cuda()

# train the factorizer
history = []
epochs  = 500

for epoch in trange(epochs, desc='epoch'):
	# Predict and calculate loss
	prediction = model(inpt1, inpt2)
	loss = loss_func(prediction, labl)

	# Backpropagate
	optimizer.zero_grad()
	loss.backward()

	# Update the parameters
	optimizer.step()
		
	history.append(float(loss.data))

	del loss, prediction

print('\n')

# save weights and losses   
model.eval()

np.save('loss.npy', np.array(history), allow_pickle=False)

if usecuda:
	torch.save(model.state_dict(), 'gpu_model.bin')
else:
	torch.save(model.state_dict(), 'cpu_model.bin')




