from random import shuffle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import os 


class JesterDataset(Dataset):
    def __init__(self, split=0.8):
        super(JesterDataset, self).__init__()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        print("loading data 1...")
        df1 = pd.read_excel(dir_path + '/../../dat/jester-data-1.xls', 
            names = ['num'] + ['joke_{}'.format(i) for i in range(100)],
            dtype=np.float32)
        print("loading data 2...")
        df2 = pd.read_excel(dir_path + '/../../dat/jester-data-2.xls', 
            names = ['num'] + ['joke_{}'.format(i) for i in range(100)],
            dtype=np.float32)
        print("loading data 3...")
        df3 = pd.read_excel(dir_path + '/../../dat/jester-data-3.xls', 
            names = ['num'] + ['joke_{}'.format(i) for i in range(100)],
            dtype=np.float32)

        df = df1.append(df2.append(df3))
        
        num_rows = df.shape[0]
        num_train = int(np.ceil(split * num_rows))
        num_test = num_rows - num_train

        self._all = df.values[:,1:].copy()

        #shuffle the data as book 3 is a little more sparse than books 1 and 2.  Good practice to shuffle data anyhow.
        #use a constant random seed so that the shuffle is always the same in case we wish to continue training.
        np.random.seed(9618) 
        np.random.shuffle(self._all)

        self._train = torch.from_numpy(self._all[:num_train, :])
        self._train.mul_(0.1)

        self._test = torch.from_numpy(self._all[num_train:, :])
        self._test.mul_(0.1)

        # if torch.cuda.is_available():
        #     self._train = self._train.cuda()
        #     self._test = self._test.cuda()

        self._active_set = self._train

        print("Training on {} entries, Testing on {}, total entries: {}".format(num_train, num_test, num_rows))

    def __len__(self):
        return len(self._active_set)
    
    def __getitem__(self, idx):
        return self._active_set[idx, :]

    def train(self):
        self.active_set = self._train

    def test(self):
        self.active_set = self._test

    def getTestBatch(self):
        return self._test

    def getTrainBatch(self):
        return self._train

