#!/bin/env/python3

import numpy as np
import torch.nn as nn
from datetime import datetime

class ConsumerClassifier(nn.Module):
    def __init__(self, front_layers = [256, 256], back_layers =[256, 256], num_embs=5, num_records=100):
        super(ConsumerClassifier, self).__init__()

        #build front network (classifier)
        front = []
        front_size = num_records
        for layer_size in front_layers:
            front.append(nn.Linear(front_size, layer_size))
            front.append(nn.ReLU())
            front_size = layer_size
        front.append(nn.Linear(front_size, num_embs))
        front.append(nn.LogSoftmax(dim=0))
        self.classifier = nn.Sequential(*front)

        #build back network (verifier)
        back = []
        back_size = num_embs
        for layer_size in back_layers:
            back.append(nn.Linear(back_size, layer_size))
            back.append(nn.ReLU())
            back_size = layer_size
        back.append(nn.Linear(back_size, num_records))
        self.verifier = nn.Sequential(*back)


        self._description = 'f' + '-'.join(str(x) for x in front_layers) + '-cLSM{}'.format(num_embs) + '-b' + '-'.join(str(x) for x in back_layers) + datetime.now().strftime('-%b%d_%H-%M-%S')


    def classify(self, vector):
        return self.classifier(vector)
    
    def forward(self, vector):
        return self.verifier(self.classifier(vector))

    def predict(self, vector):
        return self.verifier(vector)

    def description(self):
        return self._description