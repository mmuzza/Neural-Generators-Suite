import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        out = self.net(x)
        return out