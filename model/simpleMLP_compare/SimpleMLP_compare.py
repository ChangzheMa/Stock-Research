import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLPCompare(nn.Module):
    def __init__(self):
        super(SimpleMLPCompare, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(200, 2048),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.network(x)
        return x

