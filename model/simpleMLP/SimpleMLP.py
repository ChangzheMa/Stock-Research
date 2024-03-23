import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.network(x)


