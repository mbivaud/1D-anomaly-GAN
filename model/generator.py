import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        # Find value of latent_dim
        latent_dim = 5
        self.ngpu = ngpu
        self.fc1 = nn.Linear(latent_dim, 10)
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 91)
        # output should be of form (latent_dim, 91)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

