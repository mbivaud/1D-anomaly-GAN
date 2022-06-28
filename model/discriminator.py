import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        n_input = 91
        self.ngpu = ngpu
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x2 = torch.sigmoid(self.fc3(x1))
        return x1, x2


"""
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""