from random import random

import numpy as np
from torch import randn

from dataset.get_data import *
import dataset.data_division as data_division


class FNetDataset:
    def __init__(self) -> None:
        self.data = data_division.get_training_data(all_data_timeseries)
        print("Shape of the dataset : ", self.data.shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index]


class FNetTestDataset:
    def __init__(self, dataset) -> None:
        self.data = dataset
        print("Shape of the dataset : ", self.data.shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index]


def get_real_samples():
    x_real = get_data.all_vectors
    return x_real, np.ones((np.size(x_real), 1))


def generate_fake_sample(generator, latent_dim, n):
    # generate points in the latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)


def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim*n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input
