import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as functional
import tqdm

import dataset.data_division
import dataset.get_data as get_data
import model.generator as generator
import model.discriminator as discriminator
import dataset.datasets as datasets


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def my_mean_squared_error(list_diff):
    result = 0
    for i in range(len(list_diff)):
        result = pow(list_diff[i], 2)
    result = result / len(list_diff)
    return result


def anomaly_score(x, z, disc, gen, Lambda=0.1):
    """

    :param z: point of the latent space
    :param x: real data to be tested
    :param disc: discriminator
    :param gen: generator
    :param Lambda:
    :return: the loss value
    """
    G_z = gen(z)
    x_feature, _ = disc(x)
    G_z_feature, _ = disc(G_z)

    residual_loss = torch.sum(torch.abs(x-G_z))
    discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature))

    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    return total_loss


def test(dataset_to_test):
    print("Starting test")
    # parameters
    ngpu = 1
    n_epochs = 100
    lr = 0.0002
    beta1 = 0.5
    # setup dataset and dataloader
    dataloader = datasets.FNetTestDataset(dataset_to_test)
    # build model architecture
    netG = generator.Generator(ngpu)
    netD = discriminator.Discriminator(ngpu)
    # load the weights of the trained model
    netG.load_state_dict(torch.load('saved/models/modelG6.pth'))
    netD.load_state_dict(torch.load('saved/models/modelD6.pth'))
    # prepare model for testing
    netG.eval()
    netD.eval()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                          else "cpu")
    netG.to(device)
    netD.to(device)
    # Lists to keep track of progress
    anom_losses = []
    iters = 0
    # start with random noise
    z = torch.randn(91, 5, device=device)
    z = z.to(device)
    # input optimizer
    z_optim = optim.Adam([z], lr)
    list_diff = []

    for i, tested_data in enumerate(tqdm.tqdm(dataloader), 0):
        tested_data = torch.from_numpy(tested_data).float().to(device)
        for epoch in range(n_epochs):
            an_loss = anomaly_score(tested_data, z, netD, netG, Lambda=0.01)
            an_loss.backward()
            z_optim.step()

            anom_losses.append(an_loss.float())

            #if epoch % 100 == 0:
                #print(an_loss)

        reconstructed_data = netG(z)
        #print(tested_data)
        #print(reconstructed_data)
        diff_data = functional.mse_loss(reconstructed_data, tested_data)
        list_diff.append(diff_data)

    return list_diff
