import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as functional
import tqdm

import dataset.data_division
import dataset.get_data as get_data
import model.generator as generator
import model.discriminator as discriminator
import dataset.data_division as data_division
import dataset.datasets as datasets
import statistics
import seaborn


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
    netG.load_state_dict(torch.load('saved/models/modelG4.pth'))
    netD.load_state_dict(torch.load('saved/models/modelD4.pth'))
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


def main():
    dataset_healthy = dataset.data_division.get_testing_healthy_data(get_data.all_data_timeseries)
    result0 = test(dataset_healthy)
    new_result0 = []
    for i in range(len(result0)):
        new_result0.append(result0[i].item())
    print(result0)
    dataset_scz = dataset.data_division.get_testing_scz_data(get_data.all_data_timeseries)
    result1 = test(dataset_scz)
    result1 = [result1[i].item() for i in range(len(result1))]
    dataset_bd = dataset.data_division.get_bd_data(get_data.all_data_timeseries)
    result2 = test(dataset_bd)
    result2 = [result2[i].item() for i in range(len(result2))]
    dataset_adhd = dataset.data_division.get_adhd_data(get_data.all_data_timeseries)
    result3 = test(dataset_adhd)
    result3 = [result3[i].item() for i in range(len(result3))]
    df = pd.DataFrame(list(zip(new_result0, result1, result2, result3)),
                      columns=['Healthy', 'SCZ', 'BD', 'ADHD'])
    # Plot the loss
    plt.figure()
    plt.boxplot([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3]])
    plt.show()


if __name__ == '__main__':
    main()
