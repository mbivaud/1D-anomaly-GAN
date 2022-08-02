import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from matplotlib import pyplot as plt
import json
import numpy as np
import dataset.datasets as dataset
from model import discriminator, generator

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def summarize_performance(epoch, gen, disc, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = dataset.get_real_samples()
    # evaluate discriminator on real examples
    _, acc_real = disc.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = dataset.generate_fake_sample(gen, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = disc.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def train(netG, netD, criterion, optimizerG, optimizerD, epoch, data, device, batch_size, latent_dim):

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    # Format batch
    data = data.float().to(device)
    d_size = data.size(0)
    data = data.view(batch_size, -1)
    label = torch.full((10,), real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = netD(data)
    output = output[1]
    #print(type(output))
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    #print(errD_real)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(batch_size, latent_dim, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach())
    # Calculate D's loss on the all-fake batch
    output = output[1].flatten()
    errD_fake = criterion(output, label[[1]])
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake)[1].view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label[[1]])
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()

    mse_img = functional.mse_loss(fake, data)

    return errG, errD, mse_img, D_x, D_G_z1, D_G_z2
