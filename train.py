import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from matplotlib import pyplot as plt
import json
import numpy as np
import dataset.datasets as dataset
from model import discriminator, generator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

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


def train():
    print("Starting training")
    # all the parameters
    n_batch = 128
    n_epochs = 100
    n_eval = 200
    latent_dim = 91
    ngpu = 1
    lr = 0.001
    beta1 = 0.5
    nz = 91
    n_gpu = 1
    dataloader = dataset.FNetDataset()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                          else "cpu")

    netG = generator.Generator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    #print(netG)

    netD = discriminator.Discriminator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    # Print the model
    #print(netD)

    # Initialize MSELoss function
    criterion = nn.MSELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)

    # Print model's state_dict
    print("Generator's state_dict")
    for param_tensor in netG.state_dict():
        print(param_tensor, "\t", netG.state_dict()[param_tensor].size())
    print("Discriminator's state_dict")
    for param_tensor in netD.state_dict():
        print(param_tensor, "\t", netD.state_dict()[param_tensor].size())

    # Print optimiser's state_dict
    print("OptimizerG's state_dict")
    for var_name in optimizerG.state_dict():
        print(var_name, "\t", optimizerG.state_dict()[var_name])
    print("OptimizerD's state_dict")
    for var_name in optimizerD.state_dict():
        print(var_name, "\t", optimizerD.state_dict()[var_name])

    # Lists to keep track of progress
    mse_list = []
    G_losses = []
    D_losses = []
    iters = 0

    device, device_ids = prepare_device(n_gpu)

    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            data = torch.from_numpy(data).float().to(device)
            d_size = data.size(0)
            label = torch.full((10,), real_label, dtype=torch.float, device=device)
            #print(label)
            #print(type(label))
            # Forward pass real batch through D
            output = netD(data)
            output = output[1]
            #print(type(output))
            # Calculate loss on all-real batch
            errD_real = criterion(output, label[[1]])
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(5, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            output = output[1].flatten()
            errD_fake = criterion(output, label)
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
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, n_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == n_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(noise).detach().cpu()
                #img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        mse_img = functional.mse_loss(fake, data)
        mse_list.append(mse_img)

        # compute the MLSLoss for each batch of the dataloader

    # convert the list of mselosses
    mse_list = [mse_list[i].item() for i in range(len(mse_list))]

    return mse_list, G_losses, D_losses, netG, netD
