import json
import numpy as np
from matplotlib import pyplot as plt
from model import generator, discriminator
import torch
from dataset import get_data
from nilearn import plotting




# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    ngpu = 1
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
    # start with random noise
    rand_noise = torch.randn(5, device=device)
    rand_noise = rand_noise.to(device)
    # generate data
    generated_sample = netG(rand_noise)
    generated_sample = generated_sample.detach().numpy()

    # load the losses
    with open("saved/loss/G_losses5.json", 'r') as f:
        G_losses = json.load(f)
    with open("saved/loss/D_losses5.json", 'r') as f:
        D_losses = json.load(f)
    print(np.shape(generated_sample))
    generated_matrix = get_data.reshape_to_lower_matrix(generated_sample)

    fig = plt.figure()
    plotting.plot_matrix(generated_matrix, tri='lower', figure=(10, 8), vmax=1, vmin=-1)
    plt.show()


if __name__ == '__main__':
    main()
