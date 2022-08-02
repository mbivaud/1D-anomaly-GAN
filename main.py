import torch
from matplotlib import pyplot as plt
from torchinfo import summary
from torch.utils.data import DataLoader, ConcatDataset

from train import *
from test import *
from sklearn.model_selection import KFold
import dataset.datasets as dataset


def main():
    # configuration option
    k_folds = 5
    num_epochs = 1
    loss_function = nn.MSELoss()
    ngpu = 1
    lr = 0.001
    batch_size = 10
    latent_dim = 10

    torch.manual_seed(123)

    # training dataloader
    my_dataset = dataset.data_division.get_healthy(get_data.all_data_timeseries)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # lists to keep track of progress
    mse_train = {}
    mse_test = {}
    G_losses = {}
    D_losses = {}

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(my_dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=batch_size, sampler=test_subsampler)

        # init neural networks
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                              else "cpu")

        netG = generator.Generator(ngpu).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
        # print(netG)
        netD = discriminator.Discriminator(ngpu).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))
        # print the network architecture
        print("----------- generator -----------")
        summary(netG)
        print("----------- discriminator -----------")
        summary(netD)

        # Setup Adam optimizers for both G and D
        optimizerD = optim.SGD(netD.parameters(), lr=lr)
        optimizerG = optim.SGD(netG.parameters(), lr=lr)

        mse_list_temp = []
        G_losses_temp = []
        D_losses_temp = []

        # run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            # print epoch
            print(f'starting epoch {epoch + 1}')

            # set current loss value
            current_loss = 0.0

            # iterate over the dataloader for training data
            for i, data in enumerate(trainloader):

                # zero the gradient
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                # training cycle - forward, loss, backward, optimization
                errG, errD, mse_img, D_x, D_G_z1, D_G_z2 = train(netG, netD,
                                                                 loss_function,
                                                                 optimizerG, optimizerD,
                                                                 epoch, data, device,
                                                                 batch_size, latent_dim)

                # Set total and correct
                G_losses_temp.append(errG.item())
                D_losses_temp.append(errD.item())
                mse_list_temp.append(mse_img.item())
                #mse_list_temp = [mse_list_temp[i].item() for i in range(len(mse_list_temp))]

                # print statistics
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(trainloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Process is complete.
        print('Training process has finished. Saving trained model.')
        # save results
        mse_train[fold] = mse_list_temp
        G_losses[fold] = G_losses_temp
        D_losses[fold] = D_losses_temp

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path_g = f'./saved/models/modelG9-fold-{fold}.pth'
        torch.save(netG.state_dict(), save_path_g)
        save_path_d = f'./saved/models/modelD9-fold-{fold}.pth'
        torch.save(netD.state_dict(), save_path_d)
        # save the losses
        with open("saved/loss/G_losses9.json", 'w') as f:
            json.dump(G_losses, f, indent=2)
        with open("saved/loss/D_losses9.json", 'w') as f:
            json.dump(D_losses, f, indent=2)

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            result_temp = test(testloader,
                               save_path_g, save_path_d,
                               batch_size, latent_dim)
            mse_test[fold] = result_temp

            # Print accuracy
            #print('Accuracy for fold %d: %d' % (fold, result_temp))
            print('--------------------------------')

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        my_sum = 0.0
        for key, value in mse_test.items():
            print(f'Fold {key}: {value} %')
            my_sum += sum(value)
        print(f'Average: {my_sum / len(mse_test.items())} %')

    df_train = pd.DataFrame(mse_train)
    df_test = pd.DataFrame(mse_test)
    lists_g = sorted(G_losses.items())
    x = [i for i in range(38)]
    x_g, y_g = zip(*lists_g)
    lists_d = sorted(D_losses.items())
    x_d, y_d = zip(*lists_d)
    # Plot the loss
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].boxplot([df_train.iloc[:, 0],
                       df_train.iloc[:, 1],
                       df_train.iloc[:, 2],
                       df_train.iloc[:, 3],
                       df_train.iloc[:, 4]])
    axs[0, 0].set_title('MSE original-reconstructed TRAINING')
    axs[0, 0].set_xlabel('fold id')
    axs[0, 0].set_ylim([0, 0.2])
    axs[0, 1].boxplot([df_test.iloc[:, 0],
                       df_test.iloc[:, 1],
                       df_test.iloc[:, 2],
                       df_test.iloc[:, 3],
                       df_test.iloc[:, 4]])
    axs[0, 1].set_title('MSE original-reconstructed TESTING')
    axs[0, 1].set_xlabel('fold id')
    axs[0, 1].set_ylim([0, 0.2])
    axs[1, 0].plot(x, y_g[0], 'r')
    axs[1, 0].plot(x, y_g[1], 'b')
    axs[1, 0].plot(x, y_g[2], 'g')
    axs[1, 0].plot(x, y_g[3], 'y')
    axs[1, 0].plot(x, y_g[4], 'm')
    axs[1, 0].set_title('Generator loss')
    print(y_g[3])
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].set_ylim([0, 0.6])
    axs[1, 1].plot(x, y_d[0], 'r')
    axs[1, 1].plot(x, y_d[1], 'b')
    axs[1, 1].plot(x, y_d[2], 'g')
    axs[1, 1].plot(x, y_d[3], 'y')
    axs[1, 1].plot(x, y_d[4], 'm')
    axs[1, 1].set_title('Discriminator loss')
    axs[1, 1].set_xlabel('epochs')
    axs[1, 1].set_ylim([0, 0.6])
    plt.show()


if __name__ == '__main__':
    main()
