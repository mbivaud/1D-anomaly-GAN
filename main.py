from matplotlib import pyplot as plt
from torchinfo import summary

from train import *
from test import *


def main():
    mse_list, G_losses, D_losses, netG, netD = train()

    # print the network architecture
    print("----------- generator -----------")
    summary(netG)
    print("----------- discriminator -----------")
    summary(netD)

    # save the model
    torch.save(netG.state_dict(), 'saved/models/modelG8.pth')
    torch.save(netD.state_dict(), 'saved/models/modelD8.pth')

    # save the losses
    with open("saved/loss/G_losses8.json", 'w') as f:
        json.dump(G_losses, f, indent=2)
    with open("saved/loss/D_losses8.json", 'w') as f:
        json.dump(D_losses, f, indent=2)

    # Plot the loss
    #plt.figure()
    #plt.boxplot(mse_list)
    #plt.show()

    # Testing the network with new data
    dataset_healthy = dataset.data_division.get_testing_healthy_data(get_data.all_data_timeseries)
    result0 = test(dataset_healthy)
    new_result0 = []
    for i in range(len(result0)):
        new_result0.append(result0[i].item())
    #print(result0)
    dataset_scz = dataset.data_division.get_testing_scz_data(get_data.all_data_timeseries)
    result1 = test(dataset_scz)
    result1 = [result1[i].item() for i in range(len(result1))]
    dataset_bd = dataset.data_division.get_bd_data(get_data.all_data_timeseries)
    result2 = test(dataset_bd)
    result2 = [result2[i].item() for i in range(len(result2))]
    dataset_adhd = dataset.data_division.get_adhd_data(get_data.all_data_timeseries)
    result3 = test(dataset_adhd)
    result3 = [result3[i].item() for i in range(len(result3))]
    df = pd.DataFrame(list(zip(mse_list, new_result0, result1, result2, result3)),
                      columns=['Train', 'Healthy', 'SCZ', 'BD', 'ADHD'])
    # Plot the loss
    fig = plt.figure()
    fig.suptitle('Reconstruction MSE between original and reconstructed data', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3], df.iloc[:, 4]])
    ax.set_title('MSE')
    ax.set_xlabel('1=train(healthy), 12=healthy, 3=SCZ, 4=BD, 5=ADHD')
    plt.show()


if __name__ == '__main__':
    main()
