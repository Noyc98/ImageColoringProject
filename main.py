import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from DataLoader import data_loader
from Model_Handler import ModelHandler

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
LR = 0.0001


def plot_graph(loss, title, x_label='Batch', y_label='Loss'):
    plt.cla()
    plt.plot(range(len(loss)), loss, label=title)
    plt.xlabel(f'{x_label} Steps - axis')
    plt.ylabel(f'{y_label} Value - axis')
    plt.title(title)

    plt.legend()
    plt.savefig(f'{note_book_save_path}/results/graph_{title}')
    plt.close()
    return


def main():
    data = data_loader()
    train_loader_rgb, eval_loader_rgb, test_loader_rgb = data
    print("Finished data loading!")
    # Define and initialize your model handler
    model_handler = ModelHandler(train_loader_rgb, eval_loader_rgb, test_loader_rgb,
                                 batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=LR, lr_C=LR, num_epochs_pre=4)
    print("Finished ModelHandler!")
    # Train Model
    start_time = time.time()
    model_handler.pretrain_generator()
    end_time = time.time()
    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time
    print("Elapsed time in seconds:", elapsed_time)
    # Define Time
    start_time = time.time()
    c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c, mse_losses_per_epoch, wgan_losses_per_epoch, test_accuracy, val_losses_mse = model_handler.train()
    end_time = time.time()
    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time
    print("Elapsed time in seconds:", elapsed_time)
    # model_handler.results_visualization()
    # # plots
    # plot_graph(test_accuracy, "Test Accuracy", x_label='Batch', y_label="Accuracy")
    # plot_graph(g_loss_per_epoch, "Train Generator Loss", x_label='Epoch', y_label='Loss')
    # plot_graph(c_loss_per_epoch, "Train Critic Loss", x_label='Epoch', y_label='Loss')
    # plot_graph(accuracy, "Train Accuracy", x_label='Epoch', y_label="Accuracy")
    # plot_graph(val_losses_g, "Validation Generator Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(val_losses_c, "Validation Critic Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(val_losses_mse, "Validation MSE Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(mse_losses_per_epoch, "Train MSE Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(wgan_losses_per_epoch, "Train Wasserstein Loss", x_label='Epoch', y_label="Loss")
    # Averaging every 5 epochs for each metric

    # test_accuracy_avg = average_every_n_epochs(test_accuracy, n=3)
    # c_loss_per_epoch_avg = average_every_n_epochs(c_loss_per_epoch)
    # g_loss_per_epoch_avg = average_every_n_epochs(g_loss_per_epoch)
    # accuracy_avg = average_every_n_epochs(accuracy)
    # val_losses_g_avg = average_every_n_epochs(val_losses_g)
    # val_losses_c_avg = average_every_n_epochs(val_losses_c)
    # val_losses_mse_avg = average_every_n_epochs(val_losses_mse, n=10)
    # mse_losses_per_epoch_avg = average_every_n_epochs(mse_losses_per_epoch)
    # wgan_losses_per_epoch_avg = average_every_n_epochs(wgan_losses_per_epoch)

    # Plotting the averaged data
    # plot_graph(test_accuracy_avg, "Averaged Test Accuracy", x_label='Batch', y_label="Accuracy")
    # plot_graph(g_loss_per_epoch_avg, "Averaged Train Generator Loss", x_label='Epoch', y_label='Loss')
    # plot_graph(c_loss_per_epoch_avg, "Averaged Train Critic Loss", x_label='Epoch', y_label='Loss')
    # plot_graph(accuracy_avg, "Averaged Train Accuracy", x_label='Epoch', y_label="Accuracy")
    # plot_graph(val_losses_g_avg, "Averaged Validation Generator Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(val_losses_c_avg, "Averaged Validation Critic Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(val_losses_mse_avg, "Averaged Validation MSE Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(mse_losses_per_epoch_avg, "Averaged Train MSE Loss", x_label='Epoch', y_label="Loss")
    # plot_graph(wgan_losses_per_epoch_avg, "Averaged Train Wasserstein Loss", x_label='Epoch', y_label="Loss")

    # Run model
    accuracy = model_handler.test_model(test_loader_rgb)
    print(f"Test accuracy: {np.mean(accuracy)}")
    model_handler.results_visualization()


if __name__ == "__main__":
    main()
