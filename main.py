import torch
from matplotlib import pyplot as plt

from DataLoader import data_loader
from Model_Handler import ModelHandler
from PreProcessingHandler import PreProcessing


def plot_graph_loss(loss, title):
    plt.cla()
    plt.plot(range(len(loss)), loss, label=title)
    plt.xlabel('Batch Steps - axis')
    plt.ylabel('Loss Value - axis')
    plt.title("Loss")

    plt.legend()
    plt.show()
    return


def plot_graphs(train_loss_per_epoch, test_loss_per_epoch, accuracy_per_epoch):
    # plot_graph_acc(accuracy_per_epoch)
    plot_graph_loss(train_loss_per_epoch, test_loss_per_epoch)
    plot_graph_loss(train_loss_per_epoch, test_loss_per_epoch)
    plot_graph_loss(train_loss_per_epoch, test_loss_per_epoch)
    plot_graph_loss(train_loss_per_epoch, test_loss_per_epoch)


def main():
    # pre_processing = PreProcessing()
    # pre_processing.convert_folder_to_grayscale("flowers_rgb_class/splited_data", "flowers_gray_class/splited_data")
    # max_width, max_height = pre_processing.find_largest_image_size("flowers_gray")
    # target_size = (max_width, max_height)
    # pre_processing.resize_images("flowers_gray", target_size)

    train_dataset_gray, eval_loader_gray, test_dataset_gray, train_loader_gray, eval_loader_gray, test_loader_gray, train_dataset_rgb, test_dataset_rgb, eval_dataset_rgb, train_loader_rgb, eval_loader_rgb, test_loader_rgb = data_loader()

    model_handler = ModelHandler(test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb,
                                 train_loader_gray,
                                 eval_loader_gray, test_loader_gray, 64, 5, 0.0002, 0.0002)
    # Define Time
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # Train Model
    # start.record()
    # model_handler.pretrain_generator()

    g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g = model_handler.train()
    # end.record()
    model_handler.results_visualization()
    plot_graph_loss(g_loss_per_epoch, "g_loss_per_epoch")
    plot_graph_loss(d_loss_per_epoch, "d_loss_per_epoch")

    plot_graph_loss(test_losses_g, "test_losses_g")
    plot_graph_loss(val_losses_g, "val_losses_g")


if __name__ == "__main__":
    main()
