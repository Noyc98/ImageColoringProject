import pickle
from matplotlib import pyplot as plt
from DataLoader import data_loader
from Model_Handler import ModelHandler
from PreProcessingHandler import PreProcessing

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.0001

def plot_graph(loss, title, x_label='Batch', y_label='Loss'):
    plt.cla()
    plt.plot(range(len(loss)), loss, label=title)
    plt.xlabel(f'{x_label} Steps - axis')
    plt.ylabel(f'{y_label} Value - axis')
    plt.title("Loss")

    plt.legend()
    plt.savefig(f"results/graph_{title}")
    return


def main():
    # pre_processing = PreProcessing()
    # pre_processing.convert_folder_to_grayscale("flowers_rgb_class/splited_data", "flowers_gray_class/splited_data")
    # max_width, max_height = pre_processing.find_largest_image_size("flowers_gray")
    # target_size = (max_width, max_height)
    # pre_processing.resize_images("flowers_gray", target_size)

    # data_loader()
    load_path = 'saved_models/data_loader.pkl'
    # Load the saved DataLoader objects and datasets
    with open(load_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Unpack the loaded data
    train_dataset_gray, eval_loader_gray, test_dataset_gray, train_loader_gray, eval_loader_gray, test_loader_gray, train_dataset_rgb, test_dataset_rgb, eval_dataset_rgb, train_loader_rgb, eval_loader_rgb, test_loader_rgb = loaded_data

    print("Finished data loading!")
    # Define and initialize your model handler
    model_handler = ModelHandler(test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb,
                                 train_loader_gray,
                                 eval_loader_gray, test_loader_gray, BATCH_SIZE, EPOCHS, LR, LR)
    print("Finished ModelHandler!")

    # Define Time
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # print(torch.cuda.is_available())

    # Train Model
    # start.record()
    # model_handler.pretrain_generator()
    # end.record()
    # torch.cuda.synchronize()
    # print(f"Pre-Training time: {start.elapsed_time(end)} milliseconds")
    # start.record()
    g_loss_per_epoch, c_loss_per_epoch, test_losses_g, val_losses_g, accuracy = model_handler.train()
    # end.record()
    # torch.cuda.synchronize()
    # print(f"Training time: {start.elapsed_time(end)} milliseconds")

    model_handler.results_visualization()

    # plots
    plot_graph(g_loss_per_epoch, "g_loss_per_epoch")
    plot_graph(c_loss_per_epoch, "c_loss_per_epoch")
    plot_graph(accuracy, title="PSNR accuracy per epoch", y_label="Accuracy")
    # Convert CUDA tensors to numpy arrays
    test_losses_g = [l.item() for l in test_losses_g]
    val_losses_g = [l.item() for l in val_losses_g]
    plot_graph(test_losses_g, "test_losses_g")
    plot_graph(val_losses_g, "val_losses_g")


if __name__ == "__main__":
    main()
