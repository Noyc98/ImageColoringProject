import pickle  
from matplotlib import pyplot as plt  
from DataLoader import data_loader  
from Model_Handler import ModelHandler 
from PreProcessingHandler import PreProcessing  

# Constants 
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001

# Function to plot a graph
def plot_graph(loss, title, x_label='Batch', y_label='Loss'):
    # Clear the current figure
    plt.cla()
    # Plot the loss data
    plt.plot(range(len(loss)), loss, label=title)
    # Set labels for x-axis and y-axis
    plt.xlabel(f'{x_label} Steps - axis')
    plt.ylabel(f'{y_label} Value - axis')
    # Set title for the plot
    plt.title(title)
    # Add legend to the plot
    plt.legend()
    # Save the plot as an image
    plt.savefig(f"results/graph_{title}")
    # Close the plot
    plt.close()

# Main function
def main():
    # Load saved DataLoader objects and datasets from a file
    load_path = 'saved_models/data_loader.pkl'
    with open(load_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Unpack the loaded data
    train_dataset_gray, eval_loader_gray, test_dataset_gray, train_loader_gray, eval_loader_gray, test_loader_gray, train_dataset_rgb, test_dataset_rgb, eval_dataset_rgb, train_loader_rgb, eval_loader_rgb, test_loader_rgb = loaded_data

    print("Finished data loading!")

    # Initialize ModelHandler object with datasets and loaders
    model_handler = ModelHandler(test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb,
                                 train_loader_gray,
                                 eval_loader_gray, test_loader_gray,
                                 batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=LR, lr_C=LR, num_epochs_pre=4)
    print("Finished ModelHandler!")

    # Train the model
    model_handler.pretrain_generator()
    g_loss_per_epoch, c_loss_per_epoch, test_losses_g, val_losses_g, accuracy = model_handler.train()
    model_handler.results_visualization()

    # Plot various graphs
    plot_graph(g_loss_per_epoch, "g_loss_per_epoch")
    plot_graph(c_loss_per_epoch, "c_loss_per_epoch")
    plot_graph(accuracy, title="PSNR accuracy per epoch", y_label="Accuracy")
    plot_graph(test_losses_g, "test_losses_g")
    plot_graph(val_losses_g, "val_losses_g")

if __name__ == "__main__":
    main()
