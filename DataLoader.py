import pickle
from torch.utils.data import Subset
import random
import shutil
from torchvision.datasets import ImageFolder
import os
from torchvision import transforms
from torch.utils.data import DataLoader


def create_data_set(color_mode='gray'):
    """
    Create dataset and split it into train, validation, and test sets.

    Args:
    - color_mode (str): Color mode of the images. It can be 'gray' or 'rgb'.

    Returns:
    - tuple: A tuple containing paths to train, validation, and test folders, list of image filenames,
             path to the data folder, and the dataset object.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    if color_mode == 'gray':
        dataset_path = 'flowers_gray_class'
    elif color_mode == 'rgb':
        dataset_path = 'flowers_rgb_class'
    else:
        raise ValueError("Invalid color mode. Use 'gray' or 'rgb'.")

    # Create the ImageFolder dataset using the custom target_transform
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # Path to destination folders
    data_path = os.path.join(dataset_path, 'splited_data')
    train_folder = os.path.join(data_path, 'train')
    val_folder = os.path.join(data_path, 'eval')
    test_folder = os.path.join(data_path, 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Create a list of image filenames in 'data_path'
    images_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    return train_folder, val_folder, test_folder, images_list, data_path, dataset


def data_loader():
    """
    Create and save data loaders for train, validation, and test sets.

    Returns:
    - None
    """
    save_path = 'saved_models/data_loader.pkl'
    train_folder_gray, val_folder_gray, test_folder_gray, images_list_gray, data_path_gray, dataset_gray = create_data_set(
        color_mode='gray')
    train_folder_rgb, val_folder_rgb, test_folder_rgb, images_list_rgb, data_path_rgb, dataset_rgb = create_data_set(
        color_mode='rgb')

    # Sets the random seed
    random.seed(42)

    # Shuffle the list of image filenames
    random.shuffle(images_list_gray)

    # determine the number of images for each set
    train_size = int(len(images_list_gray) * 0.7)
    val_size = int(len(images_list_gray) * 0.15)
    test_size = int(len(images_list_gray) * 0.15)

    # Save index for each img
    train_idx = []
    test_idx = []
    eval_idx = []
    # Copy image files to destination folders
    for i, f in enumerate(images_list_gray):
        if i < train_size:
            train_idx.append(i)
        elif i < train_size + val_size:
            eval_idx.append(i)
        else:
            test_idx.append(i)

    # Create dataSet
    # Gray
    train_dataset_gray = Subset(dataset_gray, train_idx)
    test_dataset_gray = Subset(dataset_gray, test_idx)
    eval_dataset_gray = Subset(dataset_gray, eval_idx)
    # RGB
    train_dataset_rgb = Subset(dataset_rgb, train_idx)
    test_dataset_rgb = Subset(dataset_rgb, test_idx)
    eval_dataset_rgb = Subset(dataset_rgb, eval_idx)

    # Create data loaders
    train_loader_gray = DataLoader(train_dataset_gray, batch_size=32)
    eval_loader_gray = DataLoader(eval_dataset_gray, batch_size=32)
    test_loader_gray = DataLoader(test_dataset_gray, batch_size=32)
    train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=32)
    eval_loader_rgb = DataLoader(eval_dataset_rgb, batch_size=32)
    test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=32)

    data = (
        train_dataset_gray, eval_loader_gray, test_dataset_gray, train_loader_gray, eval_loader_gray, test_loader_gray,
        train_dataset_rgb, test_dataset_rgb, eval_dataset_rgb, train_loader_rgb, eval_loader_rgb, test_loader_rgb)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
