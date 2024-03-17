import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import random
import shutil
from torchvision.datasets import ImageFolder

# Define a custom target_transform function that returns the same label for all images
# def dummy_label(_):
#     return 0  # Assign the same label for all images
#

def data_loader():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    # Create the ImageFolder dataset using the custom target_transform
    dataset = ImageFolder(root=r'dummy', transform=transform)

    data_path = r'dummy/splited_data'

    # Path to destination folders
    train_folder = os.path.join(data_path, 'train')
    val_folder = os.path.join(data_path, 'eval')
    test_folder = os.path.join(data_path, 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

    # Sets the random seed
    random.seed(42)

    # Shuffle the list of image filenames
    random.shuffle(imgs_list)

    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.7)
    val_size = int(len(imgs_list) * 0.15)
    test_size = int(len(imgs_list) * 0.15)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Save index for each img
    train_idx = []
    test_idx = []
    eval_idx = []
    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
            train_idx.append(i)
        elif i < train_size + val_size:
            dest_folder = val_folder
            eval_idx.append(i)
        else:
            dest_folder = test_folder
            test_idx.append(i)
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))

    # Create dataSet
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_dataset, eval_loader, test_dataset, train_loader, eval_loader, test_loader

