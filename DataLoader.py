import torch
from torch.utils.data import Subset
from torchvision.datasets import  Flowers102
from torchvision import transforms
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
LR = 0.0001
def data_loader(color_mode='gray', batch_size=32):
    # Define transformations for RGB images
    rgb_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # Loading the RGB dataset
    rgb_train = Flowers102(root='./data', download=True, split='test', transform=rgb_transform)
    rgb_test = Flowers102(root='./data', split='train', transform=rgb_transform)
    rgb_val = Flowers102(root='./data', split="val", transform=rgb_transform)

    # Creating RGB dataloaders
    train_loader_rgb = torch.utils.data.DataLoader(rgb_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    eval_loader_rgb = torch.utils.data.DataLoader(rgb_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader_rgb = torch.utils.data.DataLoader(rgb_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    data = (train_loader_rgb, eval_loader_rgb, test_loader_rgb)
    return data