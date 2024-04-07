import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, autograd
from Gan import UNetGenerator, Critic
import numpy as np

# constants
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.0001

# Prepares an image tensor for saving by converting it to a numpy array and scaling it
def prepare_to_save_image(image):
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()
    return (image_np - image_np.min()) / (image_np.max() - image_np.min())

# Creates a subplot with RGB, grayscale, and generated images
def make_subplot(rbg_image, grey_image, gen_image):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rbg_image)
    axs[0].axis('off')
    axs[1].imshow(grey_image.squeeze(), cmap='gray')
    axs[1].axis('off')
    axs[2].imshow(gen_image)
    axs[2].axis('off')
    plt.tight_layout()
    return plt

# Computes the Peak Signal-to-Noise Ratio (PSNR) between two images
def psnr(input_image, target_image):
    mse = torch.mean((input_image - target_image) ** 2)
    psnr_val = 10 * torch.log10(1 / mse)
    return psnr_val.detach()

# Computes the average PSNR between grayscale and RGB images using the model
def compute_psnr(loader_gray, loader_rgb, model_handler):
    psnr_values = []
    with torch.no_grad():
        for (gray_images, _), (rgb_images, _) in zip(loader_gray, loader_rgb):
            gray_images = gray_images.to("cuda")
            rgb_images = rgb_images.to("cuda")
            gen_images = model_handler.generator(gray_images)
            psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images, rgb_images)])
    avg_psnr = sum(psnr_values) / len(psnr_values)
    return avg_psnr

# Handles the model training and evaluation
class ModelHandler:
    def __init__(self, test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb, train_loader_gray, eval_loader_gray, test_loader_gray,
                 batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=LR, lr_C=LR, num_epochs_pre=EPOCHS):
        # Initializes the ModelHandler with necessary parameters
        pass

    # Pretrains the generator model
    def pretrain_generator(self):
        pass

    # Tests the generator model using test data
    def test_model(self, loader_gray, loader_rgb):
        pass

    # Visualizes the results of the trained model
    def results_visualization(self):
        pass

    # Computes the gradient penalty for improved WGAN training
    def gradient_penalty(self, real_images, fake_images):
        pass

    # Trains the generator and critic models
    def train(self):
        pass
