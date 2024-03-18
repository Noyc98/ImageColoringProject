import cv2
import numpy as np

from DataLoader import data_loader
from PreProcessingHandler import PreProcessing
from Gan import UNetGenerator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Define a function to display images
def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, img in enumerate(images):
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axs[i].imshow(img_np)
        axs[i].axis('off')
    plt.show()

def main():
    # pre_processing = PreProcessing()
    # pre_processing.convert_folder_to_grayscale("flowers_color", "flowers_gray")
    # max_width, max_height = pre_processing.find_largest_image_size("flowers_gray")
    # target_size = (max_width, max_height)
    # pre_processing.resize_images("flowers_gray", target_size)



    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0002
    batch_size = 64
    num_epochs = 5

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = data_loader()

    # Initialize networks
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Define loss function and optimizers
    criterion_gan = nn.BCEWithLogitsLoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # for img in imgs:
            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate RGB images from grayscale
            gen_imgs = generator(real_imgs)
            fake_pred = discriminator(gen_imgs)
            fake_pred_2d = fake_pred[:, :, 0, 0]

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion_gan(fake_pred_2d, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()


            # Measure discriminator's ability to classify real and fake images
            d_real_imgs = discriminator(real_imgs)
            d_real_imgs_2d = d_real_imgs[:, :, 0, 0]
            real_loss = criterion_gan(d_real_imgs_2d, valid)

            d_gen_imgs = discriminator(gen_imgs.detach())
            d_gen_imgs_2d = d_gen_imgs[:, :, 0, 0]
            fake_loss = criterion_gan(d_gen_imgs_2d, fake)

            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )


            batches_done = epoch * len(train_loader) + i
            if batches_done % 100 == 0:
                show_images(gen_imgs[:5])
                save_image(gen_imgs.data[:25], "images/%d.jpg" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    main()