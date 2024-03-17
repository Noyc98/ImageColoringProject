from DataLoader import data_loader
from PreProcessingHandler import PreProcessing
from Gan import UNetGenerator, Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

def main():
    pre_processing = PreProcessing()
    # pre_processing.convert_folder_to_grayscale("flowers_color", "flowers_gray")
    
    # max_width, max_height = pre_processing.find_largest_image_size("flowers_gray")
    # target_size = (max_width, max_height)
    # # pre_processing.resize_images("flowers_gray", target_size)
    # pre_processing.extend_dataSet_laplacian("flowers_gray","extended_dataSet",4)
    # pre_processing.resize_images("extended_dataSet", target_size)
    # pre_processing.resize_and_replace_images("extended_dataSet","extended_dataSet", max_width, max_height)



    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0002
    batch_size = 64
    num_epochs = 5

    #disc

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = data_loader()

    # Initialize networks
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate RGB images from grayscale
            gen_imgs = generator(real_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real and fake images
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            if batches_done % 100 == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    main()