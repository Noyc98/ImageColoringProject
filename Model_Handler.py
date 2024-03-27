from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from PIL import Image
from Gan import UNetGenerator, Discriminator
import Wgan
import numpy as np


BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 0.00005

def prepare_to_save_image(image):
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()
    return (image_np - image_np.min()) / (image_np.max() - image_np.min())


def make_subplot(rbg_image, grey_image, gen_image):
    # Plot images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(rbg_image)
    axs[0].axis('off')

    axs[1].imshow(grey_image.squeeze(), cmap='gray')
    axs[1].axis('off')

    axs[2].imshow(gen_image)
    axs[2].axis('off')

    plt.tight_layout()
    return plt

def psnr(input_image, target_image):
    """
    Computes the PSNR between the input and target images.
    """
    mse = torch.mean((input_image - target_image) ** 2)
    psnr_val = 10 * torch.log10(1 / mse)
    return psnr_val


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


class ModelHandler:
    def __init__(self, test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb, train_loader_gray,
                 eval_loader_gray,
                 test_loader_gray, batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=LR, lr_D=LR,
                 num_epochs_pre=EPOCHS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_epochs_pre = num_epochs_pre
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.batch_size = batch_size
        self.train_loader_rgb = train_loader_rgb
        self.eval_loader_rgb = eval_loader_rgb
        self.test_loader_rgb = test_loader_rgb
        self.train_loader_gray = train_loader_gray
        self.eval_loader_gray = eval_loader_gray
        self.test_loader_gray = test_loader_gray
        self.test_dataset_gray = test_dataset_gray


        #GAN
        # self.generator = UNetGenerator().to(self.device)
        # self.discriminator = Discriminator().to(self.device)
        self.GANcriterion = nn.BCEWithLogitsLoss().to(self.device)
        self.MSEcriterion = nn.MSELoss()
        # self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        # self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0.5, 0.999))

        #WGAN
        self.generator = Wgan.UNetGenerator()
        self.discriminator = Wgan.Discriminator()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.discriminator_iter = 4


    def pretrain_generator(self):
        if os.path.exists('saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/pretrained_model.pth'))

        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        accuracy = []
        g_loss_per_epoch = []
        for epoch in range(self.num_epochs_pre):
            psnr_values = []
            g_loss_per_batch = []
            for batch_idx, ((gray_images, _), (rgb_images, _)) in enumerate(
                    zip(self.train_loader_gray, self.train_loader_rgb)):
                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                gen_images = self.generator(gray_images)
                loss = self.MSEcriterion(gen_images, rgb_images)
                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()

                # Save the generated images
                os.makedirs("generated_images", exist_ok=True)
                first_image_gen = prepare_to_save_image(gen_images[0])
                first_image_grey = prepare_to_save_image(gray_images[0])
                first_image_rbg = prepare_to_save_image(rgb_images[0])
                plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
                plt.savefig(f"generated_images/gen_image_{epoch}_{batch_idx}.jpg")
                plt.close()
                g_loss_per_batch.append(loss)
                # compute PSNR
                psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images, rgb_images)])
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                    % (epoch, self.num_epochs, batch_idx, len(self.train_loader_gray), loss.item())
                )

            # Compute PSNR
            avg_psnr = sum(psnr_values) / len(psnr_values)
            accuracy.append(avg_psnr)  # Append the average PSNR value to the accuracy list
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))

            print(
                "[Epoch: %d/%d] [g_loss_train: %f] [PSNR: %.2f dB]"
                % (
                    epoch, self.num_epochs, np.average([l.item() for l in g_loss_per_batch]), avg_psnr
                )
            )
            # print(f"Epoch {epoch + 1}/{self.num_epochs_pre}")
            epoch_minus_1 = EPOCHS - 1

            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(), 'saved_models/pretrained_model.pth')

    def train_generator(self, valid, fake_pred, rgb_images, gen_images):
        self.optimizer_G.zero_grad()
        # Loss measures generator's ability to fool the discriminator
        rgb_images = rgb_images.to(self.device)
        gen_images = gen_images.to(self.device)
        fake_pred_2d = fake_pred[:, :, 0, 0]
        g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
        g_loss_rgb = self.MSEcriterion(gen_images, rgb_images)
        g_loss = g_loss_pred + g_loss_rgb
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss

    def train_discriminator(self, rgb_images, gen_images, valid, fake):
        # Train Discriminator
        self.optimizer_D.zero_grad()

        # to device (cuda)
        rgb_images = rgb_images.to(self.device)
        gen_images = gen_images.to(self.device)
        valid = valid.to(self.device)
        fake = fake.to(self.device)

        # Measure discriminator's ability to classify real and fake images
        real_preds = self.discriminator(rgb_images)
        real_preds_2d = real_preds[:, :, 0, 0]
        real_loss = self.GANcriterion(real_preds_2d, valid)

        fake_preds = self.discriminator(gen_images.detach())
        fake_preds_2d = fake_preds[:, :, 0, 0]
        fake_loss = self.GANcriterion(fake_preds_2d, fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        self.optimizer_D.step()

        return d_loss

    def train(self):
        if os.path.exists('saved_models/generator_model.pth') and os.path.exists('saved_models/discriminator_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/generator_model.pth'))
            self.discriminator.load_state_dict(torch.load('saved_models/discriminator_model.pth'))
            print("Finished loading the previous trained models!")

        elif os.path.exists('saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/pretrained_model.pth'))
            print("Finished loading the pretrained generator!")

        test_losses_g = []
        val_losses_g = []
        d_loss_per_epoch = []
        g_loss_per_epoch = []
        self.generator.train()
        self.discriminator.train()
        accuracy = []

        # Training loop
        for epoch in range(self.num_epochs):
            g_loss_per_batch = []
            d_loss_per_batch = []
            psnr_values = []
            for batch_idx, ((gray_images, _), (rgb_images, _)) in enumerate(zip(self.train_loader_gray, self.train_loader_rgb)):
                # Adversarial ground truths
                valid = torch.ones(gray_images.size(0), 1).to(self.device)
                fake = torch.zeros(gray_images.size(0), 1).to(self.device)

                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                # Generate RGB images from grayscale
                gen_images = self.generator(gray_images)
                fake_pred = self.discriminator(gen_images)

                # Train Generator
                g_loss = self.train_generator(valid, fake_pred, rgb_images, gen_images)
                g_loss_per_batch.append(g_loss)
                # Train Discriminator
                d_loss = self.train_discriminator(rgb_images, gen_images, valid, fake)
                d_loss_per_batch.append(d_loss)

                # compute PSNR
                psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images, rgb_images)])

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.num_epochs, batch_idx, len(self.train_loader_gray), d_loss.item(), g_loss.item())
                )

                # Save the generated images
                os.makedirs("images_per_epoch", exist_ok=True)
                first_image_gen = prepare_to_save_image(gen_images[0])
                first_image_grey = prepare_to_save_image(gray_images[0])
                first_image_rbg = prepare_to_save_image(rgb_images[0])
                plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
                plt.savefig(f"images_per_epoch/image_epoch_{epoch}_batch_{batch_idx}.jpg")
                plt.close()

            test_loss_per_epoch = self.data_avg_loss(self.test_loader_gray, self.test_loader_rgb)
            validation_loss_per_epoch = self.data_avg_loss(self.eval_loader_gray, self.eval_loader_rgb)

            # Update losses arrays
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))
            d_loss_per_epoch.append(np.average([l.item() for l in d_loss_per_batch]))
            test_losses_g.append(test_loss_per_epoch)
            val_losses_g.append(validation_loss_per_epoch)

            # Compute PSNR
            avg_psnr = sum(psnr_values) / len(psnr_values)
            accuracy.append(avg_psnr)  # Append the average PSNR value to the accuracy list

            print(
                "[Epoch: %d/%d] [g_loss_train: %f] [d_loss_train: %f] [test_loss_per_epoch: %f] ["
                "validation_loss_per_epoch: %f] [PSNR: %.2f dB]"
                % (
                    epoch, self.num_epochs, g_loss_per_epoch[-1],
                    d_loss_per_epoch[-1],
                    test_loss_per_epoch, validation_loss_per_epoch, avg_psnr
                )
            )
            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(), 'saved_models/generator_model.pth')
            torch.save(self.discriminator.state_dict(), 'saved_models/discriminator_model.pth')

        return g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g, accuracy

    def data_avg_loss(self, loader_gray, loader_rgb):
        test_loss = []
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            # Test loop
            for (gray_images, _), (rgb_images, _) in zip(loader_gray, loader_rgb, ):
                # Adversarial ground truths
                valid = torch.ones(gray_images.size(0), 1).to(self.device)
                fake = torch.zeros(gray_images.size(0), 1).to(self.device)

                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                # Generate RGB images from grayscale
                gen_images = self.generator(gray_images)
                fake_pred = self.discriminator(gen_images)
                fake_pred_2d = fake_pred[:, :, 0, 0]
                g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
                g_loss_rgb = self.MSEcriterion(gen_images, rgb_images)
                g_loss = g_loss_pred + g_loss_rgb
                test_loss.append(g_loss)
        return sum(test_loss) / len(test_loss)

    def results_visualization(self):
        for (gray_images, _), (rgb_images, _) in zip(self.test_loader_gray, self.test_loader_rgb):
            # Configure input
            gray_images = gray_images.to(self.device)
            rgb_images = rgb_images.to(self.device)

            # Generate RGB images from grayscale
            gen_images = self.generator(gray_images)

            for idx, (gray_image, rgb_image, gen_image) in enumerate(zip(gray_images, rgb_images, gen_images)):
                gray_image_np = prepare_to_save_image(gray_image)
                gen_image_np = prepare_to_save_image(gen_image)
                rgb_image_np = prepare_to_save_image(rgb_image)
                plt = make_subplot(rgb_image_np, gray_image_np, gen_image_np)

                plt.savefig("results/%d.jpg" % idx)
                plt.close()

    def train_generator_wgan(self, fake_pred):

        self.optimizer_G.zero_grad()
        g_loss = -fake_pred.mean()
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss

    def train_discriminator_wgan(self, rgb_images, gen_images):

        # to device (cuda)
        rgb_images = rgb_images.to(self.device)
        gen_images = gen_images.to(self.device)

        # Train Discriminator
        self.optimizer_D.zero_grad()
        real_preds = self.discriminator(rgb_images)
        real_loss = real_preds.mean()
        fake_preds = self.discriminator(gen_images.detach())
        fake_loss = fake_preds.mean()

        d_loss = real_loss - fake_loss
        d_loss.backward()
        self.optimizer_D.step()

        return d_loss

    def train_wgan(self):

        if os.path.exists('saved_models/generator_model.pth') and os.path.exists('saved_models/discriminator_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/generator_model.pth'))
            self.discriminator.load_state_dict(torch.load('saved_models/discriminator_model.pth'))
            print("Finished loading the previous trained models!")

        elif os.path.exists('saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/pretrained_model.pth'))
            print("Finished loading the pretrained generator!")

        test_losses_g = []
        val_losses_g = []
        d_loss_per_epoch = []
        g_loss_per_epoch = []
        self.generator.train()
        self.discriminator.train()
        accuracy = []
        count_train_batchs = 0

        for epoch in range(self.num_epochs):
            g_loss_per_batch = []
            d_loss_per_batch = []
            psnr_values = []
            for batch_idx, ((gray_images, _), (rgb_images, _)) in enumerate(zip(self.train_loader_gray, self.train_loader_rgb)):

                # Print Epoch info
                print("[Epoch %d/%d] [Batch %d/%d] " %
                      (epoch, self.num_epochs, batch_idx, len(self.train_loader_gray)))

                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                if count_train_batchs == 0:
                    count_train_batchs = self.discriminator_iter

                # Train discriminator
                if count_train_batchs > 0:

                    # Generate rgb images for discriminator training
                    gen_images = self.generator(gray_images)

                    # Unfreeze discriminator weights
                    for param in self.discriminator.parameters():
                        param.requires_grad = True

                    # Train discriminator
                    d_loss = self.train_discriminator_wgan(rgb_images, gen_images)
                    count_train_batchs -= 1

                # After 4 batch that the discriminator has train - generator will train
                if count_train_batchs == 0:

                    # Freeze discriminator weights during generator training
                    for param in self.discriminator.parameters():
                        param.requires_grad = False

                    # Generate RGB images from grayscale
                    gen_images = self.generator(gray_images)
                    fake_pred = self.discriminator(gen_images)
                    # Train generator
                    g_loss = self.train_generator_wgan(fake_pred)

                    # Save discriminator and generator loss per batch
                    g_loss_per_batch.append(g_loss.item())
                    d_loss_per_batch.append(d_loss.item())

                    # Print info
                    print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]" %
                          (epoch, self.num_epochs, batch_idx, len(self.train_loader_gray), g_loss.item(), d_loss.item()))

                    # compute PSNR
                    psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images, rgb_images)])

                    # Save the generated images
                    os.makedirs("images_per_epoch", exist_ok=True)
                    first_image_gen = prepare_to_save_image(gen_images[0])
                    first_image_grey = prepare_to_save_image(gray_images[0])
                    first_image_rbg = prepare_to_save_image(rgb_images[0])
                    plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
                    plt.savefig(f"images_per_epoch/image_epoch_{epoch}_batch_{batch_idx}.jpg")
                    plt.close()

            # Calc test and validation loss
            test_loss_per_epoch = self.data_avg_loss(self.test_loader_gray, self.test_loader_rgb)
            validation_loss_per_epoch = self.data_avg_loss(self.eval_loader_gray, self.eval_loader_rgb)

            # Update losses arrays
            g_loss_per_epoch.append(sum(g_loss_per_batch) / len(g_loss_per_batch))
            d_loss_per_epoch.append(sum(d_loss_per_batch) / len(d_loss_per_batch))
            test_losses_g.append(test_loss_per_epoch)
            val_losses_g.append(validation_loss_per_epoch)

            # Compute PSNR
            accuracy.append(sum(psnr_values) / len(psnr_values))  # Append the average PSNR value to the accuracy list

            # Print Epoch info
            print(
                "[Epoch: %d/%d] [g_loss_train: %f] [d_loss_train: %f] [test_loss_per_epoch: %f] ["
                "validation_loss_per_epoch: %f] [PSNR: %.2f dB]"
                % (
                    epoch, self.num_epochs, g_loss_per_epoch[-1],
                    d_loss_per_epoch[-1],
                    test_loss_per_epoch, validation_loss_per_epoch, (sum(psnr_values) / len(psnr_values))
                )
            )

            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(), 'saved_models/generator_model.pth')
            torch.save(self.discriminator.state_dict(), 'saved_models/discriminator_model.pth')

        return g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g, accuracy


