
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from Gan import UNetGenerator, Discriminator

BATCH_SIZE = 10
EPOCHS = 5


class ModelHandler:
    def __init__(self, test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb, train_loader_gray,
                 eval_loader_gray,
                 test_loader_gray, batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=0.0002, lr_D=0.0002,
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
        self.GANcriterion = nn.BCEWithLogitsLoss().to(self.device)
        self.L1criterion = nn.L1Loss()
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0.5, 0.999))

    import os

    def pretrain_generator(self):
        save_dir = 'saved_models_ImageColoringProject'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        for epoch in range(self.num_epochs_pre):
            for idx, ((gray_images, _), (rgb_images, _)) in enumerate(
                    zip(self.train_loader_gray, self.train_loader_rgb)):
                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                gen_images = self.generator(gray_images)
                loss = self.L1criterion(gen_images, rgb_images)
                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()

                # Save the generated images
                os.makedirs("generated_images", exist_ok=True)
                save_image(gen_images.cpu(), f"generated_images/gen_image_{epoch}_{idx}.png", normalize=True)

            print(f"Epoch {epoch + 1}/{self.num_epochs_pre}")
            epoch_minus_1 = EPOCHS - 1

            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(),
                       os.path.join(save_dir, f'gen_model_{epoch_minus_1}.pth'))

    def train_generator(self, valid, fake_pred, rgb_images, gen_images):
        # Loss measures generator's ability to fool the discriminator
        rgb_images = rgb_images.to(self.device)
        gen_images = gen_images.to(self.device)
        fake_pred_2d = fake_pred[:, :, 0, 0]
        g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
        g_loss_rgb = self.L1criterion(gen_images, rgb_images)
        g_loss = g_loss_pred + g_loss_rgb
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss

    def train_discriminator(self, rgb_images, gen_images, valid, fake):
        # Train Discriminator
        self.optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real and fake images
        rgb_images = rgb_images.to(self.device)
        gen_images = gen_images.to(self.device)
        valid = valid.to(self.device)
        fake = fake.to(self.device)
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
        # epoch_minus_1 = EPOCHS - 1
        #
        # self.generator.load_state_dict(torch.load(
        #     'saved_models_ImageColoringProject/gen_model_{}.pth'.format(
        #         epoch_minus_1)))

        test_losses_g = []
        val_losses_g = []
        d_loss_per_epoch = []
        g_loss_per_epoch = []
        self.generator.train()
        self.discriminator.train()

        # Training loop
        for epoch in range(self.num_epochs):
            i = 0
            g_loss_per_batch = []
            d_loss_per_batch = []
            for (gray_images, _), (rgb_images, _) in zip(self.train_loader_gray, self.train_loader_rgb):
                # Adversarial ground truths
                valid = torch.ones(gray_images.size(0), 1).to(self.device)
                fake = torch.zeros(gray_images.size(0), 1).to(self.device)

                # Configure input
                gray_images = gray_images.to(self.device)
                rgb_images = rgb_images.to(self.device)

                # Generate RGB images from grayscale
                self.optimizer_G.zero_grad()
                gen_images = self.generator(gray_images)
                fake_pred = self.discriminator(gen_images)

                # Train Generator
                g_loss = self.train_generator(valid, fake_pred, rgb_images, gen_images)
                g_loss_per_batch.append(g_loss)
                # Train Discriminator
                d_loss = self.train_discriminator(rgb_images, gen_images, valid, fake)
                d_loss_per_batch.append(d_loss)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.num_epochs, i, len(self.train_loader_gray), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.train_loader_gray) + i
                if batches_done % 25 == 0:
                    # show_images(gen_images[:5])
                    save_image(gen_images.data[:25],
                               "images_per_epoch/%d.jpg" % batches_done,
                               nrow=5, normalize=True)

                i += 1

            test_loss_per_epoch = self.data_avg_loss(self.test_loader_gray, self.test_loader_rgb)
            validation_loss_per_epoch = self.data_avg_loss(self.eval_loader_gray, self.eval_loader_rgb)

            # Update losses arrays
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))
            d_loss_per_epoch.append(np.average([l.item() for l in d_loss_per_batch]))
            test_losses_g.append(test_loss_per_epoch)
            val_losses_g.append(validation_loss_per_epoch)
            print(
                "[Epoch: %d/%d] [g_loss_train: %f] [d_loss_train: %f] [test_loss_per_epoch: %f] ["
                "validation_loss_per_epoch: %f]"
                % (
                    epoch, self.num_epochs, np.average([l.item() for l in g_loss_per_batch]),
                    np.average([l.item() for l in d_loss_per_batch]),
                    test_loss_per_epoch,
                    validation_loss_per_epoch)
            )

        return g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g

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
                g_loss_rgb = self.L1criterion(gen_images, rgb_images)
                g_loss = g_loss_pred + g_loss_rgb
                test_loss.append(g_loss)
        return sum(test_loss) / len(test_loss)

    def results_visualization(self):
        i = 0
        for (gray_images, _), (rgb_images, _) in zip(self.test_loader_gray, self.test_loader_rgb):
            # Configure input
            gray_images = gray_images.to(self.device)
            rgb_images = rgb_images.to(self.device)

            # Generate RGB images from grayscale
            gen_images = self.generator(gray_images)

            for gray_img, rgb_img, gen_img in zip(gray_images, rgb_images, gen_images):
                # Convert images to numpy arrays
                gray_img_np = gray_img.permute(1, 2, 0).detach().cpu().numpy()
                gen_img_np = gen_img.permute(1, 2, 0).detach().cpu().numpy()
                rgb_img_np = rgb_img.permute(1, 2, 0).detach().cpu().numpy()

                # Normalize pixel values to [0, 1]
                gray_img_np = (gray_img_np - gray_img_np.min()) / (gray_img_np.max() - gray_img_np.min())
                gen_img_np = (gen_img_np - gen_img_np.min()) / (gen_img_np.max() - gen_img_np.min())
                rgb_img_np = (rgb_img_np - rgb_img_np.min()) / (rgb_img_np.max() - rgb_img_np.min())

                # Plot images
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(rgb_img_np)
                axs[0].axis('off')

                axs[1].imshow(gray_img_np.squeeze(), cmap='gray')
                axs[1].axis('off')

                axs[2].imshow(gen_img_np)
                axs[2].axis('off')

                plt.tight_layout()
                plt.savefig("results/%d.jpg" % i)
                plt.show()

                i += 1


# Define a function to display images
def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, img in enumerate(images):
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axs[i].imshow(img_np)
        axs[i].axis('off')
    plt.show()


# import os
# import numpy as np
# import torch
# from matplotlib import pyplot as plt
# from torch import nn, optim
# from torchvision.utils import save_image
# from Gan import UNetGenerator, Discriminator
# import torch
#
#
# class ModelHandler:
#     def __init__(self, test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb, train_loader_gray,
#                  eval_loader_gray,
#                  test_loader_gray, batch_size=128, num_epochs=5, lr_G=0.0002, lr_D=0.0002, num_epochs_pre=5):
#         self.num_epochs = num_epochs
#         self.num_epochs_pre = num_epochs_pre
#         self.lr_G = lr_G
#         self.lr_D = lr_D
#         self.batch_size = batch_size
#         self.train_loader_rgb = train_loader_rgb
#         self.eval_loader_rgb = eval_loader_rgb
#         self.test_loader_rgb = test_loader_rgb
#         self.train_loader_gray = train_loader_gray
#         self.eval_loader_gray = eval_loader_gray
#         self.test_loader_gray = test_loader_gray
#         self.test_dataset_gray = test_dataset_gray
#         self.GANcriterion = nn.BCEWithLogitsLoss()
#         self.L1criterion = nn.L1Loss()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.generator = UNetGenerator().to(self.device)
#         self.discriminator = Discriminator().to(self.device)
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0.5, 0.999))
#
#     def pretrain_generator(self):
#
#         for epoch in range(self.num_epochs_pre):
#             for idx, ((gray_images, _), (rgb_images, _)) in enumerate(
#                     zip(self.train_loader_gray, self.train_loader_rgb)):
#                 gen_images = self.generator(gray_images)
#                 loss = self.L1criterion(gen_images, rgb_images)
#                 self.optimizer_G.zero_grad()
#                 loss.backward()
#                 self.optimizer_G.step()
#
#                 # Save the generated images
#                 os.makedirs("generated_images", exist_ok=True)
#                 save_image(gen_images, f"generated_images/gen_image_{epoch}_{idx}.png", normalize=True)
#
#             print(f"Epoch {epoch + 1}/{self.num_epochs_pre}")
#             # print(f"L1 Loss: {loss.item():.5f}")
#
#             # Save the generator model after all epochs (changed to after every epoch) - need to run again
#             torch.save(self.generator.state_dict(),
#                        r'C:\Users\noycoh\Documents\GitHub\ImageColoringProject\saved_models_ImageColoringProject/')
#
#     def train_generator(self, valid, fake_pred, rgb_images, gen_images):
#         # Loss measures generator's ability to fool the discriminator
#         fake_pred_2d = fake_pred[:, :, 0, 0]
#         g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
#         g_loss_rgb = self.L1criterion(gen_images, rgb_images)
#         g_loss = g_loss_pred + g_loss_rgb
#         g_loss.backward()
#         self.optimizer_G.step()
#
#         return g_loss
#
#     def train_discriminator(self, rgb_images, gen_images, valid, fake):
#         # Train Discriminator
#         self.optimizer_D.zero_grad()
#         # Measure discriminator's ability to classify real and fake images
#         real_preds = self.discriminator(rgb_images)
#         real_preds_2d = real_preds[:, :, 0, 0]
#         real_loss = self.GANcriterion(real_preds_2d, valid)
#
#         fake_preds = self.discriminator(gen_images.detach())
#         fake_preds_2d = fake_preds[:, :, 0, 0]
#         fake_loss = self.GANcriterion(fake_preds_2d, fake)
#         d_loss = 0.5 * (real_loss + fake_loss)
#
#         d_loss.backward()
#         self.optimizer_D.step()
#
#         return d_loss
#
#     def train(self):
#         self.generator.load_state_dict(torch.load(
#             r'C:\Users\noycoh\Documents\GitHub\ImageColoringProject\saved_models_ImageColoringProject/'))
#
#         test_losses_g = []
#         val_losses_g = []
#         d_loss_per_epoch = []
#         g_loss_per_epoch = []
#         self.generator.train()
#         self.discriminator.train()
#
#         # Training loop
#         for epoch in range(self.num_epochs):
#             i = 0
#             g_loss_per_batch = []
#             d_loss_per_batch = []
#             for (gray_images, _), (rgb_images, _) in zip(self.train_loader_gray, self.train_loader_rgb):
#                 # Adversarial ground truths
#                 valid = torch.ones(gray_images.size(0), 1).to(self.device)
#                 fake = torch.zeros(gray_images.size(0), 1).to(self.device)
#
#                 # Configure input
#                 gray_images = gray_images.to(self.device)
#
#                 # Generate RGB images from grayscale
#                 self.optimizer_G.zero_grad()
#                 gen_images = self.generator(gray_images)
#                 fake_pred = self.discriminator(gen_images)
#
#                 # Train Generator
#                 g_loss = self.train_generator(valid, fake_pred, rgb_images, gen_images)
#                 g_loss_per_batch.append(g_loss)
#                 # Train Discriminator
#                 d_loss = self.train_discriminator(rgb_images, gen_images, valid, fake)
#                 d_loss_per_batch.append(d_loss)
#
#                 print(
#                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                     % (epoch, self.num_epochs, i, len(self.train_loader_gray), d_loss.item(), g_loss.item())
#                 )
#
#                 batches_done = epoch * len(self.train_loader_gray) + i
#                 if batches_done % 25 == 0:
#                     # show_images(gen_images[:5])
#                     save_image(gen_images.data[:25],
#                                r"C:\Users\noycoh\Documents\GitHub\ImageColoringProject\images_per_epoch/%d.jpg" % batches_done,
#                                nrow=5, normalize=True)
#
#                 i += 1
#
#             test_loss_per_epoch = self.data_avg_loss(self.test_loader_gray, self.test_loader_rgb)
#             validation_loss_per_epoch = self.data_avg_loss(self.eval_loader_gray, self.eval_loader_rgb)
#
#             # Update losses arrays
#             g_loss_per_epoch.append(np.average(g_loss_per_batch))
#             d_loss_per_epoch.append(np.average(d_loss_per_batch))
#             test_losses_g.append(test_loss_per_epoch)
#             val_losses_g.append(validation_loss_per_epoch)
#             print(
#                 "[Epoch: %d/%d] [g_loss_train: %f] [d_loss_train: %f] [test_loss_per_epoch: %f] ["
#                 "validation_loss_per_epoch: %f]"
#                 % (
#                     epoch, self.num_epochs, np.average(g_loss_per_batch), np.average(d_loss_per_batch),
#                     test_loss_per_epoch,
#                     validation_loss_per_epoch)
#             )
#
#         return g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g
#
#     def data_avg_loss(self, loader_gray, loader_rgb):
#         test_loss = []
#         self.generator.eval()
#         self.discriminator.eval()
#         with torch.no_grad():
#             # Test loop
#             for (gray_images, _), (rgb_images, _) in zip(loader_gray, loader_rgb, ):
#                 # Adversarial ground truths
#                 valid = torch.ones(gray_images.size(0), 1).to(self.device)
#                 fake = torch.zeros(gray_images.size(0), 1).to(self.device)
#
#                 # Configure input
#                 gray_images = gray_images.to(self.device)
#
#                 # Generate RGB images from grayscale
#                 gen_images = self.generator(gray_images)
#                 fake_pred = self.discriminator(gen_images)
#                 fake_pred_2d = fake_pred[:, :, 0, 0]
#                 g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
#                 g_loss_rgb = self.L1criterion(gen_images, rgb_images)
#                 g_loss = g_loss_pred + g_loss_rgb
#                 test_loss.append(g_loss)
#         return sum(test_loss) / len(test_loss)
#
#     def results_visualization(self):
#         i = 0
#         for (gray_images, _), (rgb_images, _) in zip(self.test_loader_gray, self.test_loader_rgb):
#             # Configure input
#             gray_images = gray_images.to(self.device)
#
#             # Generate RGB images from grayscale
#             gen_images = self.generator(gray_images)
#
#             for gray_img, rgb_img, gen_img in zip(gray_images, rgb_images, gen_images):
#                 # Convert images to numpy arrays
#                 gray_img_np = gray_img.permute(1, 2, 0).detach().cpu().numpy()
#                 gen_img_np = gen_img.permute(1, 2, 0).detach().cpu().numpy()
#                 rgb_img_np = rgb_img.permute(1, 2, 0).detach().cpu().numpy()
#
#                 # Normalize pixel values to [0, 1]
#                 gray_img_np = (gray_img_np - gray_img_np.min()) / (gray_img_np.max() - gray_img_np.min())
#                 gen_img_np = (gen_img_np - gen_img_np.min()) / (gen_img_np.max() - gen_img_np.min())
#                 rgb_img_np = (rgb_img_np - rgb_img_np.min()) / (rgb_img_np.max() - rgb_img_np.min())
#
#                 # Plot images
#                 fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
#                 axs[0].imshow(rgb_img_np)
#                 axs[0].axis('off')
#
#                 axs[1].imshow(gray_img_np.squeeze(), cmap='gray')
#                 axs[1].axis('off')
#
#                 axs[2].imshow(gen_img_np)
#                 axs[2].axis('off')
#
#                 plt.tight_layout()
#                 plt.savefig(r"C:\Users\noycoh\Documents\GitHub\ImageColoringProject\results/%d.jpg" % i)
#                 plt.show()
#
#                 i += 1
#
#     # Define a function to display images
#     def show_images(images):
#         fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
#         for i, img in enumerate(images):
#             img_np = img.permute(1, 2, 0).detach().cpu().numpy()
#             img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#             axs[i].imshow(img_np)
#             axs[i].axis('off')
#         plt.show()
#
#     def train_generator(self, valid, fake_pred, rgb_images, gen_images):
#         # Loss measures generator's ability to fool the discriminator
#         fake_pred_2d = fake_pred[:, :, 0, 0]
#         g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
#         g_loss_rgb = self.L1criterion(gen_images, rgb_images)
#         g_loss = g_loss_pred + g_loss_rgb
#         g_loss.backward()
#         self.optimizer_G.step()
#
#         return g_loss
#
#     def train_discriminator(self, rgb_images, gen_images, valid, fake):
#         # Train Discriminator
#         self.optimizer_D.zero_grad()
#         # Measure discriminator's ability to classify real and fake images
#         real_preds = self.discriminator(rgb_images)
#         real_preds_2d = real_preds[:, :, 0, 0]
#         real_loss = self.GANcriterion(real_preds_2d, valid)
#
#         fake_preds = self.discriminator(gen_images.detach())
#         fake_preds_2d = fake_preds[:, :, 0, 0]
#         fake_loss = self.GANcriterion(fake_preds_2d, fake)
#         d_loss = 0.5 * (real_loss + fake_loss)
#
#         d_loss.backward()
#         self.optimizer_D.step()
#
#         return d_loss
#
#     def train(self):
#         # self.generator.load_state_dict(torch.load(
#         #     '/content/gdrive/MyDrive/Colab Notebooks/saved_models_ImageColoringProject/preTrainingGenerator'))
#
#         test_losses_g = []
#         val_losses_g = []
#         d_loss_per_epoch = []
#         g_loss_per_epoch = []
#         self.generator.train()
#         self.discriminator.train()
#
#         # Training loop
#         for epoch in range(self.num_epochs):
#             i = 0
#             g_loss_per_batch = []
#             d_loss_per_batch = []
#             for (gray_images, _), (rgb_images, _) in zip(self.train_loader_gray, self.train_loader_rgb):
#                 # Adversarial ground truths
#                 valid = torch.ones(gray_images.size(0), 1).to(self.device)
#                 fake = torch.zeros(gray_images.size(0), 1).to(self.device)
#
#                 # Configure input
#                 gray_images = gray_images.to(self.device)
#
#                 # Generate RGB images from grayscale
#                 self.optimizer_G.zero_grad()
#                 gen_images = self.generator(gray_images)
#                 fake_pred = self.discriminator(gen_images)
#
#                 # Train Generator
#                 g_loss = self.train_generator(valid, fake_pred, rgb_images, gen_images)
#                 g_loss_per_batch.append(g_loss)
#                 # Train Discriminator
#                 d_loss = self.train_discriminator(rgb_images, gen_images, valid, fake)
#                 d_loss_per_batch.append(d_loss)
#
#                 print(
#                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                     % (epoch, self.num_epochs, i, len(self.train_loader_gray), d_loss.item(), g_loss.item())
#                 )
#
#                 batches_done = epoch * len(self.train_loader_gray) + i
#                 if batches_done % 25 == 0:
#                     # show_images(gen_images[:5])
#                     save_image(gen_images.data[:25], "images/%d.jpg" % batches_done, nrow=5, normalize=True)
#
#                 i += 1
#
#             test_loss_per_epoch = self.data_avg_loss(self.test_loader_gray, self.test_loader_rgb)
#             validation_loss_per_epoch = self.data_avg_loss(self.eval_loader_gray, self.eval_loader_rgb)
#
#             # Update losses arrays
#             g_loss_per_epoch.append(np.average(g_loss_per_batch))
#             d_loss_per_epoch.append(np.average(d_loss_per_batch))
#             test_losses_g.append(test_loss_per_epoch)
#             val_losses_g.append(validation_loss_per_epoch)
#             print(
#                 "[Epoch: %d/%d] [g_loss_train: %f] [d_loss_train: %f] [test_loss_per_epoch: %f] ["
#                 "validation_loss_per_epoch: %f]"
#                 % (
#                     epoch, self.num_epochs, np.average(g_loss_per_batch), np.average(d_loss_per_batch),
#                     test_loss_per_epoch,
#                     validation_loss_per_epoch)
#             )
#
#         return g_loss_per_epoch, d_loss_per_epoch, test_losses_g, val_losses_g
#
#     def data_avg_loss(self, loader_gray, loader_rgb):
#         test_loss = []
#         self.generator.eval()
#         self.discriminator.eval()
#         with torch.no_grad():
#             # Test loop
#             for (gray_images, _), (rgb_images, _) in zip(loader_gray, loader_rgb, ):
#                 # Adversarial ground truths
#                 valid = torch.ones(gray_images.size(0), 1).to(self.device)
#                 fake = torch.zeros(gray_images.size(0), 1).to(self.device)
#
#                 # Configure input
#                 gray_images = gray_images.to(self.device)
#
#                 # Generate RGB images from grayscale
#                 gen_images = self.generator(gray_images)
#                 fake_pred = self.discriminator(gen_images)
#                 fake_pred_2d = fake_pred[:, :, 0, 0]
#                 g_loss_pred = self.GANcriterion(fake_pred_2d, valid)
#                 g_loss_rgb = self.L1criterion(gen_images, rgb_images)
#                 g_loss = g_loss_pred + g_loss_rgb
#                 test_loss.append(g_loss)
#         return sum(test_loss) / len(test_loss)
#
#
#     def results_visualization(self):
#         i = 0
#         for (gray_images, _), (rgb_images, _) in zip(self.test_loader_gray, self.test_loader_rgb):
#             # Configure input
#             gray_images = gray_images.to(self.device)
#
#             # Generate RGB images from grayscale
#             gen_images = self.generator(gray_images)
#
#             for gray_img, rgb_img, gen_img in zip(gray_images, rgb_images, gen_images):
#                 # Convert images to numpy arrays
#                 gray_img_np = gray_img.permute(1, 2, 0).detach().cpu().numpy()
#                 gen_img_np = gen_img.permute(1, 2, 0).detach().cpu().numpy()
#                 rgb_img_np = rgb_img.permute(1, 2, 0).detach().cpu().numpy()
#
#                 # Normalize pixel values to [0, 1]
#                 gray_img_np = (gray_img_np - gray_img_np.min()) / (gray_img_np.max() - gray_img_np.min())
#                 gen_img_np = (gen_img_np - gen_img_np.min()) / (gen_img_np.max() - gen_img_np.min())
#                 rgb_img_np = (rgb_img_np - rgb_img_np.min()) / (rgb_img_np.max() - rgb_img_np.min())
#
#                 # Plot images
#                 fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
#                 axs[0].imshow(rgb_img_np)
#                 axs[0].axis('off')
#
#                 axs[1].imshow(gray_img_np.squeeze(), cmap='gray')
#                 axs[1].axis('off')
#
#                 axs[2].imshow(gen_img_np)
#                 axs[2].axis('off')
#
#                 plt.tight_layout()
#                 plt.savefig("results/%d.jpg" % i)
#                 plt.show()
#
#                 i += 1
#
#
# # Define a function to display images
# def show_images(images):
#     fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
#     for i, img in enumerate(images):
#         img_np = img.permute(1, 2, 0).detach().cpu().numpy()
#         img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#         axs[i].imshow(img_np)
#         axs[i].axis('off')
#     plt.show()
#
