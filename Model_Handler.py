import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, autograd
from Gan import UNetGenerator, Critic
import numpy as np
import random
from itertools import islice

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.0001


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
    return psnr_val.detach()


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
    def __init__(self, test_dataset_gray, test_loader_rgb, train_loader_rgb, eval_loader_rgb, train_loader_gray, eval_loader_gray, test_loader_gray,
                 batch_size=BATCH_SIZE, num_epochs=EPOCHS, lr_G=LR, lr_C=LR, num_epochs_pre=EPOCHS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.num_epochs_pre = num_epochs_pre
        self.lr_G = lr_G
        self.lr_C = lr_C
        self.batch_size = batch_size
        self.train_loader_rgb = train_loader_rgb
        self.eval_loader_rgb = eval_loader_rgb
        self.test_loader_rgb = test_loader_rgb
        self.train_loader_gray = train_loader_gray
        self.eval_loader_gray = eval_loader_gray
        self.test_loader_gray = test_loader_gray
        self.test_dataset_gray = test_dataset_gray
        self.MSEcriterion = nn.MSELoss()

        #WGAN
        self.generator = UNetGenerator()
        self.Critic = Critic()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0, 0.9))
        self.optimizer_C = optim.Adam(self.Critic.parameters(), lr=self.lr_C, betas=(0, 0.9))

    def pretrain_generator(self):
        if os.path.exists('saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/pretrained_model.pth'))

        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        accuracy = []
        g_loss_per_epoch = []
        avg_psnr_per_epoch = []

        # # Load existing files
        # if os.path.exists('saved_models/accuracy.npy'):
        #     accuracy = list(np.load('saved_models/accuracy.npy'))
        # if os.path.exists('saved_models/g_loss_per_epoch.npy'):
        #     g_loss_per_epoch = list(np.load('saved_models/g_loss_per_epoch.npy'))
        # if os.path.exists('saved_models/avg_psnr_per_epoch.npy'):
        #     avg_psnr_per_epoch = list(np.load('saved_models/avg_psnr_per_epoch.npy'))

        for epoch in range(len(accuracy), self.num_epochs_pre):
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
                    % (epoch, self.num_epochs_pre, batch_idx, len(self.train_loader_gray), loss.item())
                )

            # Compute PSNR
            avg_psnr = sum(psnr_values) / len(psnr_values)
            avg_psnr_per_epoch.append(avg_psnr)
            accuracy.append(avg_psnr)  # Append the average PSNR value to the accuracy list
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))

            print(
                "[Epoch: %d/%d] [g_loss_train: %f] [PSNR: %.2f dB]"
                % (
                    epoch, self.num_epochs, np.average([l.item() for l in g_loss_per_batch]), avg_psnr
                )
            )

            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(), 'saved_models/pretrained_model.pth')

            # # Save avg_psnr, accuracy, and g_loss_per_epoch after every epoch
            # np.save('saved_models/avg_psnr_per_epoch.npy', np.array(avg_psnr_per_epoch))
            # np.save('saved_models/accuracy.npy', np.array(accuracy))
            # np.save('saved_models/g_loss_per_epoch.npy', np.array(g_loss_per_epoch))

    def test_model(self, loader_gray, loader_rgb):
        # Set model to eval mode (optional)
        self.generator.eval()
        self.Critic.eval()
        c_loss_per_batch = []
        psnr_values = []
        for batch_idx, ((gray_images, _), (rgb_images, _)) in enumerate(zip(loader_gray, loader_rgb)):
            # --- Configure input ---
            gray_images = gray_images.to(self.device)
            rgb_images = rgb_images.to(self.device)

            # Generate rgb images for Critic training
            gen_images = self.generator(gray_images)

            # Calculate Critic loss
            fake_preds = self.Critic(gen_images.detach())
            fake_loss = fake_preds.mean()
            c_loss = fake_loss  # Since you don't calculate real loss here

            # Save Critic loss per batch
            c_loss_per_batch.append(c_loss.item())

            # compute PSNR
            psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.detach(), rgb_images)])

        # Reset model to train mode (optional)
        self.generator.train()
        self.Critic.train()
        accuracy = sum(psnr_values) / len(psnr_values)
        loss = sum(c_loss_per_batch) / len(c_loss_per_batch)
        return accuracy, loss

    def results_visualization(self):
        counter = 0
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

                plt.savefig("results/image_%d.jpg" % counter)
                plt.close()
                counter += 1

    def gradient_penalty(self, real_images, fake_images):
        device = real_images.device
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
        alpha.expand_as(real_images)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        pred = self.Critic(interpolated)
        gradients = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                        grad_outputs=torch.ones(pred.size(), device=device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self):
        if os.path.exists('saved_models/generator_model.pth') and os.path.exists(
                'saved_models/Critic_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/generator_model.pth'))
            self.Critic.load_state_dict(torch.load('saved_models/Critic_model.pth'))
            print("Finished loading the previous trained models!")

        elif os.path.exists('saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load('saved_models/pretrained_model.pth'))
            print("Finished loading the pretrained generator!")
        else:
            print("Starting to train without pretrained model!")

        c_loss_per_epoch = []
        g_loss_per_epoch = []
        accuracy = []
        test_losses_g = []
        val_losses_g = []

        # Load existing arrays
        if os.path.exists('saved_models/c_loss_per_epoch.npy'):
            c_loss_per_epoch = list(np.load('saved_models/c_loss_per_epoch.npy'))
        if os.path.exists('saved_models/g_loss_per_epoch.npy'):
            g_loss_per_epoch = list(np.load('saved_models/g_loss_per_epoch.npy'))
        if os.path.exists('saved_models/accuracy.npy'):
            accuracy = list(np.load('saved_models/accuracy.npy'))

        # configure to train mode
        self.generator.train()
        self.Critic.train()

        # Training loop
        for epoch in range(len(c_loss_per_epoch), self.num_epochs):
            psnr_values = []
            g_loss_per_batch = []
            c_loss_per_batch = []

            for batch_idx, ((gray_images, _), (rgb_images, _)) in enumerate(
                    zip(self.train_loader_gray, self.train_loader_rgb)):

                # Critic Train With 5 Random Batch
                for index_critic_train in range(5):
                    # Sample a random index
                    random_index = random.randint(0, len(self.train_loader_gray) - 1)

                    # Fetch the batch with the same index from both DataLoaders gray and rgb
                    random_gray_images = next(islice(self.train_loader_gray, random_index, None))[0]
                    random_rgb_images = next(islice(self.train_loader_rgb, random_index, None))[0]

                    # Generate RGB images from grayscale
                    gen_images = self.generator(random_gray_images)
                    self.optimizer_C.zero_grad()

                    loss_c = -torch.mean(self.Critic(random_rgb_images)) + torch.mean(self.Critic(gen_images.detach()))
                    gp = self.gradient_penalty(random_rgb_images, gen_images.detach())
                    loss_c += 10 * gp

                    loss_c.backward()
                    self.optimizer_C.step()
                    print("Critic Train Number %d Finish" %index_critic_train)

                # Freeze Critic weights during generator training
                for param in self.Critic.parameters():
                    param.requires_grad = False

                # Generator Train
                self.optimizer_G.zero_grad()
                gen_images = self.generator(gray_images)
                wgan_loss = -torch.mean(self.Critic(gen_images))
                mse_loss = self.MSEcriterion(rgb_images, gen_images)
                loss_g = wgan_loss * 0.15 + mse_loss * 0.85
                loss_g.backward()
                self.optimizer_G.step()

                # Unfreeze Critic weights
                for param in self.Critic.parameters():
                    param.requires_grad = True

                c_loss_per_batch.append(loss_c)
                g_loss_per_batch.append(loss_g)

                # compute PSNR
                psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.detach(), rgb_images)])

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Critic loss: %f] [G loss: %f] [PSNR accuracy: %f]"
                    % (epoch, self.num_epochs, batch_idx, len(self.train_loader_gray), loss_c.item(), loss_g.item(),
                       sum(psnr_values) / len(psnr_values)
                       )
                )

                # Save the generated images
                os.makedirs("images_per_epoch", exist_ok=True)
                first_image_gen = prepare_to_save_image(gen_images[0])
                first_image_grey = prepare_to_save_image(gray_images[0])
                first_image_rbg = prepare_to_save_image(rgb_images[0])
                plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
                plt.savefig(f"images_per_epoch/image_epoch_{epoch}batch{batch_idx}.jpg")
                plt.close()

            # Update losses arrays
            test_accuracy, test_loss = self.test_model(self.test_loader_gray, self.test_loader_rgb)
            test_losses_g.append(test_accuracy)
            val_accuracy, val_loss = self.test_model(self.test_loader_gray, self.test_loader_rgb)
            val_losses_g.append(val_loss)

            # Caluclate model accuracy
            # Compute PSNR
            avg_psnr = sum(psnr_values) / len(psnr_values)
            accuracy.append(avg_psnr)  # Append the average PSNR value to the accuracy list

            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))
            c_loss_per_epoch.append(np.average([l.item() for l in c_loss_per_batch]))

            # Save arrays after every epoch
            np.save('saved_models/c_loss_per_epoch.npy', np.array(c_loss_per_epoch))
            np.save('saved_models/g_loss_per_epoch.npy', np.array(g_loss_per_epoch))
            np.save('saved_models/accuracy.npy', np.array(accuracy))
            np.save('saved_models/test_losses_g.npy', np.array(test_losses_g))
            np.save('saved_models/val_losses_g.npy', np.array(val_losses_g))

            # Save the generator model after every epoch
            torch.save(self.generator.state_dict(), 'saved_models/generator_model.pth')
            torch.save(self.Critic.state_dict(), 'saved_models/Critic_model.pth')

        return g_loss_per_epoch, c_loss_per_epoch, test_losses_g, val_losses_g, accuracy