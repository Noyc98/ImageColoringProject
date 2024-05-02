import gc
from Gan import UNetGenerator, Critic
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
LR = 0.0001
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


# Computes the PSNR between the input and target images.
def psnr(input_image, target_image):
    mse = torch.mean((input_image - target_image) ** 2)
    psnr_val = 10 * torch.log10(1 / mse)
    return psnr_val.detach().numpy()


def compute_psnr(loader_gray, loader_rgb, model_handler):
    psnr_values = []
    with torch.no_grad():
        for (gray_images, _), (rgb_images, _) in zip(loader_gray, loader_rgb):
            gray_images = gray_images.to("cuda")
            rgb_images = rgb_images.to("cuda")
            gen_images = model_handler.generator(gray_images)
            psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.to("cpu"), rgb_images.to("cpu"))])
    avg_psnr = sum(psnr_values) / len(psnr_values)
    return avg_psnr

def convert_to_greyscale_batch(image_batch, grey_dir=None):
    # Ensure image_batch is a tensor
    if not torch.is_tensor(image_batch):
        image_batch = torch.tensor(image_batch)

    # Check if the input is in [B, C, H, W] format
    if len(image_batch.shape) != 4:
        raise ValueError("Input image_batch should be in [B, C, H, W] format.")

    # Convert the batch of colored images to greyscale by taking the mean along the channel dimension
    grey_batch = torch.mean(image_batch, dim=1, keepdim=True)  # Assuming image_batch is in [B, C, H, W] format

    return grey_batch

def average_every_n_epochs(data, n=5):
    num_epochs = len(data)
    num_batches = num_epochs // n
    averaged_data = []
    for i in range(num_batches):
        start = i * n
        end = (i + 1) * n
        averaged_data.append(np.mean(data[start:end]))
    return averaged_data


class ModelHandler:
    def __init__(self, train_loader_rgb, eval_loader_rgb, test_loader_rgb,
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
        self.MSEcriterion = nn.MSELoss()
        # WGAN
        self.generator = UNetGenerator().to(self.device)
        self.Critic = Critic().to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0, 0.9))
        self.optimizer_C = optim.Adam(self.Critic.parameters(), lr=self.lr_C, betas=(0, 0.9))
        # self.note_book_save_path = '/content/gdrive/MyDrive/Colab Notebooks/ImageColoringProject'
        self.note_book_save_path = ""

    def pretrain_generator(self):
        pretrained_model_path = f'{self.note_book_save_path}/saved_models/pretrained_model.pth'
        if os.path.exists(pretrained_model_path):
            self.generator.load_state_dict(torch.load(pretrained_model_path))
            print("Finished loading the previous pretrained model")
        else:
            print("Starting to pretrain the generator!")

        accuracy, g_loss_per_epoch, avg_psnr_per_epoch = self.load_pretrained_arrays()

        print("Starts to pretrain!")
        self.generator.train()
        for epoch in range(len(accuracy), self.num_epochs_pre):
            psnr_values = []
            g_loss_per_batch = []
            for batch_idx, (rgb_images, _) in enumerate(self.train_loader_rgb):

                rgb_images = rgb_images.to(self.device)
                gray_images = convert_to_greyscale_batch(rgb_images).to(self.device)

                gen_images = self.generator(gray_images)
                loss = self.MSEcriterion(gen_images, rgb_images)

                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()

                if batch_idx % 20 == 0:
                    self.save_pretrained_images(gen_images, gray_images, rgb_images, epoch, batch_idx)
                    g_loss_per_batch.append(loss)


                psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.to("cpu"), rgb_images.to("cpu"))])
                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" % (
                    epoch, self.num_epochs_pre, batch_idx, len(self.train_loader_rgb), loss.item()))

                rgb_images = rgb_images.to("cpu")
                gen_images = gen_images.to("cpu")
                gray_images = gray_images.to("cpu")

            # Calculate the average PSNR over all images
            avg_psnr = sum(psnr_values) / len(psnr_values)
            avg_psnr_per_epoch.append(avg_psnr)
            accuracy.append(avg_psnr)
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))

            print("[Epoch: %d/%d] [g_loss_train: %f] [PSNR: %.2f dB]" % (
                epoch, self.num_epochs_pre, np.average([l.item() for l in g_loss_per_batch]), avg_psnr))
            self.save_pretrained_model(pretrained_model_path)
            self.save_pretrained_arrays(accuracy, g_loss_per_epoch, avg_psnr_per_epoch)

            # Clean up pretrain tensors
            torch.cuda.empty_cache()
            gc.collect()

        return g_loss_per_epoch, accuracy, avg_psnr_per_epoch

    def load_pretrained_arrays(self):
        save_dir = f'{self.note_book_save_path}/saved_models'
        os.makedirs(save_dir, exist_ok=True)

        def load_array(filename):
            return list(np.load(filename)) if os.path.exists(filename) else []

        return (
            load_array(f'{save_dir}/pretrained_accuracy.npy'),
            load_array(f'{save_dir}/pretrained_g_loss_per_epoch.npy'),
            load_array(f'{save_dir}/pretrained_avg_psnr_per_epoch.npy')
        )

    def save_pretrained_images(self, gen_images, gray_images, rgb_images, epoch, batch_idx):
        os.makedirs(f'{self.note_book_save_path}/pre_trained_images', exist_ok=True)
        first_image_gen = prepare_to_save_image(gen_images[0])
        first_image_grey = prepare_to_save_image(gray_images[0])
        first_image_rbg = prepare_to_save_image(rgb_images[0])
        plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
        plt.savefig(f'{self.note_book_save_path}/pre_trained_images/pre_trained_image_{epoch}_{batch_idx}.jpg')
        plt.close()

    def save_pretrained_model(self, pretrained_model_path):
        torch.save(self.generator.state_dict(), pretrained_model_path)

    def save_pretrained_arrays(self, accuracy, g_loss_per_epoch, avg_psnr_per_epoch):
        save_dir = f'{self.note_book_save_path}/saved_models'
        np.save(f'{save_dir}/pretrained_accuracy.npy', np.array(accuracy))
        np.save(f'{save_dir}/pretrained_g_loss_per_epoch.npy', np.array(g_loss_per_epoch))
        np.save(f'{save_dir}/pretrained_avg_psnr_per_epoch.npy', np.array(avg_psnr_per_epoch))

    def test_model(self, loader_rgb):
        # Set model to eval mode
        self.generator.eval()
        self.Critic.eval()
        psnr_values_per_batch = []

        for batch_idx, (rgb_images, _) in enumerate(loader_rgb):
            rgb_images = rgb_images.to(self.device)
            gray_images = convert_to_greyscale_batch(rgb_images).to(self.device)

            # Generate rgb images
            gen_images = self.generator(gray_images)

            # Calculate PSNR values for each generated image
            psnr_values_per_batch.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.detach().to("cpu"), rgb_images.to("cpu"))])

            rgb_images = rgb_images.to("cpu")
            gen_images = gen_images.to("cpu")
            gray_images = gray_images.to("cpu")

        return psnr_values_per_batch

    def val_model(self, loader_rgb):
        # Set model to eval mode
        self.generator.eval()
        self.Critic.eval()
        c_loss_per_batch = []
        g_loss_per_batch = []
        mse_loss_per_batch = []

        for batch_idx, (rgb_images, _) in enumerate(loader_rgb):
            rgb_images = rgb_images.to(self.device)
            gray_images = convert_to_greyscale_batch(rgb_images).to(self.device)

            # Calculate Generator loss
            gen_images = self.generator(gray_images)
            wgan_loss = -torch.mean(self.Critic(gen_images))
            mse_loss = self.MSEcriterion(rgb_images, gen_images)
            g_loss = wgan_loss * 0.3 + mse_loss * 0.7

            # Calculate Critic loss
            c_loss = -torch.mean(self.Critic(rgb_images)) + torch.mean(self.Critic(gen_images.detach()))

            # Save Critic and Generator loss per batch
            c_loss_per_batch.append(c_loss.item())
            g_loss_per_batch.append(g_loss.item())
            mse_loss_per_batch.append(mse_loss.item())


            rgb_images = rgb_images.to("cpu")
            gen_images = gen_images.to("cpu")
            gray_images = gray_images.to("cpu")

        c_loss_avr = sum(c_loss_per_batch)/len(c_loss_per_batch)
        g_loss_avr = sum(g_loss_per_batch)/len(g_loss_per_batch)
        mse_loss_avr = sum(mse_loss_per_batch) / len(mse_loss_per_batch)


        # Reset model to train mode
        self.generator.train()
        self.Critic.train()

        return c_loss_avr, g_loss_avr, mse_loss_avr


    def results_visualization(self):
        counter = 0
        for batch_idx, (rgb_images, _) in enumerate(self.test_loader_rgb):

            rgb_images = rgb_images.to(self.device)
            gray_images = convert_to_greyscale_batch(rgb_images).to(self.device)

            # Generate RGB images from grayscale
            gen_images = self.generator(gray_images)

            for idx, (gray_image, rgb_image, gen_image) in enumerate(zip(gray_images, rgb_images, gen_images)):
                gray_image_np = prepare_to_save_image(gray_image)
                gen_image_np = prepare_to_save_image(gen_image)
                rgb_image_np = prepare_to_save_image(rgb_image)
                plt = make_subplot(rgb_image_np, gray_image_np, gen_image_np)

                plt.savefig(f'{self.note_book_save_path}/results/image_%d.jpg' % counter)
                plt.close()
                counter += 1

            rgb_images = rgb_images.to("cpu")
            gen_images = gen_images.to("cpu")
            gray_images = gray_images.to("cpu")

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

    def save_generated_images(self, gen_images, gray_images, rgb_images, epoch, batch_idx):
        os.makedirs(f'{self.note_book_save_path}/generated_images', exist_ok=True)
        first_image_gen = prepare_to_save_image(gen_images[0])
        first_image_grey = prepare_to_save_image(gray_images[0])
        first_image_rbg = prepare_to_save_image(rgb_images[0])
        plt = make_subplot(first_image_rbg, first_image_grey, first_image_gen)
        plt.savefig(f'{self.note_book_save_path}/generated_images/image_epoch_{epoch}batch{batch_idx}.jpg')
        plt.close()

    def load_models(self):
        if os.path.exists(f'{self.note_book_save_path}/saved_models/generator_model.pth') and os.path.exists(
                f'{self.note_book_save_path}/saved_models/Critic_model.pth'):
            self.generator.load_state_dict(torch.load(f'{self.note_book_save_path}/saved_models/generator_model.pth'))
            self.Critic.load_state_dict(torch.load(f'{self.note_book_save_path}/saved_models/Critic_model.pth'))
            print("Finished loading the previous trained models!")
        elif os.path.exists(f'{self.note_book_save_path}/saved_models/pretrained_model.pth'):
            self.generator.load_state_dict(torch.load(f'{self.note_book_save_path}/saved_models/pretrained_model.pth'))
            print("Finished loading the pretrained generator!")
        else:
            print("Starting to train without pretrained model!")

    def save_arrays(self, c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c, mse_losses_per_epoch, wgan_losses_per_epoch,val_losses_mse):
        np.save(f'{self.note_book_save_path}/saved_models/c_loss_per_epoch.npy', np.array(c_loss_per_epoch))
        np.save(f'{self.note_book_save_path}/saved_models/g_loss_per_epoch.npy', np.array(g_loss_per_epoch))
        np.save(f'{self.note_book_save_path}/saved_models/accuracy.npy', np.array(accuracy))
        np.save(f'{self.note_book_save_path}/saved_models/val_losses_g.npy', np.array(val_losses_g))
        np.save(f'{self.note_book_save_path}/saved_models/val_losses_c.npy', np.array(val_losses_c))
        np.save(f'{self.note_book_save_path}/saved_models/val_losses_mse.npy', np.array(val_losses_mse))
        np.save(f'{self.note_book_save_path}/saved_models/wgan_losses_per_epoch.npy', np.array(wgan_losses_per_epoch))
        np.save(f'{self.note_book_save_path}/saved_models/mse_losses_per_epoch.npy', np.array(mse_losses_per_epoch))
        torch.save(self.generator.state_dict(), f'{self.note_book_save_path}/saved_models/generator_model.pth')
        torch.save(self.Critic.state_dict(), f'{self.note_book_save_path}/saved_models/Critic_model.pth')

    def load_arrays(self):
        def load_array(filename):
            return list(np.load(filename)) if os.path.exists(filename) else []

        return(
            load_array(f'{self.note_book_save_path}/saved_models/c_loss_per_epoch.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/g_loss_per_epoch.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/accuracy.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/val_losses_g.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/val_losses_c.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/val_losses_mse.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/wgan_losses_per_epoch.npy'),
            load_array(f'{self.note_book_save_path}/saved_models/mse_losses_per_epoch.npy')
            # load_array(f'{self.note_book_save_path}/saved_models/test_accuracy.npy')
        )

    def train(self):
        # Load previous trained models and arrays
        self.load_models()

        # Initialize arrays
        c_loss_per_epoch = []
        g_loss_per_epoch = []
        accuracy = []
        val_losses_g = []
        val_losses_c = []
        val_losses_mse = []
        mse_losses_per_epoch = []
        wgan_losses_per_epoch = []
        train_critic = False

        # Load arrays
        # c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c,val_losses_mse, wgan_losses_per_epoch, mse_losses_per_epoch, test_accuracy = self.load_arrays()
        c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c,val_losses_mse, wgan_losses_per_epoch, mse_losses_per_epoch = self.load_arrays()

        # Configure to train mode
        self.generator.train()
        self.Critic.train()

        # Train the model
        for epoch in range(len(c_loss_per_epoch), self.num_epochs):
            psnr_values = []
            g_loss_per_batch = []
            c_loss_per_batch = []
            mse_losses_per_batch = []
            wgan_losses_per_batch = []
            for batch_idx, (rgb_images, _) in enumerate(self.train_loader_rgb):

                rgb_images = rgb_images.to(self.device)
                gray_images = convert_to_greyscale_batch(rgb_images).to(self.device)

                # Training the critic every 4 steps
                if batch_idx % 4 == 0 and batch_idx != 0:
                    train_critic = True
                    # Train the critic
                    gen_images = self.generator(gray_images)
                    self.optimizer_C.zero_grad()
                    loss_c = -torch.mean(self.Critic(rgb_images)) + torch.mean(self.Critic(gen_images.detach()))
                    gp = self.gradient_penalty(rgb_images, gen_images.detach())
                    loss_c += 10 * gp
                    loss_c.backward()
                    self.optimizer_C.step()

                # Training the generator
                for param in self.Critic.parameters():
                    param.requires_grad = False

                self.optimizer_G.zero_grad()
                gen_images = self.generator(gray_images)
                wgan_loss = -torch.mean(self.Critic(gen_images))
                mse_loss = self.MSEcriterion(rgb_images, gen_images)
                loss_g = wgan_loss * 0.3 + mse_loss * 0.7
                loss_g.backward()
                self.optimizer_G.step()

                for param in self.Critic.parameters():
                    param.requires_grad = True

                if train_critic:
                    c_loss_per_batch.append(loss_c)
                    g_loss_per_batch.append(loss_g)
                    mse_losses_per_batch.append(mse_loss.item())
                    wgan_losses_per_batch.append(wgan_loss.item())

                    psnr_values.extend([psnr(gen_img, rgb_img) for gen_img, rgb_img in zip(gen_images.detach().to("cpu"), rgb_images.to("cpu"))])

                    # Print loss
                    print("[Epoch %d/%d] [Batch %d/%d] [Critic loss: %f] [G loss: %f] [PSNR accuracy: %f] "
                          % (epoch, self.num_epochs, batch_idx, len(self.train_loader_rgb), loss_c.item(),
                             loss_g.item(), sum(psnr_values) / len(psnr_values)))

                    # Save generated images
                    self.save_generated_images(gen_images, gray_images, rgb_images, epoch, batch_idx)
                    train_critic = False

                rgb_images = rgb_images.to("cpu")
                gray_images = gray_images.to("cpu")
                gen_images = gen_images.to("cpu")

                # Free up pretrain tensors
                torch.cuda.empty_cache()

            # Update Validation losses arrays
            c_loss_val, g_loss_val, mse_loss_val  = self.val_model(self.eval_loader_rgb)
            val_losses_c.append(c_loss_val)
            val_losses_g.append(g_loss_val)
            val_losses_mse.append(mse_loss_val)

            # Calculate model accuracy
            accuracy.append(sum(psnr_values) / len(psnr_values))

            # Calculate loss per epoch
            g_loss_per_epoch.append(np.average([l.item() for l in g_loss_per_batch]))
            c_loss_per_epoch.append(np.average([l.item() for l in c_loss_per_batch]))
            mse_losses_per_epoch.append(np.average([l for l in mse_losses_per_batch]))
            wgan_losses_per_epoch.append(np.average([l for l in wgan_losses_per_batch]))

            # Save arrays
            self.save_arrays(c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c, mse_losses_per_epoch, wgan_losses_per_epoch, val_losses_mse)

            # Free up pretrain tensors
            torch.cuda.empty_cache()

        # Update Test losses arrays
        if not os.path.exists(f'{self.note_book_save_path}/saved_models/test_accuracy.pth'):
          test_accuracy = self.test_model(self.test_loader_rgb)
          np.save(f'{self.note_book_save_path}/saved_models/test_accuracy.npy', np.array(test_accuracy))

        return c_loss_per_epoch, g_loss_per_epoch, accuracy, val_losses_g, val_losses_c, mse_losses_per_epoch, wgan_losses_per_epoch, test_accuracy, val_losses_mse

    def return_arrays(self):
        c_loss_per_epoch, g_loss_per_epoch, accuracy, test_accuracy, val_accuracy, test_losses_g, val_losses_g, mse_losses_per_epoch, wgan_losses_per_epoch = self.load_arrays()
        return c_loss_per_epoch, g_loss_per_epoch, accuracy, test_accuracy, val_accuracy, test_losses_g, val_losses_g, mse_losses_per_epoch, wgan_losses_per_epoch
