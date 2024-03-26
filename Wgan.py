import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the generator network with an encoder-decoder architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),  # input: 256x256x1, output: 128x128x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),  # input: 128x128x64, output: 64x64x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # input: 64x64x128, output: 32x32x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # input: 32x32x256, output: 64x64x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # input: 64x64x128, output: 128x128x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),  # input: 128x128x64, output: 256x256x3
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # Decode
        x = self.decoder(x)
        return x


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.conv3(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        return x


# Define Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)