import torch
import torch.nn as nn


# Define VGG block
def vgg_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# Define Generator (U-Net) architecture with VGG blocks
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        # Define encoder layers
        self.encoder = nn.Sequential(
            vgg_block(3, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define Discriminator architecture (unchanged)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
