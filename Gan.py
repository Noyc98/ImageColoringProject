import torch
import torch.nn as nn


# Define VGG block
def vgg_block(in_channels, out_channels):
    """
    Creates a VGG block consisting of two convolutional layers with ReLU activation functions and instance normalization.

    Args:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.

    Returns:
    - Sequential: VGG block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        """
        Initializes a convolutional block consisting of two convolutional layers with batch normalization and ReLU activation functions.

        Args:
        - in_c: Number of input channels.
        - out_c: Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Forward pass of the convolutional block.

        Args:
        - inputs: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        """
        Initializes an encoder block consisting of a convolutional block followed by max pooling.

        Args:
        - in_c: Number of input channels.
        - out_c: Number of output channels.
        """
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        """
        Forward pass of the encoder block.

        Args:
        - inputs: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - x: Output tensor of shape (batch_size, out_channels, height, width).
        - p: Pooled tensor of shape (batch_size, out_channels, pooled_height, pooled_width).
        """
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        """
        Initializes a decoder block consisting of upsampling followed by a convolutional block.

        Args:
        - in_c: Number of input channels.
        - out_c: Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        """
        Forward pass of the decoder block.

        Args:
        - inputs: Input tensor of shape (batch_size, in_channels, height, width).
        - skip: Skip connection tensor from encoder block of shape (batch_size, skip_channels, skip_height, skip_width).

        Returns:
        - Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


# Define Generator (U-Net) architecture with VGG blocks
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 16)
        self.e2 = encoder_block(16, 32)
        """ Bottleneck """
        self.b = conv_block(32, 64)
        """ Decoder """
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)
        """ Classifier """
        self.outputs = nn.Conv2d(16, 3, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ 
        Forward pass of the U-Net Generator.

        Args:
        - inputs: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        """ Bottleneck """
        b = self.b(p2)
        """ Decoder """
        d3 = self.d3(b, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 =
