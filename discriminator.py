import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """CycleGAN Discriminator using PatchGAN."""

    def __init__(self, input_nc):
        """
        Args:
            input_nc (int): Number of input channels (e.g., 3 for RGB images).
        """
        super(Discriminator, self).__init__()

        #  model architecture is defined 
        self.model = nn.Sequential(
            # Input layer: layer 1--> downsamples the image
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: Adds more filters, and more details are detected 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # OUtput Layer 
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Discriminator output.
        """
        return self.model(x)
