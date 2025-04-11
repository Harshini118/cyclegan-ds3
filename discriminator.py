import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """CycleGAN Discriminator
    paper uses PatchGAN so we will too """

    def __init__(self, input_nc):
        """
        args:
            input_nc (int): number of channels (like 3 channels for rgb) of input image
        """

         super(Discriminator, self).__init__()
        
        # TODO
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

            
        )

    def forward(self, x):
        return self.model(x)
