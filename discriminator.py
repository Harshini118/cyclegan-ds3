import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """CycleGAN Discriminator (PatchGAN)"""
    
    def __init__(self, input_nc):
        """
        Args:
            input_nc (int): Number of channels in the input image (e.g., 3 for RGB)
        """
        super(Discriminator, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            conv_block(input_nc, 64, normalization=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Final output layer
        )
    
    def forward(self, x):
        return self.model(x)
