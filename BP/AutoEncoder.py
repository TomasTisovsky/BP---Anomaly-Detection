import torch.nn as nn
from torchvision.models import resnet18


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.encoder = nn.Sequential( 
                nn.Conv2d(1, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01)
        )
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 1, stride=(2, 2), kernel_size=(3, 3), padding=1), 
                nn.Sigmoid()
        )
        """
        """
        # Example: Add more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 512, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 1024, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(1024, 2048, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            # Add more layers as needed
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(1024, 512, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 16, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 1, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.Sigmoid()
            # Add more layers as needed
        )
        """
        # Example: Add more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 512, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 1024, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.Conv2d(1024, 2048, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            # Add more layers as needed
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(1024, 512, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(512, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),  # Additional layer
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.Sigmoid()
            # Add more layers as needed
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

