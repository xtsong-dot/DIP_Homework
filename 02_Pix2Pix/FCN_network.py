# import torch.nn as nn

# class FullyConvNetwork(nn.Module):

#     def __init__(self):
#         super().__init__()
#          # Encoder (Convolutional Layers)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True)
#         )
#         ### FILL: add more CONV Layers
        
#         # Decoder (Deconvolutional Layers)
#         ### FILL: add ConvTranspose Layers
#         ### None: since last layer outputs RGB channels, may need specific activation function

#     def forward(self, x):
#         # Encoder forward pass
        
#         # Decoder forward pass
        
#         ### FILL: encoder-decoder forward pass

#         output = ...
        
#         return output
    
import torch.nn as nn
import torch

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # --- Encoder (下采样: 图片尺寸减半，通道数增加) ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Decoder (上采样: 图片尺寸翻倍，通道数减少) ---
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 最后一次卷积，输出通道为 3 (RGB)，并使用 Tanh 激活函数
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 128 -> 256
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Decoder forward pass
        d1 = self.deconv1(x4)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        output = self.deconv4(d3)
        
        return output

    def __init__(self):
        super().__init__()
        
        # Encoder (Convolutional Layers)
        # Input: 3 x 256 x 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Out: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Out: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Out: 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Out: 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Out: 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Out: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Out: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),   # Out: 3 x 256 x 256
            nn.Tanh() # Tanh scales output to [-1, 1] to match the dataset normalization
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Decoder forward pass
        d1 = self.deconv1(x4)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        output = self.deconv4(d3)
        
        return output