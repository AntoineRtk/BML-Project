import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data
import torchvision.utils as vutils

def kl(mu, log_var):
    """
    :param mu: (batch, z_dim, map_x, map_y)
    :param log_var: (batch, z_dim, map_x, map_y)
    :return: Kullback-Leibler divergence between a N(mu, var) distribution and a N(0, I) distribution
    """
    loss = 0.5 * torch.sum(mu ** 2 - log_var + torch.exp(log_var) - 1, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)

def kl_delta_reversed(delta_mu, delta_log_var, mu, log_var):
    loss = 0.5 * torch.sum(delta_mu ** 2 / log_var.exp() + delta_log_var.exp() - delta_log_var - 1, dim = [1, 2, 3])
    return loss.mean(dim = 0)


def kl_delta_forward(delta_mu, delta_log_var, mu, log_var):
    loss = 0.5 * torch.sum((delta_mu ** 2) / (log_var.exp() * delta_log_var.exp()) + (1 / delta_log_var.exp()) + delta_log_var - 1)
    return loss.mean(dim = 0)

def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z
    
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)
        
# Encoder:
class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, ks=5):
        super().__init__()

        self._seq = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=ks),
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2), Swish(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=ks-2),
            nn.BatchNorm2d(out_channel), Swish()
        )

    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels, ks=5):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(ConvBlock(channels[i], channels[i + 1], ks))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class EncoderResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class Encoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([1, z_dim // 8, z_dim // 4]),  # (16, 16)
            EncoderBlock([z_dim // 4, z_dim // 4, z_dim // 2]),  # (4, 4)
            EncoderBlock([z_dim // 2, z_dim], ks=3),  # (2, 2)
        ])
        #print('enc',self.encoder_blocks)

        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim // 4),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim),
        ])
        #print('enc',self.encoder_residual_blocks)

        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv2d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = []
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            #print('X shape before',x.shape)
            x = r(e(x))
            #print('X shape after',x.shape)
            xs.append(x)

        mu, log_var = self.condition_x(x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1]

# Decoder:
class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ks=3, s=2, p=0, op=1):
        super().__init__()

        self._seq = nn.Sequential(

            nn.ConvTranspose2d(in_channel,
                               out_channel,
                               kernel_size=ks,
                               stride=s,
                               padding=p,
                               output_padding=op),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel), Swish(),
        )

    def forward(self, x):
        return self._seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=3, padding=1, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


class DecoderBlock(nn.Module):

    def __init__(self, channels, ks=3, s=2, p=0, op=1):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1], ks, s, p, op))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim, forward_kl):
        super().__init__()

        # Input channels = z_channels * 2 = x_channels + z_channels
        # Output channels = z_channels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim * 2, z_dim // 2], 3, 2, 1, 1),  # 2x upsample
            DecoderBlock([z_dim, z_dim // 4, z_dim // 4], 3, 2, 1),  # 4x upsample
            DecoderBlock([z_dim // 2, z_dim // 8], 4, 2, 3, 0)  # 8x uplsampe
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim // 2, n_group=4),
            DecoderResidualBlock(z_dim // 4, n_group=2),
            DecoderResidualBlock(z_dim // 8, n_group=1)
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 4),
                Swish(),
                nn.Conv2d(z_dim // 4, z_dim // 2, kernel_size=1)
            )
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                nn.Conv2d(z_dim, z_dim // 2, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                nn.Conv2d(z_dim // 2, z_dim // 4, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 4, z_dim // 2, kernel_size=1)
            )
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim // 8),
            nn.Conv2d(z_dim // 8, 1, kernel_size=1),
        )
        
        self.forward_kl = forward_kl


    def forward(self, z, xs=None):
        """

        :param z: shape. = (B, z_dim, map_h, map_w)
        if xs=None: sample mode; otherwise xs is list of intermediate encoder features
        """

        B, D, map_h, map_w = z.shape
        
        decoder_out = torch.zeros(B, D, map_h, map_w, device=z.device, dtype=z.dtype)

        kl_losses = []

        for i in range(len(self.decoder_residual_blocks)):

            z_sample = torch.cat([decoder_out, z], dim=1)
            #print('Z before',z_sample.shape)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))
            #print('Z after',decoder_out.shape)

            if i == len(self.decoder_residual_blocks) - 1: # stop if last block
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1) # parameter for sampling next z

            if xs is not None:
                #print('intermediate xs',xs[i].shape,'dec',decoder_out.shape)
                delta_mu, delta_log_var = self.condition_xz[i](
                                torch.cat([xs[i], decoder_out], dim=1)).chunk(2, dim=1)
                if(self.forward_kl):
                    kl_losses.append(kl_delta_forward(delta_mu, delta_log_var, mu, log_var))
                else:
                    kl_losses.append(kl_delta_reversed(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            z = reparameterize(mu, torch.exp(0.5 * log_var))

        #print('xhat',decoder_out.shape)
        x_hat = torch.sigmoid(self.recon(decoder_out))

        return x_hat, kl_losses
        
class NVAE(nn.Module):

    def __init__(self, z_dim, forward_kl):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim, forward_kl)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return: Tensor. shape = (B, C, H, W), loss list, kl_losses list
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var)) # sampling top latent variable

        decoder_output, kl_losses = self.decoder(z, xs)

        kl_losses = [kl(mu, log_var)]+kl_losses

        recon_loss = nn.MSELoss(reduction='sum')(decoder_output, x)/decoder_output.shape[0]

        return decoder_output, recon_loss, kl_losses