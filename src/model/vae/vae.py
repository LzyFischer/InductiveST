"""
VAE model for encoding time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os
import random

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.cfg = cfg

    def forward(self, x):
        # input size: B, N, C, L
        B, N, C, L = x.shape
        x = x.transpose(-1, -2).reshape(-1, L, C)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        z = self.decoder(z)
        z = z.transpose(-1, -2).reshape(B, N, C, L)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * self.cfg["vae"].get("variance", 0.1)
