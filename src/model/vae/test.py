import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VAE

input = torch.randn(1, 10, 2)
cfg = {
    "input_dim": 2,
    "hidden_dim": 64,
    "latent_dim": 16,
    "num_layers": 2,
    "dropout": 0.1,
}

vae = VAE(cfg)

output, mu, logvar = vae(input)

print(output.shape, mu.shape, logvar.shape)
