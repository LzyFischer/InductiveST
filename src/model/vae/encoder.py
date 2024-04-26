"""encoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        # self.input_length = cfg["window_size"] - cfg["horizon"]
        self.input_length = cfg["window_size"]
        cfg = cfg["vae"]
        self.input_dim = cfg["input_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.oneshot_fc_mu = nn.Linear(
            self.input_dim * self.input_length, self.latent_dim
        )
        self.oneshot_fc_logvar = nn.Linear(
            self.input_dim * self.input_length, self.latent_dim
        )

    def forward(self, x):
        # size
        # _, (h, _) = self.lstm(x)
        # h = h[-1]
        # mu = self.fc_mu(h)
        # logvar = self.fc_logvar(h)
        x = x.reshape(-1, self.input_length * self.input_dim)
        mu = self.oneshot_fc_mu(x)
        logvar = self.oneshot_fc_logvar(x)
        return mu, logvar
