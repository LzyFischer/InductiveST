"""decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import os


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.window_size = cfg["window_size"]
        # self.input_length = cfg["window_size"] - cfg["horizon"]
        self.input_length = cfg["window_size"]
        cfg = cfg["vae"]
        self.input_dim = cfg["input_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]

        self.lstmcell = nn.LSTMCell(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
        )
        self.output_fc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.final_fc = nn.Linear(self.hidden_dim, self.input_dim)

        # one shot output
        self.oneshot_fc = nn.Linear(self.hidden_dim, self.input_dim * self.input_length)

    def forward(self, x):
        # output = []
        # o = x
        # for i in range(self.window_size):
        #     if i == 0:
        #         h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        #         c = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        #     h, c = self.lstmcell(o, (h, c))
        #     o = self.output_fc(h)
        #     output.append(h)
        # output = torch.stack(output, dim=1)
        # output = self.final_fc(output)

        output = self.oneshot_fc(x)
        output = output.view(-1, self.input_length, self.input_dim)

        return output
