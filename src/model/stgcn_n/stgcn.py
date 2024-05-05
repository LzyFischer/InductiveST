import torch
import torch.nn as nn
import pdb
import os
import random
import networkx as nx
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix

from ...lib.utils import DataTrimer

# from ...utils import trim_networks
from .stconv import STConv

from ..vae.vae import VAE
from ..graph_learner.simlearner import SimLearner

import matplotlib.pyplot as plt


class STGCN_n(nn.Module):
    """
    Paper: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Trafﬁc Forecasting
    Official Code: https://github.com/VeritasYin/STGCN_IJCAI-18 (tensorflow)
    Ref Code: https://github.com/hazdzz/STGCN
    Note:
        https://github.com/hazdzz/STGCN/issues/9
    Link: https://arxiv.org/abs/1709.04875
    """

    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        cfg,
    ):
        super(STGCN_n, self).__init__()

        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        random_seed = cfg["seed"]
        train_ratio = cfg["train_node_ratio"]
        val_ratio = cfg["val_node_ratio"]

        network_path = cfg["data_root"] + cfg["networks_name"]
        adj = np.load(network_path, allow_pickle=True)
        adj = torch.tensor(adj, dtype=torch.float32)

        num_nodes = cfg["model"]["num_nodes"]
        in_channels = cfg["model"]["in_channels"]
        hidden_channels = cfg["model"]["hidden_channels"]
        out_channels = cfg["model"]["out_channels"]
        kernel_size = cfg["model"]["kernel_size"]
        K = cfg["model"]["K"]
        normalization = cfg["model"]["normalization"]
        bias = cfg["model"]["bias"]

        train_num_nodes = int(train_ratio * num_nodes)
        val_num_nodes = int(val_ratio * num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

        data_trimer = DataTrimer(cfg)
        num_nodes_list, nodes_list, graph_list = data_trimer()

        self.train_graph, self.val_graph, self.test_graph = graph_list
        # to index the graph
        self.train_graph = self.train_graph.to_sparse()._indices()
        self.val_graph = self.val_graph.to_sparse()._indices()
        self.test_graph = self.test_graph.to_sparse()._indices()

        self.st_block_1 = STConv(
            [train_num_nodes, val_num_nodes, num_nodes],
            in_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            K,
            cfg,
            normalization,
            bias,
        )

        self.st_block_2 = STConv(
            [train_num_nodes, val_num_nodes, num_nodes],
            hidden_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            K,
            cfg,
            normalization,
            bias,
        )

        self.output_layer = nn.Linear(hidden_channels * 4, out_channels)

        self.vae = VAE(cfg)

        self.graph_learner = SimLearner(cfg)

        if not self.cfg.get("dynamic_mix", True):
            self.mix_pair = (
                torch.stack(
                    [
                        torch.multinomial(
                            torch.ones(self.train_num_nodes),
                            self.train_num_nodes,
                            replacement=True,
                        )[: int(self.train_num_nodes * 1)],
                        torch.multinomial(
                            torch.ones(self.train_num_nodes),
                            self.train_num_nodes,
                            replacement=True,
                        )[: int(self.train_num_nodes * 1)],
                    ],
                    dim=0,
                )
                .to(torch.int64)
                .to(self.cfg["device"])
            )

    def initilization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """feedforward function of STGCN.

        Args:
            history_data (torch.Tensor): historical data with shape [B, N, C, L]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        # ### mixup ####
        # if self.training:
        #     B, N, C, L = history_data.shape
        #     time_1 = history_data
        #     a = torch.fft.fft(time_1)
        #     amplitude = torch.sqrt(a.real**2 + a.imag**2)
        #     phase = torch.atan(a.imag / a.real)
        #     # change high level information
        #     amplitude[..., : L // 5] = 0
        #     amplitude[..., L // 5 :] = 0
        #     # amplitude[..., L * 2 // 5 : L * 3 // 5] = 0
        #     a = amplitude * torch.exp(phase * 1j)
        #     time_1 = torch.fft.ifft(a).real
        #     # let nan to zero
        #     time_1[torch.isnan(time_1)] = 0
        #     history_data[: B // 2] = time_1[: B // 2]
        #     #### mixup done ####

        # initialize
        embedding = None
        vae_loss = 0

        if self.cfg["drop_node"] and self.training:
            drop_node = torch.randperm(history_data.shape[1])[
                : int(history_data.shape[1] * self.cfg["drop_node_ratio"])
            ]
            history_data[:, drop_node] = 0
            future_data[:, drop_node] = 0

        B, N, C, L = history_data.shape

        if self.cfg["is_vae"]:
            full_data = torch.cat([history_data, future_data], dim=3)

            ## Split high and low frequency
            B, N, C, W = full_data.shape
            time_1 = full_data
            a = torch.fft.fftshift(torch.fft.fft(time_1))
            amplitude = torch.sqrt(a.real**2 + a.imag**2)
            high_amplitude, low_amplitude = deepcopy(amplitude), deepcopy(amplitude)
            phase = torch.atan(a.imag / a.real)
            # change high level information
            # low_amplitude[..., : W // 5] = 0
            # low_amplitude[..., W * 4 // 5 :] = 0
            high_amplitude[..., W // 50 : W * 49 // 50] = 0
            # amplitude[..., L * 2 // 5 : L * 3 // 5] = 0
            a_low = low_amplitude * torch.exp(phase * 1j)
            a_high = high_amplitude * torch.exp(phase * 1j)
            time_low = torch.fft.ifft(torch.fft.ifftshift(a_low)).real
            time_high = torch.fft.ifft(torch.fft.ifftshift(a_high)).real
            # let nan to zero
            time_low[torch.isnan(time_low)] = 0

            time_high[torch.isnan(time_high)] = 0

            ## VAE
            recover_data, mu, logvar = self.vae(time_high)
            vae_loss = torch.nn.functional.mse_loss(recover_data, time_high)

            ## Mixup
            ### randomly choose a node as aug node
            rand_node_1 = torch.randperm(B * N)
            # aug_node = (
            #     self.cfg["anchor_lambda"] * mu
            #     + (1 - self.cfg["anchor_lambda"]) * mu[rand_node_1]
            # )
            # aug_logvar = (
            #     self.cfg["anchor_lambda"] * logvar
            #     + (1 - self.cfg["anchor_lambda"]) * logvar[rand_node_1]
            # )
            aug_node = mu
            aug_logvar = logvar

            z = self.vae.reparameterize(aug_node, aug_logvar)
            z = self.vae.decoder(z)
            aug_node_high = z.transpose(-1, -2).reshape(B, N, C, W)

            rand_node_2 = torch.randperm(B * N)
            aug_node_low = time_low.reshape(-1, C, W).reshape(B, N, C, W)

            ### Combine high and low frequency
            a_low = torch.fft.fft(aug_node_low)
            a_high = torch.fft.fft(aug_node_high)
            a = a_low + a_high
            time_2 = torch.fft.ifft(a).real
            time_2[torch.isnan(time_2)] = 0

            # full_data = torch.cat([full_data, time_2], dim=1)
            full_data = time_2

            history_data = time_low[..., :L]
            future_data = time_low[..., L:]

        """
        1. generate augmented nodes
            1. input: input data or after encoder
            2. output: augmented nodes need to be concatenated 
            3. need to get only original nodes from the output
            4. randomly selected pair of nodes to mix in hidden space
        2. learn with augmented nodes
        """
        if self.cfg.get("dynamic_mix", True):
            self.mix_pair = (
                torch.stack(
                    [
                        torch.multinomial(
                            torch.ones(self.train_num_nodes),
                            self.train_num_nodes,
                            replacement=True,
                        )[: int(self.train_num_nodes * 1)],
                        torch.multinomial(
                            torch.ones(self.train_num_nodes),
                            self.train_num_nodes,
                            replacement=True,
                        )[: int(self.train_num_nodes * 1)],
                    ],
                    dim=0,
                )
                .to(torch.int64)
                .to(self.cfg["device"])
            )

        ## Augmented nodes
        if self.cfg["aug_node"] and self.training:
            # transform to hidden space
            # vae = VAE(self.cfg).to(self.cfg["device"])
            # vae(history_data)
            full_data = torch.cat([history_data, future_data], dim=3)
            B, N, C, W = full_data.shape
            transformed_input = full_data.transpose(-1, -2).reshape(-1, W, C)
            mu, logvar = self.vae.encoder(transformed_input)
            embedding = self.vae.reparameterize(mu, logvar)

            vae_loss += torch.mean(
                -(self.cfg.get("vae_loss_weight", 2))
                * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1),
                dim=0,
            )

            (_, hidden_shape) = embedding.shape
            # mixup with self.mix_pair
            aug_node = (
                self.cfg["anchor_lambda"]
                * embedding.reshape(B, N, -1)[:, self.mix_pair[0]]
                + (1 - self.cfg["anchor_lambda"])
                * embedding.reshape(B, N, -1)[:, self.mix_pair[1]]
            )
            aug_node = aug_node.reshape(-1, aug_node.shape[-1])
            # transform to original space
            aug_node = self.vae.decoder(aug_node)
            aug_node = aug_node.transpose(-1, -2).reshape(B, N, C, W)

            # similarity loss
            if self.cfg.get("aug_loss", "MSE") == "MSE":
                vae_loss += torch.nn.functional.mse_loss(
                    full_data[:, self.mix_pair[0]], aug_node
                )
                vae_loss += torch.nn.functional.mse_loss(
                    full_data[:, self.mix_pair[1]], aug_node
                )
            elif self.cfg.get("aug_loss", "MSE") == "Contrastive":
                vae_loss += torch.nn.functional.cosine_similarity(
                    full_data[:, self.mix_pair[0]], aug_node
                ).sum()
                vae_loss += torch.nn.functional.cosine_similarity(
                    full_data[:, self.mix_pair[1]], aug_node
                ).sum()

                random_seq = torch.multinomial(
                    torch.ones(self.train_num_nodes * B),
                    self.train_num_nodes * B,
                    replacement=True,
                )[: int(self.train_num_nodes * B)]

                conts_loss = torch.nn.functional.cosine_similarity(
                    full_data.reshape(-1, C, W)[random_seq].reshape(
                        B, self.mix_pair.shape[1], C, W
                    ),
                    aug_node,
                ).sum()

                vae_loss /= conts_loss
            # combine with original nodes
            history_data = torch.cat([history_data, aug_node[..., :L]], dim=1)
            future_data = torch.cat([future_data, aug_node[..., L:]], dim=1)
        ## Learning graph
        self.edge_weight = None
        self.edge_index = None
        if self.cfg["graph_learning"]:
            self.graph = self.graph_learner(history_data)
            # get edge index in a batch
            if self.cfg["dynamic_graph"]:
                batch, row, col = (self.graph > 0).nonzero().t()

                self.edge_index = torch.stack([row + batch * N, col + batch * N], dim=0)
                self.edge_weight = self.graph[batch, row, col]
            else:
                row, col = (self.graph > 0).nonzero().t()
                self.edge_index = torch.stack([row, col], dim=0)
                self.edge_weight = self.graph[row, col]

        x = history_data.permute(0, 3, 1, 2).contiguous()  # [B,L,N,C]
        if self.edge_index is not None:
            x = self.st_block_1(x, self.edge_index.to(x.device), self.edge_weight)

            x = self.st_block_2(x, self.edge_index.to(x.device), self.edge_weight)
        elif self.training:
            x = self.st_block_1(x, self.train_graph.to(x.device), self.edge_weight)
            x = self.st_block_2(x, self.train_graph.to(x.device), self.edge_weight)
        elif self.val_num_nodes == x.shape[2]:
            x = self.st_block_1(x, self.val_graph.to(x.device), self.edge_weight)
            x = self.st_block_2(x, self.val_graph.to(x.device), self.edge_weight)
        else:
            x = self.st_block_1(x, self.test_graph.to(x.device), self.edge_weight)
            x = self.st_block_2(x, self.test_graph.to(x.device), self.edge_weight)

        x = x.permute(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[2], x.shape[1] * x.shape[3]
        )
        x = self.output_layer(x)  # [B,N,L]

        return (x, future_data, embedding, vae_loss)
