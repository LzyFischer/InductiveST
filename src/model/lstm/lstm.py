import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from ...lib.utils import DataTrimer
from ..vae.vae import VAE


class LSTM(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        """Init LSTM.

        Args:
            input_dim (int): number of input features.
            embed_dim (int): dimension of the input embedding layer (a linear layer).
            hidden_dim (int): hidden size in LSTM.
            end_dim (int): hidden dimension of the output linear layer.
            num_layer (int): number of layers in LSTM.
            dropout (float): dropout rate.
            horizon (int): number of time steps to be predicted.
        """
        super(LSTM, self).__init__()
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        random_seed = cfg["seed"]
        train_ratio = cfg["train_node_ratio"]
        val_ratio = cfg["val_node_ratio"]

        network_path = cfg["data_root"] + cfg["networks_name"]
        adj = np.load(network_path, allow_pickle=True)
        adj = torch.tensor(adj, dtype=torch.float32)

        input_dim = cfg["model"]["input_dim"]
        embed_dim = cfg["model"]["embed_dim"]
        hidden_dim = cfg["model"]["hidden_dim"]
        end_dim = cfg["model"]["end_dim"]
        dropout = cfg["model"]["dropout"]
        num_layer = cfg["model"]["num_layer"]
        horizon = cfg["model"]["horizon"]
        num_nodes = cfg["num_nodes"]

        train_num_nodes = int(train_ratio * num_nodes)
        val_num_nodes = int(val_ratio * num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

        data_trimer = DataTrimer(cfg)
        num_nodes_list, nodes_list, graph_list = data_trimer()

        self.train_graph, self.val_graph, self.test_graph = graph_list

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=embed_dim, kernel_size=(1, 1)
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout,
        )

        self.end_linear1 = nn.Linear(hidden_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)

        self.vae = VAE(cfg)

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

    def forward(
        self, history_data: torch.Tensor, future_data: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Feedforward function of LSTM.

        Args:
            history_data (torch.Tensor): shape [B, N, C, L]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        embedding = None
        vae_loss = 0
        B, N, C, L = history_data.shape
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

        x = history_data.transpose(1, 2)
        b, c, n, l = x.shape
        x = x.transpose(1, 2).reshape(b * n, c, 1, l)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, l)
        return x, future_data, None, vae_loss
