import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from ...lib.utils import DataTrimer


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

    def forward(
        self, history_data: torch.Tensor, future_data: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Feedforward function of LSTM.

        Args:
            history_data (torch.Tensor): shape [B, N, C, L]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """
        x = history_data.transpose(1, 2)
        b, c, n, l = x.shape
        x = x.transpose(1, 2).reshape(b * n, c, 1, l)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, l)
        return x, None, None, None
