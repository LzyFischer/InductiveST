import torch
from torch import nn
import numpy as np

from ...lib.utils import DataTrimer


class MLP(nn.Module):
    """Two fully connected layer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        random_seed = cfg["seed"]
        train_ratio = cfg["train_node_ratio"]
        val_ratio = cfg["val_node_ratio"]

        network_path = cfg["data_root"] + cfg["networks_name"]
        adj = np.load(network_path, allow_pickle=True)
        adj = torch.tensor(adj, dtype=torch.float32)

        history_seq_len = (
            cfg["model"]["window_size"] - cfg["model"]["prediction_seq_len"]
        )
        prediction_seq_len = cfg["model"]["prediction_seq_len"]
        hidden_dim = cfg["model"]["hidden_dim"]

        num_nodes = cfg["num_nodes"]

        train_num_nodes = int(train_ratio * num_nodes)
        val_num_nodes = int(val_ratio * num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

        data_trimer = DataTrimer(cfg)

        self.fc1 = nn.Linear(history_seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, prediction_seq_len)
        self.act = nn.ReLU()

    def forward(
        self, history_data: torch.Tensor, future_data: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, N, C, L].

        Returns:
            torch.Tensor: outputs with shape [B, N, L]
        """

        history_data = history_data[..., 0, :]  # B, N, L
        prediction = self.fc2(self.act(self.fc1(history_data)))  # B, N, L
        return prediction, None, None, None  # B, L, N
