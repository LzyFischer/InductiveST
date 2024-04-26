import torch
from torch import nn
import torch.nn.functional as F
import pdb
import numpy as np

from torch_geometric.nn import MessagePassing


class CG(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, cfg=None, **model_args):
        super().__init__()
        # attributes

        self.cfg = cfg
        self.node_input_dim = int(cfg["train_node_ratio"] * cfg["num_of_vertices"])

        self.feature_input_dim = self.cfg["input_size"]
        self.time_input_dim = 12

        self.embed_dim_1 = self.feature_input_dim * self.time_input_dim

        self.time_output_dim = 12
        self.feature_output_dim = 1

        self.time_feature_encode_layer_1 = MLPLayer(
            self.feature_input_dim * self.time_input_dim, self.embed_dim_1
        )
        self.classifier = nn.Conv2d(
            in_channels=self.embed_dim_1,
            out_channels=self.node_input_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.domain_feature_encode_layer_1 = MLPLayer(
            self.embed_dim_1, self.embed_dim_1
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        input_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        # input_data = emb[..., range(self.feature_input_dim)]

        (
            batch_size,
            num_nodes,
            feature_dim,
            time_length,
        ) = input_data.shape  # [B, N, C, L]

        # time series embedding
        input_data = input_data.transpose(3, 2).contiguous()
        # input_data: [B, N, L, C]
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        # input_data: [B, L* C, N, 1]
        embed = self.time_feature_encode_layer_1(input_data)

        final_pred = []
        embed = self.domain_feature_encode_layer_1(embed)
        for i in range(embed.shape[-2]):
            final_pred.append(
                self.softmax(
                    self.classifier(embed[..., [i], :]).squeeze(-1).squeeze(-1)
                )
            )
        # final_pred: [B, N, N]
        final_pred = torch.stack(final_pred, dim=-1)

        return final_pred


class MLPLayer(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden
