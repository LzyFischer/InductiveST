import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    scatter,
)

import pdb


class SimLearner(nn.Module):
    def __init__(self, cfg):
        super(SimLearner, self).__init__()
        self.cfg = cfg
        self.fc = nn.Linear(
            cfg["input_size"] * (cfg["window_size"] - cfg["horizon"]),
            cfg["graph"]["hidden_dim"],
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.cat_linear = nn.Linear(cfg["graph"]["hidden_dim"] * 2, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[-1] * x.shape[-2])
        x = self.fc(x)
        # normalize x
        x = F.normalize(x, p=2, dim=2)

        if self.cfg["graph"]["node2edge_type"] == "cat":
            x_expand = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            x_expand_t = x_expand.transpose(1, 2)
            pre_similarity_matrix = torch.cat([x_expand, x_expand_t], dim=-1)
            similarity_matrix = self.tanh(
                self.cat_linear(pre_similarity_matrix)
            ).squeeze(-1)
        elif self.cfg["graph"]["node2edge_type"] == "cosine":
            similarity_matrix = torch.bmm(x, x.transpose(1, 2))

        # with threshold
        if not self.cfg["dynamic_graph"]:
            similarity_matrix = torch.mean(similarity_matrix, dim=0)

            if self.cfg["random_graph"]:
                similarity_matrix = torch.randn(similarity_matrix.shape).to(
                    similarity_matrix.device
                )
            # similarity_matrix = torch.ones(similarity_matrix.shape).to(
            #     similarity_matrix.device
            # )

        if self.cfg["gumbel_softmax"]:
            if self.cfg["sparse_threshold"] != 0:
                # count how many elements are larger than threshold
                if self.cfg["aug_node"] == True:
                    aug_sim_matrix = (similarity_matrix - similarity_matrix.min()) / (
                        similarity_matrix.max() - similarity_matrix.min()
                    )
                    similarity_matrix = similarity_matrix - self.cfg["sparse_threshold"]

                similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
                    similarity_matrix.max() - similarity_matrix.min()
                )
                similarity_matrix = similarity_matrix - self.cfg["sparse_threshold"]

            tmp_matrix = -(similarity_matrix)

            similarity_matrix = torch.exp(
                torch.stack([similarity_matrix, tmp_matrix], dim=-1) * 10
            )
            test = similarity_matrix
            similarity_matrix = F.gumbel_softmax(
                similarity_matrix, tau=self.cfg["gumbel_tau"], hard=True
            )[..., 0]

        # plt.imshow(similarity_matrix.cpu().detach().numpy())
        # plt.savefig("similarity_matrix.png")

        return similarity_matrix

    def normalize_adjacency_matrices(self, adj_matrices):
        # get degree matrix
        if self.cfg["dynamic_graph"] == False:
            adj_matrices = adj_matrices + torch.eye(adj_matrices.shape[1]).to(
                adj_matrices.device
            )
            degree_matrix = torch.sum(adj_matrices, dim=1)
            degree_matrix = torch.diag_embed(degree_matrix)

            degree_matrix = torch.inverse(degree_matrix)
            normalized_adj_matrices = torch.mm(degree_matrix, adj_matrices)

        else:
            adj_matrices = adj_matrices + torch.eye(adj_matrices.shape[1]).to(
                adj_matrices.device
            ).unsqueeze(0).repeat(adj_matrices.shape[0], 1, 1)
            degree_matrix = torch.sum(adj_matrices, dim=2)
            degree_matrix = torch.diag_embed(degree_matrix)

            degree_matrix = torch.inverse(degree_matrix)
            normalized_adj_matrices = torch.bmm(degree_matrix, adj_matrices)
        return normalized_adj_matrices
