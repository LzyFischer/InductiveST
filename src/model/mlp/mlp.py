import torch
from torch import nn
import numpy as np

from ..vae.vae import VAE
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
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, N, C, L].

        Returns:
            torch.Tensor: outputs with shape [B, N, L]
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

        history_data = history_data[..., 0, :]  # B, N, L
        prediction = self.fc2(self.act(self.fc1(history_data)))  # B, N, L
        return prediction, future_data, None, vae_loss  # B, L, N
