import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import networkx as nx
import pandas as pd
import pdb

from .dcrnn import DCRNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, num_node_list, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get("K", 2))
        # self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size_train = num_node_list[0] * self.rnn_units
        self.hidden_state_size_val = num_node_list[1] * self.rnn_units
        self.hidden_state_size_test = num_node_list[2] * self.rnn_units
        self.train_num_nodes = num_node_list[0]
        self.val_num_nodes = num_node_list[1]
        self.test_num_nodes = num_node_list[2]


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, num_node_list, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, num_node_list, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(
            model_kwargs.get("window_size", 1) - model_kwargs.get("horizon", 1)
        )
        self.dcgru_layers = nn.ModuleList(
            [
                DCRNN(self.rnn_units, adj_mx, self.max_diffusion_step, num_node_list)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            if self.training:
                hidden_state = torch.zeros(
                    (self.num_rnn_layers, batch_size, self.hidden_state_size_train)
                ).to(inputs.device)
            elif self.val_num_nodes == inputs.size(1):
                hidden_state = torch.zeros(
                    (self.num_rnn_layers, batch_size, self.hidden_state_size_val)
                ).to(inputs.device)
            else:
                hidden_state = torch.zeros(
                    (self.num_rnn_layers, batch_size, self.hidden_state_size_test)
                ).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, num_node_list, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, num_node_list, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                DCRNN(self.rnn_units, adj_mx, self.max_diffusion_step, num_node_list)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        if self.training:
            output = projected.view(-1, self.train_num_nodes * self.output_dim)
        elif self.val_num_nodes == inputs.size(1):
            output = projected.view(-1, self.val_num_nodes * self.output_dim)
        else:
            output = projected.view(-1, self.test_num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, cfg):
        super(DCRNNModel, self).__init__()

        network_path = cfg["data_root"] + cfg["networks_name"]
        adj = np.load(network_path, allow_pickle=True)
        adj = torch.tensor(adj, dtype=torch.float32)

        self.dataset_name = cfg["dataset_name"]
        random_seed = cfg["seed"]
        train_ratio = cfg["train_node_ratio"]
        val_ratio = cfg["val_node_ratio"]

        num_nodes = cfg["num_nodes"]

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        state = random.getstate()

        train_num_nodes = int(train_ratio * num_nodes)
        val_num_nodes = int(val_ratio * num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

        self.raw_data_path = (
            "/home/zhenyu/program/TSF/framework/ZeroTS/datasets/raw_data/"
        )
        self.raw_graph_path = (
            self.raw_data_path + f"{self.dataset_name}/{self.dataset_name}.csv"
        )
        self.raw_graph = pd.read_csv(self.raw_graph_path, header=0).astype(int)
        self.raw_graph = self.raw_graph.iloc[:, :2].values

        # generate random index and make sure the graph based on the index is connected
        G = nx.Graph()
        G.add_edges_from(self.raw_graph)

        # Initialize the connected subgraph with a seed node
        subgraph = nx.Graph()
        seed_node = random.choice(list(G.nodes()))
        subgraph.add_node(seed_node)

        # Create a set to keep track of visited nodes in the subgraph
        visited_nodes = set([seed_node])
        unvisited_nodes = set(G.nodes()) - visited_nodes

        # While the subgraph has fewer nodes than desired
        while len(subgraph) < val_num_nodes:
            if len(subgraph.nodes()) == train_num_nodes:
                train_nodes = list(subgraph.nodes())
            # Get a random node from the subgraph
            random_node = random.choice(list(visited_nodes))
            # Get the neighbors of the random node in the original graph
            neighbors = list(G.neighbors(random_node))

            # Filter out neighbors that are already in the subgraph
            unvisited_neighbors = [n for n in neighbors if n not in subgraph]

            # If there are unvisited neighbors, select one and add it to the subgraph
            if unvisited_neighbors:
                new_node = random.choice(unvisited_neighbors)
                subgraph.add_node(new_node)
                subgraph.add_edge(random_node, new_node)
                visited_nodes.add(new_node)
            else:
                try:
                    random_node = random.choice(list(unvisited_nodes))
                    visited_nodes.add(random_node)
                    subgraph.add_node(random_node)
                    unvisited_nodes.remove(random_node)
                except:
                    print("The situation only happens when full running.")
                    break

        val_nodes = list(subgraph.nodes())
        if len(subgraph.nodes()) == train_num_nodes:
            train_nodes = list(subgraph.nodes())
        train_nodes.sort()
        val_nodes.sort()

        file_dir = cfg["processed_root"] + f"{self.dataset_name}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # save train and val nodes with specific name
        train_nodes_path = (
            file_dir + f"train_nodes_{random_seed}_{cfg['train_node_ratio']}.pt"
        )
        val_nodes_path = (
            file_dir + f"val_nodes_{random_seed}_{cfg['val_node_ratio']}.pt"
        )
        torch.save(train_nodes, train_nodes_path)
        torch.save(val_nodes, val_nodes_path)
        self.train_graph = adj[train_nodes, :][:, train_nodes]
        self.val_graph = adj[val_nodes, :][:, val_nodes]
        self.test_graph = adj[val_nodes, :][:, train_nodes]

        # to index the graph
        self.train_graph_id = self.train_graph.to_sparse()._indices()
        self.val_graph_id = self.val_graph.to_sparse()._indices()
        self.test_graph_id = self.test_graph.to_sparse()._indices()

        Seq2SeqAttrs.__init__(
            self,
            [self.train_num_nodes, self.val_num_nodes, num_nodes],
            [self.train_graph, self.val_graph, self.test_graph],
            **cfg["model"],
        )

        self.encoder_model = EncoderModel(
            [self.train_num_nodes, self.val_num_nodes, num_nodes],
            [self.train_graph, self.val_graph, self.test_graph],
            **cfg["model"],
        )
        self.decoder_model = DecoderModel(
            [self.train_num_nodes, self.val_num_nodes, num_nodes],
            [self.train_graph, self.val_graph, self.test_graph],
            **cfg["model"],
        )

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state
            )

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state):
        batch_size = encoder_hidden_state.size(1)
        if self.training:
            go_symbol = torch.zeros(
                (batch_size, self.train_num_nodes * self.decoder_model.output_dim)
            ).to(encoder_hidden_state.device)
        elif self.val_num_nodes == encoder_hidden_state.size(1):
            go_symbol = torch.zeros(
                (batch_size, self.val_num_nodes * self.decoder_model.output_dim)
            ).to(encoder_hidden_state.device)
        else:
            go_symbol = torch.zeros(
                (batch_size, self.test_num_nodes * self.decoder_model.output_dim)
            ).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs

    def forward(
        self,
        history_data: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]

            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction wit shape [L, B, N*C_out]
        """
        # history_data: [B, N, C, L]
        history_data = history_data.permute(0, 3, 1, 2)
        # reshape data
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(
            batch_size, length, num_nodes * channels
        )  # [B, L, N*C]
        history_data = history_data.transpose(0, 1)  # [L, B, N*C]

        # DCRNN
        encoder_hidden_state = self.encoder(history_data)
        outputs = self.decoder(encoder_hidden_state)  # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, num_nodes, self.decoder_model.output_dim)
        outputs = outputs.squeeze(-1).transpose(1, 2)  # [B, N, L]
        return outputs
