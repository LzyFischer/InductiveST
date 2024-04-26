import torch
import torch.nn as nn
import pdb
import os
import random
import networkx as nx
import pandas as pd
import numpy as np

from ..lib.utils import trim_networks
from .stgcn_layers_lg import STConvBlock, OutputBlock


class STGCN_LG(nn.Module):
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
        super(STGCN_LG, self).__init__()
        modules = []

        self.dataset_name = cfg["dataset_name"]
        random_seed = cfg["seed"]
        train_ratio = cfg["train_node_ratio"]
        val_ratio = cfg["val_node_ratio"]

        network_path = cfg["data_root"] + cfg["networks_name"]
        gso = np.load(network_path, allow_pickle=True)
        gso = torch.tensor(gso, dtype=torch.float32)

        Kt = cfg["stgcn"]["Kt"]
        Ks = cfg["stgcn"]["Ks"]
        blocks = cfg["stgcn"]["blocks"]
        T = cfg["stgcn"]["T"]
        n_vertex = cfg["stgcn"]["n_vertex"]
        act_func = cfg["stgcn"]["act_func"]
        graph_conv_type = cfg["stgcn"]["graph_conv_type"]
        bias = cfg["stgcn"]["bias"]
        droprate = cfg["stgcn"]["droprate"]

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        state = random.getstate()

        num_nodes = n_vertex
        self.full_nodes = num_nodes
        train_num_nodes = int(train_ratio * num_nodes)
        val_num_nodes = int(val_ratio * num_nodes)
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes + train_num_nodes

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

        while len(subgraph) < train_num_nodes + val_num_nodes:
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
        gso_train = gso[train_nodes, :][:, train_nodes]
        gso_val = gso[val_nodes, :][:, val_nodes]

        """===================================================================== below is the same as stgcn.py =====================================================================""" ""
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(
                    cfg,
                    Kt,  # temporal convolution kernel 3
                    Ks,  # spatial convolution kernel 3
                    n_vertex,
                    blocks[l][-1],  # output size for last layer 64
                    blocks[l + 1],  # size for this layer
                    act_func,  # relu
                    graph_conv_type,  # cheb?
                    gso,  # graph structure
                    gso_train,
                    gso_val,
                    self.train_num_nodes,
                    self.val_num_nodes,
                    bias,
                    droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (
            Kt - 1
        )  # each_layer_reduction = layer_num * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(
            cfg,
            Ko,
            blocks[-3][-1],  # last working layer
            blocks[-2],  # linear layer
            blocks[-1][0],  # output time step
            n_vertex,
            self.train_num_nodes,
            self.val_num_nodes,
            act_func,
            bias,
            droprate,
        )

        self.graph_generator = Graph(cfg)

        self.eps_f_learnable = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        history_data: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """feedforward function of STGCN.

        Args:
            history_data (torch.Tensor): historical data with shape [B, C, L, N]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        gso = self.graph_generator(history_data)

        x = history_data.permute(0, 2, 3, 1).contiguous()[:, [0], ...]  # [B, C, L, N]
        x, gso = self.st_blocks((x, gso))
        x = self.output(x)
        x = x.permute(0, 3, 1, 2).squeeze(-1)  # [B,L,C,N]->[B,N,L,C]->[B,N,L]

        return x


class Graph(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(Graph, self).__init__()
        self.input_dim = cfg["input_size"]
        self.input_length = cfg["window_size"] - cfg["horizon"]
        self.dim = self.input_dim * self.input_length
        self.MLP = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )
        # zero initialize the weight of MLP
        nn.init.zeros_(self.MLP[0].weight)
        nn.init.zeros_(self.MLP[2].weight)

    def forward(self, x):
        # x: [B, C, L, N)
        x = x.permute(0, 3, 2, 1).contiguous().view(x.shape[0], x.shape[1], -1)
        # x: [B, N, L* C)
        x = self.MLP(x)
        adj = torch.einsum("bik,bjk->bij", x, x)
        return adj
