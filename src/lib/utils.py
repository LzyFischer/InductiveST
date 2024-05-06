import os
import logging

import torch
import networkx as nx

import numpy as np
import pandas as pd
import pickle
import random
from scipy.sparse.linalg import eigs
import subprocess
from frstl import fast_robustSTL

from matplotlib import pyplot as plt
import pdb


def get_gpu_memory_map():
    """Get the current GPU usage."""
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = np.array([int(x) for x in result.strip().split("\n")])
    return gpu_memory


def data_prepare(configs):
    print("preparing data ...")
    data_root = configs["data_root"]
    values_name = configs["values_name"]

    values = np.load(os.path.join(data_root, values_name), allow_pickle=True)["data"]
    values = values.transpose(1, 2, 0)

    split_ratio = configs["split_ratio"]
    train_ratio = split_ratio[0]
    val_ratio = split_ratio[0] + split_ratio[1]

    window_size = configs["window_size"]

    train_nodes = torch.load(
        configs["processed_root"]
        + configs["dataset_name"]
        + "/train_nodes_"
        + str(configs["seed"])
        + "_"
        + str(configs["node_seed"])
        + "_"
        + str(configs["train_node_ratio"])
        + ".pt"
    )
    val_nodes = torch.load(
        configs["processed_root"]
        + configs["dataset_name"]
        + "/val_nodes_"
        + str(configs["seed"])
        + "_"
        + str(configs["node_seed"])
        + "_"
        + str(configs["val_node_ratio"])
        + ".pt"
    )
    train_start_ends = [
        np.arange(i, i + window_size)
        for i in range(0, int(values.shape[-1] * train_ratio) - window_size)
    ]
    val_start_ends = [
        np.arange(i, i + window_size)
        for i in range(
            int(values.shape[-1] * train_ratio) - window_size,
            int(values.shape[-1] * val_ratio) - window_size,
        )
    ]
    test_start_ends = [
        np.arange(i, i + window_size)
        for i in range(
            int(values.shape[-1] * val_ratio) - window_size,
            values.shape[-1] - window_size,
        )
    ]

    # if configs['rstl']:

    train_values = values[train_nodes][..., train_start_ends].astype(np.float32)
    val_values = values[val_nodes][..., val_start_ends].astype(np.float32)
    test_values = values[..., test_start_ends].astype(np.float32)

    # get scalar of the all sets, and will use scalar at the prediction stage
    train_values, val_values, test_values, scaler = normalize(
        train_values, val_values, test_values
    )
    print("data prapare done")
    return (
        train_values,
        val_values,
        test_values,
        scaler,
    )


def normalize(train_values, val_values, test_values):
    """
    normalize data by the mean and std of training set
    normalize by node,time, not feature
    Parameters
    ----------
    train_values: np.ndarray, shape is (batch, node, window, time)
    val_values: np.ndarray, shape is (batch, node, window, time)
    test_values: np.ndarray, shape is (batch, node, window, time)

    """
    mean = np.mean(train_values, axis=(0, 2, 3), keepdims=True)
    std = np.std(train_values, axis=(0, 2, 3), keepdims=True)

    train_values = (train_values - mean) / std
    val_values = (val_values - mean) / std
    test_values = (test_values - mean) / std

    return train_values, val_values, test_values, (mean, std)


def trim_networks(networks, threshold):
    """
    to sparse networks by threshold
    """
    for i in range(len(networks)):
        networks[i][networks[i] < threshold] = 0
    return networks


class DataTrimer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.random_seed = cfg["seed"]
        self.train_ratio = cfg["train_node_ratio"]
        self.val_ratio = cfg["val_node_ratio"]
        self.num_nodes = cfg["num_nodes"]

        self.network_path = cfg["data_root"] + cfg["networks_name"]

    def __call__(self):
        if self.cfg.get("node_seed", None) is not None:
            torch.manual_seed(self.cfg["node_seed"])
            random.seed(self.cfg["node_seed"])
        else:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)

        adj = np.load(self.network_path, allow_pickle=True)
        adj = torch.tensor(adj, dtype=torch.float32)

        train_num_nodes = int(self.train_ratio * self.num_nodes)
        val_num_nodes = int(self.val_ratio * self.num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

        self.raw_data_path = "datasets/raw_data/"
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

        file_dir = self.cfg["processed_root"] + f"{self.dataset_name}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # save train and val nodes with specific name
        train_nodes_path = (
            file_dir
            + f"train_nodes_{self.cfg['seed']}_{self.cfg['node_seed']}_{self.cfg['train_node_ratio']}.pt"
        )
        val_nodes_path = (
            file_dir
            + f"val_nodes_{self.cfg['seed']}_{self.cfg['node_seed']}_{self.cfg['val_node_ratio']}.pt"
        )
        torch.save(train_nodes, train_nodes_path)
        torch.save(val_nodes, val_nodes_path)
        self.train_graph = adj[train_nodes, :][:, train_nodes]
        self.val_graph = adj[val_nodes, :][:, val_nodes]
        self.test_graph = adj[val_nodes, :][:, train_nodes]

        # seed back
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        return (
            [train_num_nodes, val_num_nodes, self.num_nodes],
            [train_nodes, val_nodes],
            [
                self.train_graph,
                self.val_graph,
                self.test_graph,
            ],
        )


class DataTrimer_STGODE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.random_seed = cfg["seed"]
        self.train_ratio = cfg["train_node_ratio"]
        self.val_ratio = cfg["val_node_ratio"]
        self.num_nodes = cfg["num_nodes"]

        A_sp_path = "src/model/stgode/datasets/{0}/sp_matrix_normalized.pkl".format(
            self.dataset_name
        )
        sp = np.load(A_sp_path, allow_pickle=True)
        self.sp = torch.tensor(sp, dtype=torch.float32)
        # self.sp = torch.randn((self.num_nodes, self.num_nodes))

        A_se_path = "src/model/stgode/datasets/{0}/dtw_matrix_normalized.pkl".format(
            self.dataset_name
        )
        se = np.load(A_se_path, allow_pickle=True)
        self.se = torch.tensor(se, dtype=torch.float32)
        # self.se = torch.randn((self.num_nodes, self.num_nodes))

        train_num_nodes = int(self.train_ratio * self.num_nodes)
        val_num_nodes = int(self.val_ratio * self.num_nodes) + train_num_nodes
        self.train_num_nodes = train_num_nodes
        self.val_num_nodes = val_num_nodes

    def __call__(self):
        if self.cfg.get("node_seed", None) is not None:
            torch.manual_seed(self.cfg["node_seed"])
            random.seed(self.cfg["node_seed"])
            np.random.seed(self.cfg["node_seed"])
        else:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.se_edge_index = self.se.to_sparse()._indices().T

        self.sp_edge_index = self.sp.to_sparse()._indices().T

        # # -------------------------------------------------------Graph 1-------------------------------------------------------
        # # generate random index and make sure the graph based on the index is connected
        # G = nx.Graph()
        # G.add_edges_from(self.se_edge_index)

        # # Initialize the connected subgraph with a seed node
        # subgraph = nx.Graph()
        # seed_node = random.choice(list(G.nodes()))
        # subgraph.add_node(seed_node)

        # # Create a set to keep track of visited nodes in the subgraph
        # visited_nodes = set([seed_node])
        # unvisited_nodes = set(G.nodes()) - visited_nodes

        # # While the subgraph has fewer nodes than desired
        # while len(subgraph) < self.val_num_nodes:
        #     if len(subgraph.nodes()) == self.train_num_nodes:
        #         train_nodes = list(subgraph.nodes())
        #     # Get a random node from the subgraph
        #     random_node = random.choice(list(visited_nodes))
        #     # Get the neighbors of the random node in the original graph
        #     neighbors = list(G.neighbors(random_node))

        #     # Filter out neighbors that are already in the subgraph
        #     unvisited_neighbors = [n for n in neighbors if n not in subgraph]

        #     # If there are unvisited neighbors, select one and add it to the subgraph
        #     if unvisited_neighbors:
        #         new_node = random.choice(unvisited_neighbors)
        #         subgraph.add_node(new_node)
        #         subgraph.add_edge(random_node, new_node)
        #         visited_nodes.add(new_node)
        #     else:
        #         try:
        #             random_node = random.choice(list(unvisited_nodes))
        #             visited_nodes.add(random_node)
        #             subgraph.add_node(random_node)
        #             unvisited_nodes.remove(random_node)
        #         except:
        #             print("The situation only happens when full running.")
        #             break

        # val_nodes = list(subgraph.nodes())
        # if len(subgraph.nodes()) == self.train_num_nodes:
        #     train_nodes = list(subgraph.nodes())
        # train_nodes.sort()
        # val_nodes.sort()

        file_dir = self.cfg["processed_root"] + f"{self.dataset_name}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # save train and val nodes with specific name
        # save train and val nodes with specific name
        train_nodes_path = (
            file_dir
            + f"train_nodes_{self.cfg['seed']}_{self.cfg['node_seed']}_{self.cfg['train_node_ratio']}.pt"
        )
        val_nodes_path = (
            file_dir
            + f"val_nodes_{self.cfg['seed']}_{self.cfg['node_seed']}_{self.cfg['val_node_ratio']}.pt"
        )
        # torch.save(train_nodes, train_nodes_path)
        # torch.save(val_nodes, val_nodes_path)
        train_nodes = torch.load(train_nodes_path)
        val_nodes = torch.load(val_nodes_path)

        self.se_train_graph = self.se[train_nodes, :][:, train_nodes]
        self.se_val_graph = self.se[val_nodes, :][:, val_nodes]
        self.se_test_graph = self.se

        # -------------------------------------------------------Graph 2-------------------------------------------------------
        self.sp_train_graph = self.sp[train_nodes, :][:, train_nodes]
        self.sp_val_graph = self.sp[val_nodes, :][:, val_nodes]
        self.sp_test_graph = self.sp

        # seed back
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        return (
            [self.train_num_nodes, self.val_num_nodes, self.num_nodes],
            [train_nodes, val_nodes],
            [
                self.se_train_graph,
                self.se_val_graph,
                self.se_test_graph,
            ],
            [
                self.sp_train_graph,
                self.sp_val_graph,
                self.sp_test_graph,
            ],
        )


def get_metrics_full(y, y_pred):
    """
    compute metrics for one-step prediction

    Parameters
    ----------
    y: np.ndarray, shape is (batch, node, horizon, feature)
    y_pred: np.ndarray, shape is (batch, node, horizon, feature)

    Returns
    ----------
    mae: float
    rmse: float
    mape: float
    """
    mae_12 = torch.mean(torch.abs(y - y_pred))
    mae_6 = torch.mean(torch.abs(y[..., :6] - y_pred[..., :6]))
    mae_3 = torch.mean(torch.abs(y[..., :3] - y_pred[..., :3]))
    rmse_12 = torch.sqrt(torch.mean(torch.square(y - y_pred)))
    rmse_6 = torch.sqrt(torch.mean(torch.square(y[..., :6] - y_pred[..., :6])))
    rmse_3 = torch.sqrt(torch.mean(torch.square(y[..., :3] - y_pred[..., :3])))
    # masked mape
    mask = y >= 1
    mape_12 = torch.mean(torch.abs((y - y_pred) / y)[mask])
    mape_6 = torch.mean(
        torch.abs((y[..., :6] - y_pred[..., :6]) / y[..., :6])[mask[..., :6]]
    )
    mape_3 = torch.mean(
        torch.abs((y[..., :3] - y_pred[..., :3]) / y[..., :3])[mask[..., :3]]
    )
    print(
        "mae_12: {:.4f}, rmse_12: {:.4f}, mape_12: {:.4f}".format(
            mae_12, rmse_12, mape_12
        )
    )
    print("mae_6: {:.4f}, rmse_6: {:.4f}, mape_6: {:.4f}".format(mae_6, rmse_6, mape_6))
    print("mae_3: {:.4f}, rmse_3: {:.4f}, mape_3: {:.4f}".format(mae_3, rmse_3, mape_3))
    return mae_12, rmse_12, mape_12, mae_6, rmse_6, mape_6, mae_3, rmse_3, mape_3


def get_metrics(y, y_pred):
    """
    compute metrics for one-step prediction

    Parameters
    ----------
    y: np.ndarray, shape is (batch, node, horizon, feature)
    y_pred: np.ndarray, shape is (batch, node, horizon, feature)

    Returns
    ----------
    mae: float
    rmse: float
    mape: float
    """
    mae = torch.mean(torch.abs(y - y_pred))
    rmse = torch.sqrt(torch.mean(torch.square(y - y_pred)))
    # masked mape
    mask = y >= 1
    mape = torch.mean(torch.abs((y - y_pred) / y)[mask])
    print("mae: {:.4f}, rmse: {:.4f}, mape: {:.4f}".format(mae, rmse, mape))
    return mae, rmse, mape


def auto_select_device():
    if torch.cuda.is_available():
        # argmin available memory
        gpu_memory = get_gpu_memory_map()
        device = torch.device(f"cuda:{np.argmin(gpu_memory)}")

    else:
        device = torch.device("cpu")
    return device
