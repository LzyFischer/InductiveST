import yaml
import torch
import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import pdb
import argparse
from src.loader.data_iterator import DataIterator

from src.trainer.train import Trainer
from src.trainer.train_grad_aug import TrainerGradAug
from src.lib.utils import data_prepare, auto_select_device

from src.model.stgcn_n.stgcn import STGCN_n
from src.model.lstm.lstm import LSTM
from src.model.mlp.mlp import MLP
from src.model.hi.hi import HI
from src.model.nbeats.nbeats import NBeats
from src.model.stgode.stgode import STGODE


import optuna


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def trainer_opt(configs):
    if configs["if_grad_aug"]:
        return TrainerGradAug
    else:
        return Trainer


class Main:
    def __init__(self, configs) -> None:
        # self.model= get_dstagnn_model(self.configs)
        self.configs = configs
        self.device = self.configs["device"]
        # self.initialize_inductive_setting()
        self.model = eval(self.configs["model_name"])(cfg=self.configs)
        self.mode = self.configs["mode"]
        self.trainer = trainer_opt(self.configs)

        (
            self.train_values,
            self.val_values,
            self.test_values,
            self.scaler,
        ) = data_prepare(self.configs)

        self.train_iterator = DataIterator(
            configs=self.configs,
            values=self.train_values,
            scaler=self.scaler,
            mode="train",
        )

        self.val_iterator = DataIterator(
            configs=self.configs,
            values=self.val_values,
            scaler=self.scaler,
            mode="val",
        )

        self.test_iterator = DataIterator(
            configs=self.configs,
            values=self.test_values,
            scaler=self.scaler,
            mode="test",
        )

    def run(self, trial=None):
        if self.mode == "train":
            val_loss = self.train()
        elif self.mode == "eval":
            val_loss = self.eval()
        else:
            raise ValueError("invalid mode")
        return val_loss

    def train(self):
        exit()
        trainer = self.trainer(
            configs=self.configs,
            model=self.model,
            train_iterator=self.train_iterator,
            val_iterator=self.val_iterator,
            test_iterator=self.test_iterator,
        )
        return trainer.train()

    def eval(self):
        evaluator = self.trainer(
            configs=self.configs,
            model=self.model,
            train_iterator=self.train_iterator,
            val_iterator=self.val_iterator,
            test_iterator=self.test_iterator,
            is_eval=True,
        )
        return evaluator.test()

    def initialize_inductive_setting(self):
        self.dataset_name = self.configs["dataset_name"]
        random_seed = self.configs["seed"]
        train_ratio = self.configs["train_node_ratio"]
        val_ratio = self.configs["val_node_ratio"]

        num_nodes = self.configs["num_nodes"]
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

        file_dir = self.configs["processed_root"] + f"{self.dataset_name}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # save train and val nodes with specific name
        train_nodes_path = (
            file_dir
            + f"train_nodes_{random_seed}_{self.configs['train_node_ratio']}.pt"
        )
        val_nodes_path = (
            file_dir + f"val_nodes_{random_seed}_{self.configs['val_node_ratio']}.pt"
        )
        torch.save(train_nodes, train_nodes_path)
        torch.save(val_nodes, val_nodes_path)


# here we use the argparse to get the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--configs_path", type=str, required=False, default="configs/STGCN/PEMS04.yml"
)
parser.add_argument("-m", "--mode", type=str, required=False, default="train")

parser.add_argument("-d", "--device", type=str, required=False, default=None)
parser.add_argument("-s", "--seed", type=int, required=False, default=None)
parser.add_argument(
    "-tr", "--train_node_ratio", type=float, required=False, default=None
)
parser.add_argument("-mn", "--model_name", type=str, required=False, default=None)
parser.add_argument("-win", "--window_size", type=int, required=False, default=None)
parser.add_argument("-ho", "--horizon", type=int, required=False, default=None)
parser.add_argument(
    "-sh", "--sparse_threshold", type=float, required=False, default=None
)
parser.add_argument("-dp", "--dropout", type=float, required=False, default=None)
parser.add_argument(
    "-rg",
    "--random_graph",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="Activate nice mode.",
)
parser.add_argument(
    "-wb",
    "--wandb",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="Activate nice mode.",
)

parser.add_argument("-lr", "--lr", type=float, required=False, default=None)
parser.add_argument("-wd", "--weight_decay", type=float, required=False, default=None)
parser.add_argument("-vv", "--vae_variance", type=float, required=False, default=None)
parser.add_argument(
    "-vw", "--vae_loss_weight", type=float, required=False, default=None
)
parser.add_argument("-ns", "--node_seed", type=int, required=False, default=None)
parser.add_argument("-wn", "--wandb_name", nargs="+", required=False, default=None)
parser.add_argument("-al", "--anchor_lambda", type=float, required=False, default=None)
parser.add_argument(
    "-sl",
    "--sim_loss",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="Similarity loss.",
)
parser.add_argument(
    "-fl",
    "--fst_loss",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="Forecast loss.",
)
parser.add_argument(
    "-ag",
    "--aug_node",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="node augmentation",
)
parser.add_argument(
    "-gl",
    "--graph_learning",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="whether use original graph",
)
parser.add_argument(
    "-ng",
    "--no_graph",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="whether no graph",
)
parser.add_argument(
    "-gs",
    "--gumbel_softmax",
    type=str2bool,
    nargs="?",
    const=True,
    default=None,
    help="whether sim graph",
)


"""arg config"""
args = parser.parse_args()
configs_path = args.configs_path
configs = yaml.safe_load(open(configs_path))
if args.seed is not None:
    configs["seed"] = args.seed
if args.node_seed is not None:
    configs["node_seed"] = args.node_seed
if args.train_node_ratio is not None:
    configs["train_node_ratio"] = args.train_node_ratio
if args.model_name is not None:
    configs["model_name"] = args.model_name
if args.sparse_threshold is not None:
    configs["sparse_threshold"] = args.sparse_threshold
if args.random_graph is not None:
    configs["random_graph"] = args.random_graph
if args.dropout is not None:
    configs["dropout"] = args.dropout
if args.device is not None:
    configs["device"] = args.device
if args.window_size is not None:
    configs["window_size"] = args.window_size
if args.horizon is not None:
    configs["horizon"] = args.horizon
if args.wandb is not None:
    configs["wandb"] = args.wandb
if args.wandb_name is not None:
    configs["wandb_name"] = args.wandb_name
if args.lr is not None:
    configs["lr"] = args.lr
if args.weight_decay is not None:
    configs["weight_decay"] = args.weight_decay
if args.vae_variance is not None:
    configs["vae"]["variance"] = args.vae_variance
if args.vae_loss_weight is not None:
    configs["vae_loss_weight"] = args.vae_loss_weight
if args.anchor_lambda is not None:
    configs["anchor_lambda"] = args.anchor_lambda
if args.sim_loss is not None:
    configs['sim_loss'] = args.sim_loss
if args.fst_loss is not None:
    configs['fst_loss'] = args.fst_loss
if args.aug_node is not None:
    configs['aug_node'] = args.aug_node
if args.graph_learning is not None:
    configs['graph_learning'] = args.graph_learning
if args.no_graph is not None:
    configs['no_graph'] = args.no_graph
if args.gumbel_softmax is not None:
    configs['gumbel_softmax'] = args.gumbel_softmax

configs["mode"] = args.mode
""""""

"""initialze the random seeds"""
if configs["seed"] is not None:
    torch.manual_seed(configs["seed"])
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    torch.backends.cudnn.deterministic = True


def run_grid(trial):
    configs["lr"] = trial.suggest_categorical(
        "lr",
        [
            # 2e-3,
            1e-2,
        ],
    )
    configs["weight_decay"] = trial.suggest_categorical("weight_decay", [1e-3])
    # configs["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])
    main = Main(configs=configs)
    return main.run()


if __name__ == "__main__":
    main = Main(configs=configs)
    main.run()
    # study = optuna.create_study(direction="minimize")
    # study.optimize(run_grid, n_trials=4, n_jobs=-1)
