import torch
from tqdm import tqdm
import os
import numpy as np
import logging
from ..loader.data_iterator import DataIterator
from ..lib.utils import get_metrics, get_metrics_full
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
# from tensorboardX import SummaryWriter
from src.model.stgcn_n.stgcn import STGCN_n
import pdb
import copy
import wandb


# writer = SummaryWriter(flush_secs=5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_values(configs, key):
    if "." in key:
        keys = key.split(".")
        value = configs
        for k in keys:
            value = value[k]
    else:
        value = configs[key]
    return value


class Trainer:
    def __init__(
        self,
        configs,
        model: STGCN_n,
        train_iterator,
        val_iterator,
        test_iterator,
        is_eval=False,
    ):
        if configs.get("wandb", True):
            # str join
            try:
                name = ".".join(
                    [f"{k}_{get_values(configs, k)}" for k in configs["wandb_name"]]
                )
            except:
                name = f"model_name_{configs['model_name']}.dataset_name_{configs['dataset_name']}"
            self.run = wandb.init(
                project=configs.get("wandb_project", "InductiveST"),
                config=configs,
                name=name,
            )
        else:
            self.run = wandb.init(
                project=configs.get("wandb_project", "InductiveST"),
                config=configs,
                mode="disabled",
            )

        self.configs = configs
        self.device = configs["device"]
        self.model = model
        self.model = self.model.to(self.device)
        self.horizon = configs["horizon"]
        self.window_size = configs["window_size"]

        self.scaler = train_iterator.get_scaler().to(self.device)

        if self.configs["optimizer"] == "Adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=configs["lr"],
                weight_decay=configs["weight_decay"],
            )
        elif self.configs["optimizer"] == "SGD":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=configs["lr"],
                momentum=0.9,
                weight_decay=configs["weight_decay"],
            )
        else:
            raise ValueError("Optimizer not implemented")

        self.scheduler = lr_scheduler.MultiStepLR(
            self.opt, milestones=self.configs["milestones"], gamma=self.configs["gamma"]
        )

        self.train_iterator = train_iterator.get_loader()
        self.val_iterator = val_iterator.get_loader()
        self.test_iterator = test_iterator.get_loader()
        self.loss_min = np.inf
        self.best_test_mae = np.inf
        self.n_tolerance = 0

        if not os.path.exists(self.configs["save_dir"]):
            os.makedirs(self.configs["save_dir"])

        if is_eval:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.configs["save_dir"], self.configs["model_file"])
                )
            )

    def train(self):
        for epoch in range(self.configs["epochs"]):
            self.aug_node_epoch = self.configs["aug_node"] and epoch % 2 == 0
            self.aug_node_n_epoch = self.configs["aug_node"] and (epoch % 2 == 1)
            train_loss = self._train_epoch(epoch)
            if (self.configs["is_vae"] and epoch < self.configs["vae_epochs"]) or (
                self.aug_node_epoch
            ):
                wandb.log({"aug_loss": train_loss}, commit=False)
                continue
            if self.configs["model_name"] != "HI":
                self.scheduler.step()
            eval_loss = self.eval(epoch)
            test_loss, metrics = self.test()
            if eval_loss < self.loss_min:
                self.loss_min = eval_loss
                self.best_test = test_loss
                self.best_metrics = metrics
                self.n_tolerance = 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            else:
                self.n_tolerance += 1
                logger.info(
                    f"Early stopping count: {self.n_tolerance} / {self.configs['tolerance']}"
                )
                if self.n_tolerance >= self.configs["tolerance"]:
                    logger.info("Early stopping at epoch {}".format(epoch))
                    break
            wandb.log(
                {
                    "loss": {
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "test_loss": test_loss,
                        "best_test_loss": self.best_test,
                    },
                    "metrics": {
                        "best_mae_12": self.best_metrics[0],
                        "best_rmse_12": self.best_metrics[1],
                        "best_mape_12": self.best_metrics[2],
                        "best_mae_6": self.best_metrics[3],
                        "best_rmse_6": self.best_metrics[4],
                        "best_mape_6": self.best_metrics[5],
                        "best_mae_3": self.best_metrics[6],
                        "best_rmse_3": self.best_metrics[7],
                        "best_mape_3": self.best_metrics[8],
                    },
                }
            )
        logger.info("Training finished")

        self.model.load_state_dict(self.best_model)
        test_loss, metrics = self.test()
        wandb.log(
            {
                "final_metrics": f"{metrics[6]:.2f} {metrics[7]:.2f} {metrics[8]*100:.2f} {metrics[3]:.2f} {metrics[4]:.2f} {metrics[5]*100:.2f} {metrics[0]:.2f} {metrics[1]:.2f} {metrics[2]*100:.2f}"
            }
        )
        print(
            f"{metrics[6]:.2f} {metrics[7]:.2f} {metrics[8]*100:.2f} {metrics[3]:.2f} {metrics[4]:.2f} {metrics[5]*100:.2f} {metrics[0]:.2f} {metrics[1]:.2f} {metrics[2]*100:.2f}"
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.configs["save_dir"], f"model_{test_loss}.pt"),
        )
        return test_loss

    def _train_epoch(self, epoch):
        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        epoch_loss = 0
        n_iter = 0
        for values in tqdm(self.train_iterator, ncols=0, leave=False):
            values = values.to(self.device)

            self.opt.zero_grad()
            y_pred, future_data, embedding, vae_loss = self._iter(
                values=values, epoch=epoch
            )
            y = values[..., -self.horizon :]

            if self.configs["aug_node"]:
                if self.aug_node_epoch:
                    y = future_data
                else:
                    y_pred = y_pred[:, : values.shape[1]]

            iter_loss = 0

            if self.configs["embedding_loss"]:
                embedding_loss = []
                for i in range(1, self.horizon):
                    embedding_loss.append(torch.var(embedding[..., i], dim=1).mean())
                embedding_loss = torch.stack(embedding_loss).mean()
                iter_loss += 1 * embedding_loss

            if self.configs["balanced_loss"]:
                balanced_loss = torch.var(
                    self.get_each_series_mae(y=y[..., 0, :], y_pred=y_pred), dim=-1
                )
                balanced_loss = torch.mean(balanced_loss)
                iter_loss += 1 * balanced_loss

            if not self.configs["train_normalized"]:
                y = y * self.scaler[1][..., :] + self.scaler[0][..., :]
                y_pred = y_pred * self.scaler[1][..., 0, :] + self.scaler[0][..., 0, :]

            # random mask the nodes

            if self.configs["mask_node_loss"]:
                if self.aug_node_epoch:
                    pass
                else:
                    mask = torch.randperm(y.shape[1])
                    mask = mask[
                        : int(y.shape[1] * (1 - self.configs["mask_node_ratio"]))
                    ]
                    mask = mask.to(self.device)
                    y = y[:, mask]
                    y_pred = y_pred[:, mask]

            if self.aug_node_epoch:
                iter_loss = 0.01 * vae_loss
                # fix all of the model
                for name, param in self.model.named_parameters():
                    if "vae" not in name:
                        param.requires_grad = False

                y = y[:, values.shape[1] :]
                y_pred = y_pred[:, values.shape[1] :]
                if self.configs.get('fst_loss', False):
                    iter_loss += self.get_mae(y=y[..., 0, :], y_pred=y_pred)
                # regularization on vae loss
                iter_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

                n_iter += 1
                epoch_loss += iter_loss.item()
                continue
            elif self.aug_node_n_epoch:
                for name, param in self.model.named_parameters():
                    if "vae" in name:
                        param.requires_grad = False
            iter_loss += self.get_mae(y=y[..., 0, :], y_pred=y_pred)

            if self.configs["model_name"] != "HI":
                iter_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

            n_iter += 1
            epoch_loss += iter_loss.item()
        epoch_loss /= n_iter
        # writer.add_scalar("Loss/train", epoch_loss, epoch)
        logger.info("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
        # wandb.log({"train_loss": epoch_loss})
        return epoch_loss

    def _iter(self, values, epoch=0):
        embedding = None
        future_data = None
        vae_loss = 0
        if True:
            try:
                (y_pred, future_data, embedding, vae_loss) = self.model(
                    values[..., : -self.horizon], values[..., -self.horizon :]
                )

            except:
                raise ValueError("Embedding loss is not implemented for this model")
        elif self.configs["is_vae"] and not self.configs["embedding_loss"]:
            # try:
            (y_pred, future_data, embedding, vae_loss) = self.model(
                values[..., : -self.horizon], values[..., -self.horizon :]
            )
            # except:
            #     raise ValueError("VAE loss is not implemented for this model")
        elif self.configs["is_vae"] and self.configs["embedding_loss"]:
            try:
                (y_pred, future_data, embedding, vae_loss) = self.model(
                    values[..., : -self.horizon], values[..., -self.horizon :]
                )
            except:
                raise ValueError(
                    "VAE and embedding loss are not implemented for this model"
                )
        else:
            try:
                (y_pred, future_data, embedding, vae_loss) = self.model(
                    values[..., : -self.horizon], values[..., -self.horizon :]
                )
            except:
                raise ValueError(
                    "VAE and embedding loss are not implemented for this model"
                )

        return (
            y_pred,
            future_data,
            embedding,
            vae_loss,
        )

    def eval(self, epoch):
        self.model.eval()
        y_preds = []
        y_trues = []
        for values in tqdm(self.val_iterator, ncols=0, leave=False):
            values = values.to(self.device)
            y_pred, _, _, _ = self._iter(values=values)
            torch.cuda.empty_cache()
            y_preds.append(y_pred.detach().cpu())
            y_trues.append(values[..., -self.horizon :].detach().cpu())
        y_preds = torch.cat(y_preds, axis=0)
        y_trues = torch.cat(y_trues, axis=0)
        y_preds = (
            y_preds * self.scaler[1][..., 0, :].cpu() + self.scaler[0][..., 0, :].cpu()
        )
        y_trues = y_trues * self.scaler[1].cpu() + self.scaler[0].cpu()
        loss = self.get_mae(y=y_trues[..., 0, :], y_pred=y_preds)
        logger.info("Eval loss: {}".format(loss))
        get_metrics(y_trues[..., 0, :], y_preds)
        # writer.add_scalar("Loss/eval", loss, epoch)
        return loss

    def test(self):
        self.model.eval()
        y_preds = []
        y_trues = []
        for values in tqdm(self.test_iterator, ncols=0, leave=False):
            values = values.to(self.device)
            y_pred, _, _, _ = self._iter(values=values)
            torch.cuda.empty_cache()
            y_preds.append(y_pred.detach().cpu())
            y_trues.append(values[..., -self.horizon :].detach().cpu())
        y_preds = torch.cat(y_preds, axis=0)
        y_trues = torch.cat(y_trues, axis=0)
        y_preds = (
            y_preds * self.scaler[1][..., 0, :].cpu() + self.scaler[0][..., 0, :].cpu()
        )
        y_trues = y_trues * self.scaler[1].cpu() + self.scaler[0].cpu()

        loss = self.get_mae(y=y_trues[..., 0, :], y_pred=y_preds)
        logger.info("Test loss: {}".format(loss))
        if self.configs.get("full_metrics", False):
            metrics = get_metrics_full(y_trues[..., 0, :], y_preds)
        else:
            metrics = get_metrics(y_trues[..., 0, :], y_preds)

        return loss, metrics

    def get_mse(self, y, y_pred):
        return self.mse(y=y, y_pred=y_pred)

    def get_mae(self, y, y_pred):
        loss = torch.abs(y - y_pred)
        loss = torch.sum(loss) / y.numel()
        return loss

    def get_each_series_mae(self, y, y_pred):
        loss = torch.abs(y - y_pred)
        loss = torch.mean(loss, axis=[0, 2])
        return loss

    def get_rmse(self, y, y_pred):
        return torch.sqrt(self.mse(y=y, y_pred=y_pred))

    def mse(self, y, y_pred):
        # calculate mse
        loss = (y - y_pred) ** 2
        # mask loss
        loss = torch.sum(loss) / y.numel()
        return loss
