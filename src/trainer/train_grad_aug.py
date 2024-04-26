import torch
from tqdm import tqdm
import os
import numpy as np
import logging
from ..loader.data_iterator import DataIterator
from ..lib.utils import get_metrics
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pdb
import copy

from ..model.stgcn_lg import STGCN_LG

writer = SummaryWriter(flush_secs=5)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainerGradAug:
    def __init__(
        self,
        configs,
        model: STGCN_LG,
        train_iterator,
        val_iterator,
        test_iterator,
        is_eval=False,
    ):
        self.configs = configs
        self.device = configs["device"]
        self.model = model
        self.model = self.model.to(self.device)
        self.horizon = configs["horizon"]
        self.window_size = configs["window_size"]
        self.aug_model = eval(configs["aug_model_name"])(cfg=configs).to(self.device)

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

        # augmentation model
        self.aug_opt = torch.optim.Adam(
            self.aug_model.parameters(),
            lr=configs["lr"],
            weight_decay=configs["weight_decay"],
        )
        self.aug_scheduler = lr_scheduler.MultiStepLR(
            self.aug_opt,
            milestones=self.configs["milestones"],
            gamma=self.configs["gamma"],
        )

        self.train_iterator = train_iterator.get_loader()
        self.val_iterator = val_iterator.get_loader()
        self.test_iterator = test_iterator.get_loader()
        self.loss_min = np.inf
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
            self._train_epoch(epoch)
            self.scheduler.step()
            eval_loss = self.eval(epoch)
            test_loss = self.test()
            if eval_loss < self.loss_min:
                self.loss_min = eval_loss
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

        logger.info("Training finished")

        self.model.load_state_dict(self.best_model)
        val_loss = self.eval(epoch)
        test_loss = self.test()
        torch.save(
            self.best_model,
            os.path.join(self.configs["save_dir"], f"model_{test_loss}_{val_loss}.pt"),
        )
        return val_loss

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        n_iter = 0
        for values in tqdm(self.train_iterator):
            values = values.to(self.device)

            history_data = values[..., : -self.horizon]
            domain_input = copy.deepcopy(history_data)  # for domain input
            y = values[..., -self.horizon :]
            batch, num_nodes, feature, length = history_data.shape
            # generate domain lable on the fly
            domain_label = torch.zeros(batch, num_nodes, num_nodes)
            for i in range(domain_label.shape[-1]):
                domain_label[:, i, i] = 1
            domain_label = domain_label.to(values.device)

            # generate domain pertubation
            y_pertubated = y.expand(history_data.shape)
            final_input = []
            final_output = []
            for i in range(self.configs["grad_iter_num"]):
                history_data.requires_grad = True
                y_pertubated.requires_grad = True
                loss_d = F.cross_entropy(self.aug_model(history_data), domain_label)
                loss_dv = F.cross_entropy(self.aug_model(y_pertubated), domain_label)
                loss_d.backward()
                loss_dv.backward()
                grad_d = torch.clamp(history_data.grad, min=-0.1, max=0.1)
                grad_dv = torch.clamp(y_pertubated.grad, min=-0.1, max=0.1)

                # eps_f_learnable = self.model.eps_f_learnable
                # history_data_grad = (
                #     history_data.data + eps_f_learnable * grad_d * self.configs["eps_f"]
                # )
                # y_pertubated_grad = (
                #     y_pertubated.data
                #     + eps_f_learnable * grad_dv * self.configs["eps_f"]
                # )

                # history_data = (
                #     history_data.data
                #     + eps_f_learnable.data * grad_d * self.configs["eps_f"]
                # )
                # y_pertubated = (
                #     y_pertubated.data
                #     + eps_f_learnable.data * grad_dv * self.configs["eps_f"]
                # )

                # final_input.append(history_data_grad)
                # final_output.append(y_pertubated_grad)
                ###
                history_data = history_data.data + self.configs["eps_f"] * grad_d
                y_pertubated = y_pertubated.data + self.configs["eps_f"] * grad_dv

                self.aug_model.zero_grad()

                final_input.append(history_data)
                final_output.append(y_pertubated)

            final_input = torch.stack(final_input, dim=0)
            final_output = torch.stack(final_output, dim=0)
            final_input = final_input.permute(1, 4, 0, 2, 3)
            final_input = final_input.reshape(
                final_input.shape[0], final_input.shape[1], -1, final_input.shape[-1]
            )
            final_output = final_output.permute(1, 4, 0, 2, 3)
            final_output = final_output.reshape(
                final_output.shape[0],
                final_output.shape[1],
                -1,
                final_output.shape[-1],
            )

            final_input = final_input.permute(0, 2, 3, 1)
            final_output = final_output.permute(0, 2, 3, 1)

            #### update part ####
            self.opt.zero_grad()
            y_pred = self._iter(history_data=final_input, epoch=epoch)
            y = final_output * self.scaler[1][..., :] + self.scaler[0][..., :]
            y_pred = y_pred * self.scaler[1][..., 0, :] + self.scaler[0][..., 0, :]

            iter_loss = self.get_mae(y=y[..., 0, :], y_pred=y_pred)
            iter_loss.backward()
            self.opt.step()
            self.scheduler.step()

            n_iter += 1
            epoch_loss += iter_loss.item()

            #### update augment model ####
            # if epoch == 1:
            # pdb.set_trace()
            # for i in range(self.configs["grad_iter_num"]):
            #     history_data.requires_grad = True
            #     loss_l = self.get_mae(y, self._iter(history_data))
            #     loss_l.backward()
            #     grad_d = torch.clamp(history_data.grad, min=-0.1, max=0.1)
            #     grad_dv = torch.clamp(y_pertubated.grad, min=-0.1, max=0.1)
            #     history_data = history_data.data + self.configs["eps_f"] * grad_d
            #     y_pertubated = y_pertubated.data + self.configs["eps_f"] * grad_dv
            #     final_input.append(history_data)

            self.aug_opt.zero_grad()
            loss_d = F.cross_entropy(self.aug_model(history_data), domain_label)
            loss_d.backward()
            self.aug_opt.step()
            self.aug_scheduler.step()

        epoch_loss /= n_iter
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        logger.info("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
        return epoch_loss

    def _iter(self, history_data, epoch=0):
        y_pred = self.model(history_data, epoch=epoch)
        return y_pred

    def eval(self, epoch):
        self.model.eval()
        y_preds = []
        y_trues = []
        for values in tqdm(self.val_iterator):
            values = values.to(self.device)
            y_pred = self._iter(history_data=values[..., : -self.horizon])
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
        writer.add_scalar("Loss/eval", loss, epoch)
        return loss

    def test(self):
        self.model.eval()
        y_preds = []
        y_trues = []
        for values in tqdm(self.test_iterator):
            values = values.to(self.device)
            y_pred = self._iter(history_data=values[..., : -self.horizon])
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
        get_metrics(y_trues[..., 0, :], y_preds)

        return loss

    def get_mse(self, y, y_pred):
        return self.mse(y=y, y_pred=y_pred)

    def get_mae(self, y, y_pred):
        loss = torch.abs(y - y_pred)
        loss = torch.sum(loss) / y.numel()
        return loss

    def get_rmse(self, y, y_pred):
        return torch.sqrt(self.mse(y=y, y_pred=y_pred))

    def mse(self, y, y_pred):
        # calculate mse
        loss = (y - y_pred) ** 2
        # mask loss
        loss = torch.sum(loss) / y.numel()
        return loss

    def get_mape(self, y, y_pred):
        loss = torch.abs(y - y_pred) / y
        loss = torch.sum(loss) / y.numel()
        return loss
