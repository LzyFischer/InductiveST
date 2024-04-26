"""
This STGCN is a new model which for simpler implementation
"""

import math
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    # def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    #     super(TemporalConv, self).__init__()
    #     self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
    #     self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
    #     self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    # def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
    #     """Forward pass through temporal convolution block.

    #     Arg types:
    #         * **X** (torch.FloatTensor) -  Input data of shape
    #             (batch_size, input_time_steps, num_nodes, in_channels).

    #     Return types:
    #         * **H** (torch.FloatTensor) - Output data of shape
    #             (batch_size, in_channels, num_nodes, input_time_steps).
    #     """
    #     X = X.permute(0, 3, 2, 1)
    #     P = self.conv_1(X)

    #     Q = torch.sigmoid(self.conv_2(X))
    #     PQ = P * Q
    #     H = F.relu(PQ + self.conv_3(X))
    #     H = H.permute(0, 3, 2, 1)
    #     return H

    #     class TemporalConvLayer(nn.Module):

    # # Temporal Convolution Layer (GLU)
    # #
    # #        |--------------------------------| * Residual Connection *
    # #        |                                |
    # #        |    |--->--- CasualConv2d ----- + -------|
    # # -------|----|                                   ⊙ ------>
    # #             |--->--- CasualConv2d --- Sigmoid ---|
    # #

    # # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, in_channels, out_channels, kernel_size=3, act_func="glu"):
        super(TemporalConv, self).__init__()
        self.Kt = kernel_size
        self.c_in = in_channels
        self.c_out = out_channels
        self.align = Align(in_channels, out_channels)
        if act_func == "glu" or act_func == "gtu":
            self.causal_conv = CausalConv2d(
                in_channels=in_channels,
                out_channels=2 * out_channels,
                kernel_size=(kernel_size, 1),
                enable_padding=False,
                dilation=1,
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                enable_padding=False,
                dilation=1,
            )

        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_in = self.align(x)[:, :, self.Kt - 1 :, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == "glu" or self.act_func == "gtu":
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out :, :, :]

            if self.act_func == "glu":
                # GLU was first purposed in
                # *Language Modeling with Gated Convolutional Networks*.
                # URL: https://arxiv.org/abs/1612.08083
                # Input tensor X is split by a certain dimension into tensor X_a and X_b.
                # In the original paper, GLU is defined as Linear(X_a) ⊙ Sigmoid(Linear(X_b)).
                # However, in PyTorch, GLU is defined as X_a ⊙ Sigmoid(X_b).
                # URL: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # Because in original paper, the representation of GLU and GTU is ambiguous.
                # So, it is arguable which one version is correct.

                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x = torch.mul((x_p + x_in), self.sigmoid(x_q))

            else:
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))

        elif self.act_func == "relu":
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == "leaky_relu":
            x = self.leaky_relu(x_causal_conv + x_in)

        elif self.act_func == "silu":
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(
                f"ERROR: The activation function {self.act_func} is not implemented."
            )
        x = x.permute(0, 2, 3, 1)
        return x


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)
        )

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        [batch_size, self.c_out - self.c_in, timestep, n_vertex]
                    ).to(x),
                ],
                dim=1,
            )
        else:
            x = x

        return x


class CausalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [
                int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))
            ]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class STConv(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(
        self,
        num_nodes_list: list,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        cfg: dict,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.cfg = cfg
        self.num_nodes = num_nodes_list[-1]
        self.train_num_nodes = num_nodes_list[0]
        self.val_num_nodes = num_nodes_list[1]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._layer_norm = nn.LayerNorm(hidden_channels)
        self._batch_norm = nn.BatchNorm2d(
            hidden_channels,
        )

        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)

        if self.cfg["dynamic_graph"]:
            T_0 = T_0.permute(1, 0, 2, 3).reshape(T_0.size(1), -1, T_0.size(3))
        else:
            T_0 = T_0.reshape(-1, T_0.size(2), T_0.size(3))

        T = self._graph_conv(T_0, edge_index, edge_weight)

        if self.cfg["dynamic_graph"]:
            T = T.reshape(T.size(0), X.size(0), -1, T.size(-1)).permute(1, 0, 2, 3)
        else:
            T = T.reshape(X.size(0), -1, T.size(1), T.size(2))

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)

        if self.cfg["batch_norm"]:
            # T = self._layer_norm(T.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            T = self._batch_norm(T.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        T = self.dropout(T)
        T = T.permute(0, 2, 1, 3)
        return T


class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        cfg,
        num_nodes_list,
        act_func="glu",
        bias=True,
    ):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConv(
            cfg["hidden_channels"], cfg["hidden_channels"], cfg["K"], act_func
        )
        self.fc1 = nn.Linear(
            in_features=cfg["hidden_channels"],
            out_features=cfg["hidden_channels"],
            bias=bias,
        )
        self.fc2 = nn.Linear(
            in_features=cfg["hidden_channels"],
            out_features=cfg["out_channels"],
            bias=bias,
        )
        self.tc1_ln = nn.LayerNorm([num_nodes_list[0], cfg["hidden_channels"]])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=cfg["dropout"])

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x
