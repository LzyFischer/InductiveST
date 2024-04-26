import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pdb


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in  # 1
        self.c_out = c_out  # 64
        self.align_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)
        )

    def forward(self, x):
        # here n_vertex is gotten from the shape of x
        # x: batch * channel [1] * time * vertex
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


class CausalConv1d(nn.Conv1d):
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
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]

        return result


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


class TemporalConvLayer(nn.Module):
    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|
    #

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == "glu" or act_func == "gtu":
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
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
        # x batch * channel [1] * time * vertex
        x_in = self.align(x)[
            :, :, self.Kt - 1 :, :
        ]  # 调整到c_out |  x_in的time的前几步不使用？--> 考虑到kernel size 用后面几个对齐
        x_causal_conv = self.causal_conv(x)  # 注意这里是x，不是x_in

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

        return x


class ChebGraphConv(nn.Module):
    def __init__(self, configs, c_in, c_out, Ks, gso, bias, gso_train, gso_val):
        super(ChebGraphConv, self).__init__()
        self.configs = configs
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.gso_train = gso_train
        self.gso_val = gso_val
        self.weight = nn.Parameter(
            torch.FloatTensor(Ks, c_in, c_out)
        )  # parameter tensor as input
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))  # always out
        else:
            self.register_parameter("bias", None)  # self.bias = None
        self.reset_parameters()  # init the parameter

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        x = torch.permute(x, (0, 2, 3, 1))
        # bs, ts, n_vertex, c_in 64 10 307 16
        self.val = False
        if x.shape[2] == len(self.gso_val):
            self.val = True
        if self.training:
            self.temp_gso = self.gso_train.to(x.device)
        elif self.val:
            self.temp_gso = self.gso_val.to(x.device)
        else:
            self.temp_gso = self.gso.to(x.device)

        if self.configs["no_graph"]:
            self.temp_gso = (
                torch.zeros(x.shape[-2], x.shape[-2])
                .expand(x.shape[0], -1, -1)
                .to(x.device)
            )
        elif self.configs["original_graph"]:
            self.temp_gso = self.temp_gso.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            self.temp_gso = gso

        if self.Ks - 1 < 0:
            raise ValueError(
                f"ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}."
            )  # kernel of Ks and Kt should be positive
        elif self.Ks - 1 == 0:  # no convolution
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:  # one layer
            x_0 = x
            x_1 = torch.einsum("bhi,btij->bthj", self.temp_gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum("bhi,btij->bthj", self.temp_gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(
                    torch.einsum("bhi,btij->bthj", 2 * self.temp_gso, x_list[k - 1])
                    - x_list[k - 2]  # chebyshev polynomials
                )  # h * i --> graph, i * j

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum(
            "btkhi,kij->bthj", x, self.weight
        )  # adding weight to learn for

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias, gso_train, gso_val):
        """
        Only for one layer graph is not used that often
        """
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.gso_train = gso_train
        self.gso_val = gso_val
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # bs, c_in, ts, n_vertex = x.shape
        self.gso_train = gso
        self.gso_val = gso
        self.gso = gso

        x = torch.permute(x, (0, 2, 3, 1))
        self.val = False
        if x.shape[2] == len(self.gso_val):
            self.val = True
        if self.training:
            self.gso_train = self.gso_train.to(x.device)

            first_mul = torch.einsum("bhi,btij->bthj", self.gso_train, x)
            second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)
        elif self.val:
            self.gso_val = self.gso_val.to(x.device)

            first_mul = torch.einsum("bhi,btij->bthj", self.gso_val, x)
            second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)
        else:
            self.gso = self.gso.to(x.device)

            first_mul = torch.einsum("bhi,btij->bthj", self.gso, x)
            second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(
        self, configs, graph_conv_type, c_in, c_out, Ks, gso, gso_train, gso_val, bias
    ):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        self.configs = configs

        self.gso_train = gso_train
        self.gso_val = gso_val

        if self.graph_conv_type == "cheb_graph_conv":
            self.cheb_graph_conv = ChebGraphConv(
                configs, c_out, c_out, Ks, gso, bias, self.gso_train, self.gso_val
            )
        elif self.graph_conv_type == "graph_conv":
            self.graph_conv = GraphConv(
                c_out, c_out, gso, bias, self.gso_train, self.gso_val
            )

        self.alpha = nn.Parameter(torch.FloatTensor(1))
        # initialize alpha
        self.alpha.data.fill_(0.0)

    def forward(self, x, gso):
        x_gc_in = self.align(x)
        if self.graph_conv_type == "cheb_graph_conv":
            x_gc = self.cheb_graph_conv(x_gc_in, gso)
        elif self.graph_conv_type == "graph_conv":
            x_gc = self.graph_conv(x_gc_in, gso)
        x_gc = x_gc.permute(0, 3, 1, 2)
        # x_gc_out = x_gc_in + self.alpha * x_gc
        x_gc = torch.nn.functional.normalize(x_gc)
        x_gc_in = torch.nn.functional.normalize(x_gc_in)
        if self.configs["learnable_alpha"]:
            x_gc_out = torch.add(x_gc_in, self.alpha * x_gc)
        else:
            x_gc_out = torch.add(x_gc_in, self.configs["alpha"] * x_gc)

        return x_gc_out


class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(
        self,
        configs,
        Kt,
        Ks,
        n_vertex,
        last_block_channel,
        channels,
        act_func,
        graph_conv_type,
        gso,
        gso_train,
        gso_val,
        n_vertex_train,
        n_vertex_val,
        bias,
        droprate,
    ):
        super(STConvBlock, self).__init__()

        self.tmp_conv1 = TemporalConvLayer(
            Kt,
            last_block_channel,
            channels[0],
            n_vertex,
            act_func,  # this is one block, last block end as input size
        )
        self.graph_conv = GraphConvLayer(
            configs,
            graph_conv_type,
            channels[0],
            channels[1],
            Ks,
            gso,
            gso_train,
            gso_val,
            bias,
        )  # support inductive now
        self.tmp_conv2 = TemporalConvLayer(
            Kt, channels[1], channels[2], n_vertex, act_func
        )
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], elementwise_affine=False)
        self.tc2_ln_train = nn.LayerNorm(
            [n_vertex_train * (configs["grad_iter_num"]), channels[2]],
            elementwise_affine=False,
        )
        self.tc2_ln_val = nn.LayerNorm(
            [n_vertex_val, channels[2]], elementwise_affine=False
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.n_vertex_train = n_vertex_train
        self.n_vertex_val = n_vertex_val

        ###

    def forward(self, input):
        # TGATND
        x, gso = input
        x = self.tmp_conv1(x)
        x = self.graph_conv(x, gso)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        self.val = False
        if x.shape[-1] == self.n_vertex_val:
            self.val = True
        if self.training:
            x = self.tc2_ln_train(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        elif self.val:
            x = self.tc2_ln_val(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x, gso


class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        configs,
        Ko,
        last_block_channel,
        channels,
        end_channel,
        n_vertex,
        n_vertex_train,
        n_vertex_val,
        act_func,
        bias,
        droprate,
    ):
        super(OutputBlock, self).__init__()
        self.n_vertex_val = n_vertex_val
        self.tmp_conv1 = TemporalConvLayer(
            Ko, last_block_channel, channels[0], n_vertex, act_func
        )  # Ko is the left part of the T, then Ko -> 1, here always use the CNN
        self.fc1 = nn.Linear(
            in_features=channels[0], out_features=channels[1], bias=bias
        )
        self.fc2 = nn.Linear(
            in_features=channels[1], out_features=end_channel, bias=bias
        )
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], elementwise_affine=False)
        self.tc1_ln_train = nn.LayerNorm(
            [n_vertex_train * (configs["grad_iter_num"]), channels[0]],
            elementwise_affine=False,
        )  # no T because T is already 1
        self.tc1_ln_val = nn.LayerNorm(
            [n_vertex_val, channels[0]], elementwise_affine=False
        )
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        self.val = False
        if x.shape[-1] == self.n_vertex_val:
            self.val = True
        if self.training:
            x = self.tc1_ln_train(x.permute(0, 2, 3, 1))
        elif self.val:
            x = self.tc1_ln_val(x.permute(0, 2, 3, 1))
        else:
            x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)  # channel switch from C to T
        return x


"""
Need to mind the complete the shape for next operation
从刚才的结果中我有学到什么吗？
1. 只有读了代码才能知道，这个代码是怎么运行的
2. 积累已有的经验非常重要，从chatgpt学到该怎么搜索。
"""
