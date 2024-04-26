import torch
import torch.nn as nn


class VAR(nn.Module):
    def __init__(self, p, input_size):
        super(VAR, self).__init__()
        self.p = p
        self.input_size = input_size
        self.W = nn.Parameter(torch.randn(p, input_size, input_size))

    def forward(self, x):
        """Forward function of HI.

        Args:
            history_data (torch.Tensor): shape = [B,N,C, L_in,]

        Returns:
            torch.Tensor: model prediction [B, N, L].
        """
        x = x[:, :, 0, :].transpose(1, 2)
        y = torch.zeros(x.shape[0], x.shape[1], self.input_size)
        res = []
        for i in range(12):
            y = torch.einsum("ijk, bik -> bij", self.W, x)
            y = y.sum(dim=1)
            res.append(y)
            x = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)
        res = torch.stack(res, dim=1)
        res = res.transpose(1, 2)
        return res
