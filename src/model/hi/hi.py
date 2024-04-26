import torch
import torch.nn as nn
import torch.nn.functional as F


class HI(nn.Module):
    """
    Paper: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
    Link: https://arxiv.org/abs/2103.16349
    """

    def __init__(self, cfg):
        """
        Init HI.

        Args:
            input_length (int): input time series length
            output_length (int): prediction time series length
            channel (list, optional): selected channels. Defaults to None.
            reverse (bool, optional): if reverse the prediction of HI. Defaults to False.
        """

        super(HI, self).__init__()
        self.input_length = cfg["model"]["window_size"] - cfg["model"]["horizon"]
        self.output_length = cfg["model"]["horizon"]
        assert (
            self.input_length >= self.output_length
        ), "HI model requires input length > output length"

        self.channel = cfg["model"]["channel"]
        self.reverse = cfg["model"].get("reverse", False)
        self.fake_param = nn.Linear(1, 1)

    def forward(
        self, history_data: torch.Tensor, future_data, **kwargs
    ) -> torch.Tensor:
        """Forward function of HI.

        Args:
            history_data (torch.Tensor): shape = [B,N,C, L_in,]

        Returns:
            torch.Tensor: model prediction [B, L_out, N, C].
        """
        history_data = history_data.permute(0, 3, 1, 2)
        B, L_in, N, C = history_data.shape
        assert self.input_length == L_in, "error input length"
        if self.channel is not None:
            history_data = history_data[..., self.channel]
        # historical inertia
        prediction = history_data[:, -self.output_length :, :, :]
        # last point
        # prediction = history_data[:, [-1], :, :].expand(-1, self.output_length, -1, -1)
        if self.reverse:
            prediction = prediction.flip(dims=[1])
        prediction = prediction.permute(0, 2, 3, 1).squeeze(-2)
        return (prediction, None, None, None)
