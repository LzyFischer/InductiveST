import torch
import pdb
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DataIterator(Dataset):
    def __init__(
        self,
        configs,
        values,
        scaler=(0, 1),
        mode="train",
    ):
        self.configs = configs
        self.values = values
        self.batch_size = configs["batch_size"]
        self.mode = mode
        self.scaler = scaler

    def __len__(self):
        return self.values.shape[2]

    def __getitem__(self, idx):
        return torch.from_numpy(self.values[:, :, idx])

    def get_scaler(self):
        self.scaler = np.concatenate((self.scaler[0], self.scaler[1]), axis=0)
        self.scaler = torch.from_numpy(self.scaler).float().permute(0, 2, 1, 3)

        return self.scaler

    def get_loader(self):
        if self.mode == "train":
            shuffle = True
        else:
            shuffle = False
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
