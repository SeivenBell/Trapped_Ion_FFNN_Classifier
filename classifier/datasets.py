import h5py
import torch
from torch.utils.data import Dataset


class LabelledDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]
        self.labels = self.file["labels"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

    def close(self):
        self.file.close()


class UnlabelledDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])
        return x

    def close(self):
        self.file.close()
