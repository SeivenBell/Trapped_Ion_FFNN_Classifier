# %%
import torch
import os
import h5py
import random
import numpy as np
from torch import nn
from icecream import ic
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

# file_path = "binary/reformatted_dataset.h5"  # adjust this path based on your setup
# print("Does the file exist?", os.path.exists(file_path))

class Labelled_Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]
        self.labels = self.file["labels"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])  # , dtype=torch.float32)
        y = torch.tensor(self.labels[idx])  # , dtype=torch.float32)
        return x, y


file_path = "binary/reformatted_dataset.h5"

dataset = Labelled_Dataset(file_path)

batch_size = 64


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# print("Number of samples in dataset:", len(dataset))


# # Fetch the first sample
# first_sample, first_label = dataset[0]
# print("First measurement:", first_sample)
# print("First label:", first_label)

# # Fetch multiple samples to inspect
# for i in range(5):
#     measurement, label = dataset[i]
#     print(f"Sample {i}: Measurement shape {measurement.shape}, Label {label}")

# def load_even_mnist_data(data_path, batch_size, test_size):
#     """
#     Loads and processes the even MNIST dataset from a CSV file.

#     This function reads the MNIST dataset (containing only even digits),
#     preprocesses the images, and splits the data into training and test sets.
#     The data is then converted into PyTorch DataLoader objects for easy batch processing
#     during model training.

#     Args:
#         data_path (str): The path to the CSV file containing the dataset.
#         batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
#         test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

#     Returns:
#         Tuple[DataLoader, DataLoader]: A tuple containing the training and test DataLoaders.

#     """
    
#     # Load the data from the CSV file
#     data = np.genfromtxt(data_path, delimiter=' ')
#     print("Data loading...")
#     print(f"Initial Data Shape: {data.shape}")
    

#     # Extract and preprocess images
#     images = data[:, :-1]  # All columns except the last one
#     num_samples, _ = images.shape
#     images = images.reshape((num_samples, 1, 14, 14))
#     print(f"Images Shape: {images.shape}")
#     images = images / 255.0  # Normalize between 0-1

#     # Convert to PyTorch tensors
#     images = torch.from_numpy(images).float()
#     print(f"Images Shape: {images.shape}")

#     # Split into training and test sets
#     num_train = int((1 - test_size) * num_samples)
#     train_images, test_images = images[:num_train], images[num_train:]
#     print("Train Images Shape: ", train_images.shape)

#     # Create TensorDatasets
#     train_dataset = TensorDataset(train_images)
#     test_dataset = TensorDataset(test_images)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     print("Data loading done.")

#     return train_loader, test_loader


