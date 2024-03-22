from copy import copy
import sys
import torch
import torch.nn as nn
import icecream as ic
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import numpy as np
from collections import Counter
#-------------------------------------------------------
# Testing
import unittest


#-------------------------------------------------------
# Initialize device to use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Your current device: {str(device)}")

# Path to your dataset
file_path = "binary/halfpi.pt"


num_images = 4
pixel_width = 5 
pixel_height = 5
total_pixels = pixel_width * pixel_height
hidden_layer_size = 512  # Adjusted to match the saved model
output_size = 1


class IonImagesDataset(Dataset):
    """Dataset class for ion images."""
    def __init__(self, file_path):
        """
        Initialize the dataset with images and labels from the file path.
        :param file_path: Path to the dataset file.
        """
        loaded_data_dict = torch.load(file_path)
        self.images = loaded_data_dict['images']
        self.labels = loaded_data_dict['labels']

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the item (image, label) by index."""
        image_tensor = self.images[idx]
        label_tensor = self.labels[idx]
        return image_tensor, label_tensor
        
# Load the dataset
dataset = IonImagesDataset(file_path)

# Split the dataset into training and validation subsets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation datasets
batch_size = min(1000, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class IndexDependentDense(nn.Module):
    """A custom dense layer with index-dependent parameters."""
    def __init__(self, num_images, total_pixels, output_size, activation=nn.ReLU()):
        """
        Initialize the layer with given configurations.
        :param num_images: Number of images.
        :param total_pixels: Total pixels in each image.
        :param output_size: Output size of the layer.
        :param activation: Activation function to use.
        """
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.empty(num_images, total_pixels, output_size))
        self.b = nn.Parameter(torch.empty(num_images, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        """Forward pass of the layer."""
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        return self.activation(y) if self.activation is not None else y

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = IndexDependentDense(num_images=4, total_pixels=25, output_size=256, activation=nn.ReLU())

    def forward(self, x):
        return self.dense(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = IndexDependentDense(num_images=4, total_pixels=512, output_size=1, activation=None)

    def forward(self, x):
        return torch.sigmoid(self.dense(x))

class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(25, 256)

    def forward(self, x):
        return self.dense(x)



class MultiIonReadout(nn.Module):
    """A model combining an encoder and a classifier for ion readout."""
    def __init__(self, encoder, classifier):
        """
        Initialize the model with an encoder and a classifier.
        :param encoder: The encoder part of the model.
        :param classifier: The classifier part of the model.
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        """Forward pass of the model."""
        x = x.reshape(*x.shape[:-2], -1).to(torch.float32)  # Reshape input
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def bceloss(self, X, y):
        """Compute binary cross-entropy loss."""
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true):
        """Compute accuracy."""
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        return (y_true == mod_y_pred).to(dtype=torch.float32).mean() * 100

    def accuracy(self, x, y):
        """Compute model accuracy."""
        return self._accuracy(self(x), y)
    #-----------------------------------------------------------------------------------
# TESTING PROCESS: 


    
def msedistloss(self, preds):
    # Convert predictions to counts
    binary_preds = (preds > 0.5).float()
    tensor_str_counts = Counter([str(tensor.tolist()) for tensor in binary_preds])

    # Extract counts and normalize
    counts = list(tensor_str_counts.values())
    normalized_counts = [x / sum(counts) for x in counts]
    uniform_counts = [1 / len(tensor_str_counts) for _ in tensor_str_counts]

    # Calculate the root mean squared error
    loss = torch.sqrt(torch.mean((torch.tensor(normalized_counts) - torch.tensor(uniform_counts)) ** 2))
    return loss

        
encoder = Encoder()
classifier = Classifier()
model = MultiIonReadout(encoder, classifier)
shared_encoder = SharedEncoder()

N_epochs = 100
lr = 1e-3
optimizer = Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ConstantLR(optimizer, factor=1)  # Adjust learning rate

# Training loop
for epoch in range(N_epochs):

    total_train_loss = 0
    
    for (inputs, _) in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()
        preds = model(inputs)

        # Compute loss for unlabeled data
        loss = model.msedistloss(preds)
        loss.backward()
        optimizer.step()

        # total_train_loss += loss.item()

    # avg_train_loss = total_train_loss / len(train_loader)
    
    
    
    sys.stdout.flush()
    # writer.add_scalar('Training Loss', avg_train_loss, epoch)

    # Evaluation loop
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            loss = model.bceloss(inputs, labels)
            accuracy = model.accuracy(inputs, labels)
            total_loss += loss.item()
            total_accuracy += accuracy.item()

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        # writer.add_scalar('Validation Loss', avg_loss, epoch)
        # writer.add_scalar('Validation Accuracy', avg_accuracy, epoch)
        
    ic("\r Epoch {}/{}, Training Loss = {}, Val Loss = {}, Val Acc = {}".format(epoch+1, N_epochs, loss.item(), avg_loss, avg_accuracy), end="")
