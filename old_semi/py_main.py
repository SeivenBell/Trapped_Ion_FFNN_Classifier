import h5py
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from icecream import ic
from FFNN_model import *


class Labelled_Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]
        self.labels = self.file["labels"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])
        x = x - 200  # , dtype=torch.float32)
        y = torch.tensor(self.labels[idx])  # , dtype=torch.float32)
        return x, y

    def close(self):
        self.file.close()


file_path = "binary/fully_reformatted_dataset.h5"

dataset = Labelled_Dataset(file_path)
print(len(dataset))  # PRINT
data, labels = dataset[0]  # Get the first sample from the dataset
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

batch_size = 64

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------------
# Data loading part is done

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device))

N = 4  # number of images in whole picture
N_i = 25  # num of features in each image
N_h = 128  # hidden units
N_o = 1  # output units (binary classification) for each image in the picture (N)
N_c = (
    1024  # combined, num of features in the whole picture encoded by the shared encoder
)

device = torch.device("cpu")
# model
# encoder = Encoder(
#     N, N_i, N_h
# )  # Encoder for each image in the picture to get the hidden representation
# shared_encoder = SharedEncoder(N_c, N_h)  #
# classifier = Classifier(N, 2 * N_h, N_o)
# model = MultiIonReadout(encoder, shared_encoder, classifier)
encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = SimpleModel(encoder, classifier)

# Setting the hyperparameters
N_epochs = 100
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)

training_losses = []
validation_losses = []
validation_accuracies = []

for epoch in range(N_epochs):

    total_train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model.bceloss(inputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Store average training loss
    avg_training_loss = total_train_loss / len(train_loader)
    training_losses.append(avg_training_loss)

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

        validation_losses.append(avg_loss)
        validation_accuracies.append(avg_accuracy)

    print(
        "\n\r Epoch {}/{}, Training Loss = {}, Val Loss = {}, Val Acc = {}".format(
            epoch + 1, N_epochs, avg_training_loss, avg_loss, avg_accuracy
        ),
        end="",
    )

torch.save(model.state_dict(), "Super_model.pth")
# Plotting the metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, N_epochs + 1), training_losses, label="Training Loss")
plt.plot(range(1, N_epochs + 1), validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, N_epochs + 1), validation_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
