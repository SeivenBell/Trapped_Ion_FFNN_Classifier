import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))
print(torch.cuda.is_available())


def correlation_loss(y):
    return torch.mean(y[:, :-1, :] * y[:, 1:, :]) - 0.5


class MyCustomLoss(nn.Module):
    def __init__(self):
        super(MyCustomLoss, self).__init__()

    def forward(self, y):
        loss = torch.mean(y[:, :-1, :] * y[:, 1:, :]) - 0.5
        return loss
    pass

class SpecModel(nn.Module):
    def __init__(self):
        super(SpecModel, self).__init__()
        self.iffnn_weights = nn.Parameter(torch.randn(4, 32, 5 * 5))
        self.iffnn_biases = nn.Parameter(torch.randn(4, 32))
        self.hidden1 = nn.Linear(32, 16)  # Adjusted to output size 16
        
    def forward(self, x):
        B, _ = x.size()
        x = x.view(B, -1)
        x = torch.relu(torch.einsum('ij,jk->ik', x, self.iffnn_weights.view(4*32, 5*5)) + self.iffnn_biases.view(-1))
        x = self.hidden1(x)
        return x


class SRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.iffnns = nn.ModuleList([SpecModel() for _ in range(4)])
        self.hidden1 = nn.Linear(64, 128)  # Hidden layer 1
        self.hidden2 = nn.Linear(128, 64)  # Hidden layer 2
        self.gffnn = nn.Linear(64, 4)  # Reshaper NN that outputs a 4-digit output

    def forward(self, x):
        assert x.shape[1:] == torch.Size([4, 5, 5])
        B = x.shape[0]
        y = x.clone()
        y = y.view(B, 4, 25)
        y = torch.cat([iffnn(y[:, i, :]) for i, iffnn in enumerate(self.iffnns)], dim=-1)
        y = self.hidden1(y)
        y = self.hidden2(y)
        y = self.gffnn(y)
        return y

feature_extractor = nn.Sequential(*list(SpecModel().children())[:-1])

class IonImagesDataset(Dataset):
    def __init__(self, file_path, ion_position):
        with h5py.File(file_path, "r") as f:
            self.keys = [key for key in f.keys() if f"ion_{ion_position}" in key]

            self.images = [np.array(f[key]) for key in self.keys]

            self.labels = [self.get_label(key) for key in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the four related images

        images = [
            torch.from_numpy(self.images[(idx + i) % len(self.images)]).float()
            for i in range(4)
        ]

        # Stack the images along a new dimension to create a 4-channel image

        images = torch.stack(images, dim=0)

        return images, self.labels[idx]

    def get_label(self, key):
        # Extract the label from the key

        label = key.split("_")[0]

        return 0 if label == "dark" else 1


datasets = [IonImagesDataset("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5", i) for i in range(4)]
train_datasets = []
test_datasets = []

for dataset in datasets:
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

models = [SRA().to(device) for _ in range(4)]
batch_size = 1000
learning_rate = 0.001
criterion = MyCustomLoss()  # Use the custom loss function

for ion_position in range(4):
    model = models[ion_position]
    train_loader = DataLoader(train_datasets[ion_position], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_datasets[ion_position], batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min")
    early_stopping_counter = 0
    early_stopping_limit = 10
    best_loss = float("inf")

    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.float().to(device))
            loss = criterion(outputs)  # Use the custom loss function
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        scheduler.step(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_limit:
                print("Stopping early.")
                break

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()

    print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Accuracy on test set: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), f"C:/Users/Seiven/Desktop/MY_MLmodels/ions2/FFNNCombined_WandB_Saved_ion_{ion_position}.pth")
