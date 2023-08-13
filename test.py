import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import numpy as np
import copy
from torch.utils.data import Dataset
from torchinfo import summary
from torch.optim import Adam
import h5py
import numpy as np
import re

# bright_data = np.load("bright.npy")
# dark_data = np.load("dark.npy")
# bright_data = torch.from_numpy(bright_data) - 200
# bright_label = torch.ones(*bright_data.shape[:-2], 1)
# dark_data = torch.from_numpy(dark_data) - 200
# dark_label = torch.zeros(*dark_data.shape[:-2], 1)

# mixed_data = torch.cat([bright_data, dark_data], dim=0)
# mixed_label = torch.cat([bright_label, dark_label], dim=0)

# full_ds = TensorDataset(mixed_data, mixed_label)

# val_ratio = 0.2
# train_size = int((1 - val_ratio) * len(full_ds))
# val_size = len(full_ds) - train_size
# train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# batch_size = 1000
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
#-------------------------------------------------------



#-------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))


class IonImagesDataset(Dataset):
    def __init__(self, file_path, ion_position):
        with h5py.File(file_path, "r") as f:
            self.keys = [key for key in f.keys() if f"ion_{ion_position}" in key]
            self.images = [np.array(f[key]) for key in self.keys]
            self.labels = [self.get_label(key) for key in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = [
            torch.from_numpy(self.images[(idx + i) % len(self.images)]).float()
            for i in range(4)
        ]
        images = torch.stack(images, dim=0)
        return images, self.labels[idx] * torch.ones((4,1))

    def get_label(self, key):
        label = key.split("_")[0]
        return 0 if label == "dark" else 1

# Create datasets for each ion position
datasets = [IonImagesDataset("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5", i) for i in range(4)]

# Concatenate the datasets
full_ds = ConcatDataset(datasets)

# Split the full dataset into training and validation datasets
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# Create DataLoaders for the training and validation datasets
batch_size = 1000
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)


#---------------------------------------------------------------------------------------------


class IndexDependentDense(nn.Module):
    def __init__(self, N, N_i, N_o, activation=nn.ReLU()):
        super().__init__()
        
        self. N = N
        self.N_i = N_i
        self.N_o = N_o
        self.activation = activation
        self.register_parameter(
            "W", nn.Parameter(torch.empty(self.N, self.N_i, self.N_o))
        )
        self.register_parameter("b", nn.Parameter(torch.empty(self.N, self.N_o)))
        
        self._reset_parameters()
        
        pass
    
    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #     pass
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self,x):
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        if self.activation is not None:
            return self.activation(y)
        else:
            return y
    pass

#---------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()
        
        self. N = N
        self.N_i = N_i
        self.N_o = N_o
        
        self.dense = IndexDependentDense(N, N_i, N_o, activation=None)
        pass
    
    def forward(self, x):
        y = self.dense(x)
        return y
    pass

#---------------------------------------------------------------------------------------------

class Classifier(nn.Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()
        
        self. N = N
        self.N_i = N_i
        self.N_o = N_o
        self.dense = IndexDependentDense(N, N_i, N_o, activation=F.sigmoid)
        pass
    def forward(self, x):
        y = self.dense(x)
        y = torch.sigmoid(y)  # Apply sigmoid activation here
        return y
    pass

#---------------------------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()
        
        self. N = N
        self.N_i = N_i
        self.N_o = N_o
        
        self.dense = nn.Linear(N_i, N_o)
        pass
        
    def forward(self, x):
        y = self.dense(x)
        return y
    
    pass
   
#---------------------------------------------------------------------------------------------    
        
class SharedClassifier(nn.Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()
        
        self. N = N
        self.N_i = N_i
        self.N_o = N_o
        
        self.dense = nn.Linear(N_i, N_o)
        pass
    
    def forward(self, x):
        y = self.dense(x)
        return y
    
    pass

#---------------------------------------------------------------------------------------------


class MultiIonReadout(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        y = self.encoder(y)
        y = self.classifier(y)
        return y

    def bceloss(self, X, y):
        return F.binary_cross_entropy(self(X), y)

    @staticmethod
    def _accuracy(y_pred, y_true):
        mod_y_pred = (y_pred > 0.5).to(torch.float32)
        accuracy = (y_true == mod_y_pred).to(dtype=torch.float32).mean()
        return accuracy * 100

    def accuracy(self, x, y):
        return self._accuracy(self(x), y)
#---------------------------------------------------------------------------------------------

class EnhancedMultiIonReadout(nn.Module):
    def __init__(self, mir):
        super().__init__()
        
        self.mir = copy.deepcopy(mir)
        
        for p in self.mir.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        y = self.mir.encoder(y)
        y = self.mir.classifier(y)
        return y

device = torch.device("cpu")
N = 4
L_x = 5 
L_y = 5
N_i = L_x * L_y
N_h = 256
N_o = 1

encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)
enhanced_model = EnhancedMultiIonReadout(model)


###############################################################

# # Check the HDF5 file
# with h5py.File("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5", "r") as f:
#     keys = [key for key in f.keys() if "ion_" in key]
#     for key in keys[:5]:
#         print(f"Key: {key}")
#         data = f[key]
#         print(f"Data shape: {data.shape}")
#         label = 0 if key.split("_")[0] == "dark" else 1
#         print(f"Label: {label}")

# # Check the dataset
dataset = IonImagesDataset("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5", 0)
# for i in range(5):  # Check the first 10 samples
#     image, label = dataset[i]
#     print(f"Image shape: {image.shape}, Label: {label}", sep=' ')

# Check the data loader
loader = DataLoader(dataset, batch_size=1000, shuffle=True, drop_last=True)
# for images, labels in loader:
#     print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
#     if images.shape[0] != 1000 or labels.shape[0] != 1000:
#         print("Warning: Batch size is not 1000. Check if your dataset size is a multiple of the batch size.")
#     break  # Check only the first batch

# # Check the label dimensions
# for images, labels in loader:
#     images, labels = images.to(device), labels.to(device)
#     outputs = model(images)
#     print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
#     if outputs.shape != labels.shape:
#         print("Warning: Mismatch between the dimensions of the labels and the model's output.")
#     break  # Check only the first batch

# # Try a smaller batch size
# loader = DataLoader(dataset, batch_size=500, shuffle=True, drop_last=True)
# for images, labels in loader:
#     images, labels = images.to(device), labels.to(device)
#     try:
#         outputs = model(images)
#         print("Batch size of 500 works fine.")
#     except RuntimeError as e:
#         print(f"Error with batch size of 500: {e}")
#     break  # Check only the first batch

def visualize_images(dataset, title):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(title)
    axes = axes.flatten()  # Flatten the axes array
    for i in range(4):
        sample_idx = np.random.randint(len(dataset))
        images, labels = dataset[sample_idx]
        label = labels[0].item()  # Take the first element of the label tensor
        axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

##############################################################

N_epochs = 50
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

# Training loop
for epoch in range(N_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = model.bceloss(inputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % log_every == 0:
        print(f"Epoch {epoch}/{N_epochs}, Loss: {loss.item()}")

    # Evaluation loop
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            loss = model.bceloss(inputs, labels)
            accuracy = model.accuracy(inputs, labels)
            total_loss += loss.item()
            total_accuracy += accuracy.item()

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        print(f"Validation Loss: {avg_loss}, Validation Accuracy: {avg_accuracy}")
