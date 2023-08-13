import torch
import h5py
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import numpy as np
import random
import copy
from torch.utils.data import Dataset
from torchinfo import summary
from torch.optim import Adam
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import re

#-------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))
file_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5"

file_path_dark = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/dark_ions.h5"
file_path_bright = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/bright_ions.h5"


N = 4
L_x = 5 
L_y = 5
N_i = L_x * L_y
N_h = 256
N_o = 1


class IonImagesDataset(Dataset):
    def __init__(self, file_path, label, ion_positions=[0, 1, 2, 3]):
        self.images = []
        self.labels = []

        with h5py.File(file_path, "r") as f:
            print(f"h5 file has {len(f.keys())} keys.")  # Print the number of keys

            # Get the unique image numbers from the keys
            image_numbers = set(int(re.search(r'(\d+)_ion_', key).group(1)) for key in f.keys())

            for image_number in image_numbers:
                image_tensors = []
                for ion_position in ion_positions:
                    key = f"{label}_{image_number}_ion_{ion_position}"
                    ion_image = np.array(f[key])  # Load data as a numpy array
                    ion_image_tensor = torch.tensor(ion_image, dtype=torch.float32).view(L_x, L_y)  # Reshape the tensor
                    image_tensors.append(ion_image_tensor)

                # Concatenate the image tensors for all ion positions
                combined_image_tensor = torch.stack(image_tensors)
                self.images.append(combined_image_tensor)
                self.labels.append(0 if label == "dark" else 1)

        print("Total images:", len(self.images))  # Debug print
        print("Total labels:", len(self.labels))  # Debug print


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tensor = self.images[idx].unsqueeze(0)  # Add a channel dimension
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float).view(1, 1)
        return image_tensor, label_tensor



# Separate datasets for dark and bright categories
dark_dataset = IonImagesDataset(file_path_dark, label="dark")
bright_dataset = IonImagesDataset(file_path_bright, label="bright")

print(f"Dark dataset size: {len(dark_dataset)}")  # Print the size of the dark_dataset
print(f"Bright dataset size: {len(bright_dataset)}")  # Print the size of the bright_dataset

# Split the dark and bright datasets into training and validation subsets
dark_train_size = int(0.8 * len(dark_dataset))
dark_val_size = len(dark_dataset) - dark_train_size
print(f"Dark train size: {dark_train_size}, Dark val size: {dark_val_size}")  # Print the sizes of dark subsets

bright_train_size = int(0.8 * len(bright_dataset))
bright_val_size = len(bright_dataset) - bright_train_size
print(f"Bright train size: {bright_train_size}, Bright val size: {bright_val_size}")  # Print the sizes of bright subsets

dark_train_dataset, dark_val_dataset = random_split(dark_dataset, [dark_train_size, dark_val_size])
bright_train_dataset, bright_val_dataset = random_split(bright_dataset, [bright_train_size, bright_val_size])

# Combine dark and bright datasets for training and validation
train_dataset = ConcatDataset([dark_train_dataset, bright_train_dataset])
val_dataset = ConcatDataset([dark_val_dataset, bright_val_dataset])

print(f"Train dataset size: {len(train_dataset)}")  # Print the size of the train_dataset
print(f"Validation dataset size: {len(val_dataset)}")  # Print the size of the val_dataset

# Create DataLoaders for the training and validation datasets
batch_size = min(1000, len(train_dataset)) # or choose a smaller value
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)



class IndexDependentDense(nn.Module):
    def __init__(self, N, N_i, N_o, activation=nn.ReLU()):
        super().__init__()
        
        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.activation = activation
        self.register_parameter(
            "W", nn.Parameter(torch.empty(self.N, self.N_i, self.N_o))
        )
        self.register_parameter("b", nn.Parameter(torch.empty(self.N, self.N_o)))
        
        self._reset_parameters()
        
        pass
    
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
        
        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        
        self.dense = IndexDependentDense(N, N_i, N_o, activation=nn.ReLU())
        pass
    
    def forward(self, x):
        y = self.dense(x)
        return y
    pass

#---------------------------------------------------------------------------------------------

class Classifier(nn.Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()
        
        self.N = N
        self.N_i = N_i
        self.N_o = N_o
        self.dense = IndexDependentDense(N, N_i, N_o, activation=None)
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
        
        self.N = N
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
        
        self.N = N
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


encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)
enhanced_model = EnhancedMultiIonReadout(model)


###############################################################
def print_sample_keys_and_labels(dataset, num_samples=5):
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    for global_idx in sample_indices:
        # Determine the underlying IonImagesDataset object this index corresponds to
        idx = global_idx
        underlying_dataset = dataset
        while isinstance(underlying_dataset, ConcatDataset):
            dataset_idx, sample_idx = underlying_dataset.cumulative_sizes.index_right(idx)
            underlying_dataset = underlying_dataset.datasets[dataset_idx]
            idx = sample_idx
        if isinstance(underlying_dataset, Subset):
            underlying_dataset = underlying_dataset.dataset  # Get the underlying IonImagesDataset
        key = underlying_dataset.keys[idx]  # Access the key using the corrected index
        images, labels = dataset[global_idx]
        print(f"Key: {key}, Labels: {labels.numpy()}")






# def visualize_images(dataset, title):
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#     fig.suptitle(title)
#     axes = axes.flatten()  # Flatten the axes array
#     for i in range(4):
#         sample_idx = np.random.randint(len(dataset))
#         print(sample_idx, len(dataset))
#         images, labels = dataset[sample_idx]
#         print(dataset[sample_idx])
#         label = labels[0].item()  # Take the first element of the label tensor
#         axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
#         axes[i].set_title(f"Label: {label}")
#         axes[i].axis('off')
#     plt.show()


# def check_data_distribution(dataset, dataset_name):
#     labels_count = {0: 0, 1: 0}
#     for _, labels in dataset:
#         for label in labels.squeeze():  # Iterate through the batch of labels
#             labels_count[label.item()] += 1
#     print(f"{dataset_name} Labels Distribution: {labels_count}")

# visualize_images(train_dataset, "Training Dataset")
# visualize_images(val_dataset, "Validation Dataset")
# check_data_distribution(train_dataset, "Training Dataset")
# check_data_distribution(val_dataset, "Validation Dataset")

model = model.to(device)
##############################################################

N_epochs = 5
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

# Training loop
for epoch in range(N_epochs):
    print(f"Epoch {epoch + 1}/{N_epochs}")

    total_train_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss = model.bceloss(inputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

        if i % log_every == 0:  # Log every 'log_every' batches
            print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item()}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss}")

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


