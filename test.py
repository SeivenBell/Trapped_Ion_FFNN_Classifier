import sys
import re
import h5py
import copy
import torch
import random
import torchvision
import numpy as np
import numpy as np
from torch import nn
from torch.optim import Adam
from torchinfo import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split


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
    def __init__(self, file_paths, labels, ion_positions=[0, 1, 2, 3]):
        images = []
        categories = []

        for file_path, label in zip(file_paths, labels):
            with h5py.File(file_path, "r") as f:
                print(f"h5 file has {len(f.keys())} keys.")  # Print the number of keys

                # # Get the unique image numbers from the keys
                # image_numbers = [int(re.search(r'(\d+)_ion_0', key).group(1)) for key in f.keys()]
                
                image_number = 0
                # while image_number<100:
                while True:
                    try:
                        image_tensors = []
                        for ion_position in ion_positions:
                            key = f"{label}_{image_number}_ion_{ion_position}"
                            ion_image = np.array(f[key])  # Load data as a numpy array
                            ion_image_tensor = torch.tensor(ion_image, dtype=torch.float32).view(L_x, L_y) -200  # Reshape the tensor
                            image_tensors.append(ion_image_tensor)

                        # Concatenate the image tensors for all ion positions
                        combined_image_tensor = torch.stack(image_tensors)
                        images.append(combined_image_tensor[None,...])
                        categories.append(torch.tensor([0,0,0,0],dtype=torch.float32)[:,None] if label == "dark" else torch.tensor([1,1,1,1],dtype=torch.float32)[:,None])
                    
                        image_number += 1
                    except:
                        break
            
        self.images = torch.concat(images,dim=0)
        self.labels = torch.stack(categories)
        print("Saving...")
        torch.save(self.images, 'C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_combined_images.pt')
        torch.save(self.labels, 'C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_combined_labels.pt')

        print("Total images:", len(self.images))  # Debug print
        print("Total labels:", len(self.labels))  # Debug print

        print("Total images:", self.images.size())  # Debug print
        print("Total labels:", self.labels.size())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tensor = self.images[idx]  # Add a channel dimension
        label_tensor = self.labels[idx] # Repeat the label for each ion position
        return image_tensor, label_tensor


writer = SummaryWriter('runs/ion_images_experiment')

file_paths = [file_path_dark, file_path_bright]
labels = ["dark", "bright"]
dataset = IonImagesDataset(file_paths, labels)

# Split the dataset into training and validation subsets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"Train size: {train_size}, Validation size: {val_size}")  # Print the sizes of subsets

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")

encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)
# enhanced_model = EnhancedMultiIonReadout(model)

model = model.to(device)

# # Create a SummaryWriter
# writer = SummaryWriter('runs/ion_images_experiment')

# # Log model architecture (Optional)
# images, _ = next(iter(train_loader))
# writer.add_graph(model, images)

N_epochs = 100
lr = 1e-3
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

# Training loop
for epoch in range(N_epochs):

    total_train_loss = 0
    for (inputs, labels) in train_loader:
    
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = model.bceloss(inputs, labels)
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
        
    print("\r Epoch {}/{}, Training Loss = {}, Val Loss = {}, Val Acc = {}".format(epoch+1, N_epochs, loss.item(), avg_loss, avg_accuracy), end="")
    
# Close the writer
# writer.close()
