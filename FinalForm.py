import torch
import torch.nn as nn
import icecream as ic
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import numpy as np
from collections import Counter


#-------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ic(f"Your current device: {str(device)}")
#file_path = "/binary/cropped_ions.h5"

# file_path_dark = "/binary/dark_ions.h5"
# file_path_bright = "/binary/bright_ions.h5"
# file_path_halfpi = "/binary/halfpi.pt"
file_path = "binary/halfpi.pt"


num_images = 4
pixel_width = 5 
pixel_height = 5
total_pixels = pixel_width * pixel_height
hidden_layer_size = 512  # Adjusted to match the saved model
output_size = 1


class IonImagesDataset(Dataset):
    def __init__(self, file_path):
        loaded_data_dict = torch.load(file_path) # loading dataset into dataloader
        ic(loaded_data_dict.keys())

        self.images = loaded_data_dict['images'] # creating of 2 datasents of imgages and keys to them
        self.labels = loaded_data_dict['labels']

    def __len__(self):
        return len(self.images) # just return len function

    def __getitem__(self, idx):
        image_tensor = self.images[idx]  # Add a channel dimension
        label_tensor = self.labels[idx] # Repeat the label for each ion position
        ic(image_tensor.shape, label_tensor.shape)
        return image_tensor, label_tensor
        
#writer = SummaryWriter('runs/ion_images_experiment')

# Create dataset
dataset = IonImagesDataset(file_path)

# Split the dataset into training and validation subsets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(f"Train size: {train_size}, Validation size: {val_size}")

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for the training and validation datasets
batch_size = min(1000, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


class IndexDependentDense(nn.Module):
    def __init__(self, num_images, total_pixels, output_size, activation=nn.ReLU()):
        super().__init__()
        
        self.num_images = num_images
        self.total_pixels = total_pixels
        self.output_size = output_size
        self.activation = activation
        self.register_parameter(
            "W", nn.Parameter(torch.empty(self.num_images, self.total_pixels, self.output_size))
        )
        self.register_parameter("b", nn.Parameter(torch.empty(self.num_images, self.output_size)))
        
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

# class Encoder(nn.Module):
#     def __init__(self, num_images, total_pixels, output_size):
#         super().__init__()
        
#         # self.num_images = num_images
#         # self.total_pixels = total_pixels
#         # self.output_size = output_size
        
#         self.dense = IndexDependentDense(num_images, total_pixels, output_size, activation=nn.ReLU())
#         pass
    
#     def forward(self, x):
#         y = self.dense(x)
#         return y
#     pass

# #---------------------------------------------------------------------------------------------

# class Classifier(nn.Module):
#     def __init__(self, num_images, total_pixels, output_size):
#         super().__init__()
        
#         self.num_images = num_images
#         self.total_pixels = total_pixels
#         self.output_size = output_size
#         self.dense = IndexDependentDense(num_images, total_pixels, output_size, activation=None)
#         pass
#     def forward(self, x):
#         y = self.dense(x)
#         y = torch.sigmoid(y)  # Apply sigmoid activation here
#         return y
#     pass

# #---------------------------------------------------------------------------------------------

# class SharedEncoder(nn.Module):
#     def __init__(self, num_images, total_pixels, output_size):
#         super().__init__()
        
#         self.num_images = num_images
#         self.total_pixels = total_pixels
#         self.output_size = output_size
        
#         self.dense = nn.Linear(total_pixels, output_size)
#         pass
        
#     def forward(self, x):
#         y = self.dense(x)
#         return y
    
#     pass
   
#---------------------------------------------------------------------------------------------    
##############################################################################################
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
##############################################################################################

        
# class SharedClassifier(nn.Module):
#     def __init__(self, num_images, total_pixels, output_size):
#         super().__init__()
        
#         self.num_images = num_images
#         self.total_pixels = total_pixels
#         self.output_size = output_size
        
#         self.dense = nn.Linear(total_pixels, output_size)
#         pass
    
#     def forward(self, x):
#         y = self.dense(x)
#         return y
    
#     pass

#---------------------------------------------------------------------------------------------


class MultiIonReadout(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        
        ic(f"Original input shape: {x.shape}")
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        ic(f"Shape after initial reshape: {y.shape}")
        
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
        self.coupling = nn.Linear(hidden_layer_size * num_images, hidden_layer_size * num_images)

        # Make sure coupling layer is trainable
        for p in self.coupling.parameters():
            p.requires_grad = True

    def forward(self, x):
        ic(f"Original input shape: {x.shape}")
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        ic(f"Shape after initial reshape: {y.shape}")

        y = self.mir.encoder(y)
        ic(f"Shape after encoder in mir: {y.shape}")

        y = y.reshape(y.shape[0], -1)  # desired shapeS
        ic(f"Shape before coupling layer: {y.shape}")

        y = self.coupling(y)
        ic(f"Shape after coupling layer: {y.shape}")

        y = y.reshape(y.shape[0], -1)   # desired shape for classifier
        ic(f"Shape before classifier in mir: {y.shape}")

        y = self.mir.classifier(y)
        ic(f"Shape after classifier in mir: {y.shape}")

        return y

    
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

        


device = torch.device("cpu")

# encoder = Encoder(num_images, total_pixels, hidden_layer_size)
# classifier = Classifier(num_images, hidden_layer_size, output_size)
# model = MultiIonReadout(encoder, classifier)
encoder = Encoder()
classifier = Classifier()
model = MultiIonReadout(encoder, classifier)
shared_encoder = SharedEncoder()
enhanced_model = EnhancedMultiIonReadout(model)


# model_path = "golden_WandB.pth"
# model.load_state_dict(torch.load(model_path))
# model = model.to(device)


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
