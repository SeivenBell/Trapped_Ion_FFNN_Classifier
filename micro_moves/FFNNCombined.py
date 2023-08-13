import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import h5py
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))



class OldFFNN(nn.Module):
    def __init__(self):
        super(OldFFNN, self).__init__()
        self.iffnn = nn.Linear(5 * 5, 32)
        self.gffnn = nn.Linear(32, 16)  # Adjusted to output size 16


    def forward(self, x):
        x = x.view(-1, 5 * 5)
        x = torch.relu(self.iffnn(x))
        x = self.gffnn(x)
        return x

class GlobalFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 4)

    def forward(self, x):
        return self.linear1(x)

class SRA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.iffnn = OldFFNN()  # Replacing IndividualFFNN with OldFFNN
        self.gffnn = GlobalFFNN()
         
    def forward(self, x):
        assert x.shape[1:] == torch.Size([4, 5, 5])
        
        B = x.shape[0]
        
        y = x.clone()
        
        #print(f"Shape after input: {y.shape}")
        
        y  = y.view(B, 4, 25)
        
        #print(f"Shape after reshaping: {y.shape}")
        
        y = self.iffnn(y)
        
        #print(f"Shape after OldFFNN: {y.shape}")
        
        y = y.view(B, 64)
        
        #print(f"Shape after reshaping: {y.shape}")
        
        y = self.gffnn(y)
        
        #print(f"Shape after GlobalFFNN: {y.shape}")
        
        return y



# Instantiate the old model
old_model_instance = OldFFNN().to(device)

# Load the weights from the old model
state_dict = torch.load("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/best_model_V3.pth", map_location=device)
state_dict = {k: v for k, v in state_dict.items() if 'gffnn' not in k}
old_model_instance.load_state_dict(state_dict, strict=False)


# Remove the last layer from the old model to use it as a feature extractor
feature_extractor = nn.Sequential(*list(old_model_instance.children())[:-1])



# Dataset
class IonImagesDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.keys = list(f.keys())
            self.images = [np.array(f[key]) for key in self.keys]
            self.labels = [self.get_label(key) for key in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the four related images
        images = [torch.from_numpy(self.images[(idx + i) % len(self.images)]).float() for i in range(4)]
        # Stack the images along a new dimension to create a 4-channel image
        images = torch.stack(images, dim=0)
        return images, self.labels[idx]


    def get_label(self, key):
        # Extract the label from the key
        label = key.split('_')[0]
        return 0 if label == 'dark' else 1


# Load the dataset
dataset = IonImagesDataset('C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5')

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])



model = SRA()


from torch.optim.lr_scheduler import ReduceLROnPlateau

batch_size = 1000

# Create your data loaders with the specified batch size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


learning_rate = 0.001


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


scheduler = ReduceLROnPlateau(optimizer, 'min')


early_stopping_counter = 0
early_stopping_limit = 10
best_loss = float('inf')

# Training loop
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())  # Ensure that inputs are floats
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    scheduler.step(average_loss)

    # Early stopping
    if average_loss < best_loss:
        best_loss = average_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_limit:
            print("Stopping early.")
            break

    # Evaluation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.float())  # Ensure that inputs are floats
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Accuracy on test set: {100 * correct / total:.2f}%")


# Save the model's weights
torch.save(model.state_dict(), r"C:\Users\Seiven\Desktop\MY_MLmodels\ions2\FFNNCombined_WandB_Saved.pth")
