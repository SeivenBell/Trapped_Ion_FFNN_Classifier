import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))
print(torch.cuda.is_available())

class IonDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.labels = []

        with h5py.File(self.file_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x.split('_')[-1]))

            for key in self.keys:
                img = f[key][()]
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                label = 1 if 'bright' in key else 0

                self.data.append(img)
                self.labels.append(torch.tensor(label, dtype=torch.long))

        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(5 * 5, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
data_file_path = 'C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5'
dataset = IonDataset(data_file_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

model = FFNN().to(device)
weights = torch.tensor([1.0, 3.5]).to(device)  # Adjust the weights as needed
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
num_epochs = 200
best_val_loss = float('inf')
early_stopping_patience = 10
no_improvement_epochs = 0

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    bright_correct = 0
    dark_correct = 0
    bright_total = 0
    dark_total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            bright_mask = labels == 1
            dark_mask = labels == 0
            bright_correct += (predicted[bright_mask] == labels[bright_mask]).sum().item()
            dark_correct += (predicted[dark_mask] == labels[dark_mask]).sum().item()
            bright_total += bright_mask.sum().item()
            dark_total += dark_mask.sum().item()
    return running_loss, correct, total, bright_correct, dark_correct, bright_total, dark_total

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss, correct, total, bright_correct, dark_correct, bright_total, dark_total = validate(model, val_loader, criterion)

    bright_state_accuracy = 100 * bright_correct / bright_total
    dark_state_accuracy = 100 * dark_correct / dark_total
    current_val_loss = val_loss / len(val_loader)
    scheduler.step(current_val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {100 * correct / total:.2f}%, '
          f'Bright State Accuracy: {bright_state_accuracy:.2f}%, '
          f'Dark State Accuracy: {dark_state_accuracy:.2f}%')

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), 'best_model_V2.pth')
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs == early_stopping_patience:
            print('Early stopping due to no improvement after {} epochs.'.format(early_stopping_patience))
            break

# Final evaluation on validation set
val_loss, correct, total, bright_correct, dark_correct, bright_total, dark_total = validate(model, val_loader, criterion)
bright_state_accuracy = 100 * bright_correct / bright_total
dark_state_accuracy = 100 * dark_correct / dark_total
print(f'Final Validation: Val Loss: {val_loss / len(val_loader):.4f}, '
      f'Val Accuracy: {100 * correct / total:.2f}%, '
      f'Bright State Accuracy: {bright_state_accuracy:.2f}%, '
      f'Dark State Accuracy: {dark_state_accuracy:.2f}%')


class IonDatasetWithKeys(IonDataset):
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            key = self.keys[idx]
            img = f[key][()]
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            label = 1 if 'bright' in key else 0
            return img, torch.tensor(label, dtype=torch.long), key

def predict_ion_states(model, dataset):
    ion_states = defaultdict(int)
    model.eval()

    with torch.no_grad():
        for img, _, key in dataset:
            if 'halfpi' not in key:
                continue

            img = img.unsqueeze(0).to(device)
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            state = predicted.item()
            ion_states[key] += state

    return ion_states


def count_combined_states(ion_states, num_ions=4):
    combined_states = defaultdict(int)

    for i in range(len(ion_states) // num_ions):
        state = ''
        for j in range(num_ions):
            ion_key = f"halfpi_{i}_ion_{j}"
            state += str(ion_states[ion_key])

        combined_states[state] += 1

    return combined_states

# Load the model
model = FFNN().to(device)
model.load_state_dict(torch.load('best_model_V2.pth'))
model.eval()

# Create dataset for predictions
prediction_dataset = IonDatasetWithKeys(data_file_path)

# Use the model to predict the state of each ion in the "halfpi_" images
ion_states = predict_ion_states(model, prediction_dataset)

# Count the occurrences of each unique state
combined_states = count_combined_states(ion_states)

# Create a graph displaying the frequency of each state
states = list(combined_states.keys())
counts = list(combined_states.values())

states = sorted(combined_states.keys(), key=lambda x: int(x, 2))

# Get the counts in the same order
counts = [combined_states[state] for state in states]

plt.bar(states, counts)
plt.xlabel('States')
plt.ylabel('Frequency')
plt.title('Frequency of Ion States in Unlabeled Images')
plt.show()