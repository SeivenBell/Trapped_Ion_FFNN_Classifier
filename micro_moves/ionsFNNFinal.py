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


hyperparameter_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'num_epochs': [50, 100, 150],
}

def train_and_validate_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    early_stopping_patience = 3
    no_improvement_epochs = 0

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

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # In the validation loop, add these counters:
        bright_correct = 0
        dark_correct = 0
        bright_total = 0
        dark_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update the counters for bright and dark states
                bright_mask = labels == 1
                dark_mask = labels == 0
                bright_correct += (predicted[bright_mask] == labels[bright_mask]).sum().item()
                dark_correct += (predicted[dark_mask] == labels[dark_mask]).sum().item()
                bright_total += bright_mask.sum().item()
                dark_total += dark_mask.sum().item()

        # Calculate the bright state and dark state accuracies
        bright_state_accuracy = 100 * bright_correct / bright_total
        dark_state_accuracy = 100 * dark_correct / dark_total
        current_val_loss = val_loss / len(val_loader)
        scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stopping_patience:
                break

    return best_val_loss, bright_state_accuracy, dark_state_accuracy

best_hyperparameters = None
best_val_loss = float('inf')

for learning_rate in hyperparameter_grid['learning_rate']:
    for batch_size in hyperparameter_grid['batch_size']:
        for num_epochs in hyperparameter_grid['num_epochs']:
            print(f"Training with learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}")

            # Create new model, criterion, optimizer, and scheduler with current hyperparameters
            model = FFNN().to(device)
            criterion = nn.CrossEntropyLoss()


data_file_path = 'C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5'
dataset = IonDataset(data_file_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

model = FFNN().to(device)
weights = torch.tensor([1.0, 3.0]).to(device)  # Adjust the weights as needed
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
num_epochs = 100
best_val_loss = float('inf')
early_stopping_patience = 10
no_improvement_epochs = 0

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

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # In the validation loop, add these counters:
    bright_correct = 0
    dark_correct = 0
    bright_total = 0
    dark_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the counters for bright and dark states
            bright_mask = labels == 1
            dark_mask = labels == 0
            bright_correct += (predicted[bright_mask] == labels[bright_mask]).sum().item()
            dark_correct += (predicted[dark_mask] == labels[dark_mask]).sum().item()
            bright_total += bright_mask.sum().item()
            dark_total += dark_mask.sum().item()

    # Calculate the bright state and dark state accuracies
    bright_state_accuracy = 100 * bright_correct / bright_total
    dark_state_accuracy = 100 * dark_correct / dark_total
    current_val_loss = val_loss / len(val_loader)
    scheduler.step(current_val_loss)

    # Print the accuracies for bright state and dark state
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {100 * correct / total:.2f}%, '
          f'Bright State Accuracy: {bright_state_accuracy:.2f}%, '
          f'Dark State Accuracy: {dark_state_accuracy:.2f}%')

    current_val_loss = val_loss / len(val_loader)
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), 'ion_detection_ffnn_bestnewV1.pth')
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping")
            break

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'ion_detection_ffnn.pth')
# ------------------------------------------------------------------
model = FFNN()
model.load_state_dict(torch.load('ion_detection_ffnn_bestnewV1.pth'))
model.eval()
model.to(device)


def predict_ion_states(model, dataset):
    ion_states = defaultdict(int)
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            key = dataset.keys[idx]
            img, _ = dataset[idx]

            if 'halfpi' not in key:
                continue

            img_tensor = img.to(device)
            output = model(img_tensor)
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


# Use the model to predict the state of each ion in the "halfpi_" images
ion_states = predict_ion_states(model, dataset)

# Count the occurrences of each unique state
combined_states = count_combined_states(ion_states)

# Create a graph displaying the frequency of each state
states = list(combined_states.keys())
counts = list(combined_states.values())

plt.bar(states, counts)
plt.xlabel('States')
plt.ylabel('Frequency')
plt.title('Frequency of Ion States in Unlabeled Images')
plt.show()
