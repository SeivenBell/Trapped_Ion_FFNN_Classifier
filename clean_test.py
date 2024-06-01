import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from FFNN_model import Encoder, Classifier, SimpleModel, MultiIonReadout, SharedEncoder
from collections import Counter
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from icecream import ic
import numpy as np
from torch.nn import functional as F


class UnlabelledDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]
        # self.labels = self.file["labels"] no labels in the unlabelled_dataset

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])
        x = x - 200  # , dtype=torch.float32)
        # y = torch.tensor(self.labels[idx])  # , dtype=torch.float32)
        return x

    def close(self):
        self.file.close()


batch_size = 100
unlabelled_file_path = "binary/reformatted_dataset_halfpi.h5"
unlabelled_dataset = UnlabelledDataset(unlabelled_file_path)
unlabelled_loader = DataLoader(unlabelled_dataset, batch_size, shuffle=False)
# Check the length of the dataset
print("Length of the unlabelled dataset:", len(unlabelled_dataset))
print(len(unlabelled_dataset))  # PRINT

# -----------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device))

N = 4  # number of images in whole picture
N_i = 25  # num of features in each image
N_h = 128  # hidden units
N_o = 1  # output units (binary classification) for each image in the picture (N)
N_c = N * N_h

device = torch.device("cpu")
encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = SimpleModel(encoder, classifier)

model.load_state_dict(torch.load("./Super_model.pth"))
model.to(device)
model.train()

shared_encoder = SharedEncoder(N_c, N_c)
new_model = MultiIonReadout(
    encoder=model.encoder, shared_encoder=shared_encoder, classifier=model.classifier
)
new_model.to(device)
new_model.train()

# freez the weights here:
for p in new_model.encoder.parameters():
    p.requires_grad = False
for p in new_model.classifier.parameters():
    p.requires_grad = False
####################################################################################


def gumbel_to_onehot(gumbel_output):
    _, indices = gumbel_output.max(dim=-1)
    onehot = F.one_hot(indices, num_classes=2).float()
    return onehot


def count_states(onehot_states):
    states = onehot_states.view(onehot_states.size(0), -1)
    unique_states, counts = torch.unique(states, dim=0, return_counts=True)
    return unique_states, counts


def custom_loss(gumbel_output, target_distribution):
    print(gumbel_output.requires_grad)
    onehot_states = gumbel_to_onehot(gumbel_output)
    unique_states, counts = count_states(onehot_states)

    # Calculate the observed distribution of states
    observed_distribution = counts.float() / counts.sum()

    # Calculate the KL divergence between the observed and target distributions
    loss = F.kl_div(
        observed_distribution.log(), target_distribution, reduction="batchmean"
    )

    return loss


# Define optimizer
optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-2)

# Training loop to fine-tune the model
num_epochs = 1
num_classes = 2**N  # Number of epochs for fine-tuning
for epoch in range(num_epochs):
    epoch_predictions = []
    for inputs in unlabelled_loader:
        inputs = inputs.to(device)
        predictions = new_model(inputs)

        target_distribution = torch.ones(16) / 16

        # Calculate the custom loss
        loss = custom_loss(predictions, target_distribution)

        # print(f"Custom loss: {loss.item()}")

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("Loss before backward:", loss.item())


# Make predictions on the original unlabelled data
predictions = []
with torch.no_grad():
    for inputs in unlabelled_loader:
        inputs = inputs.to(device)
        outputs = new_model(inputs)
        predictions.append(outputs.cpu().numpy())

flat_predictions = np.concatenate(predictions, axis=0)
binary_preds = (flat_predictions > 0.5).astype(int)
binary_strings = ["".join(map(str, preds)) for preds in binary_preds.reshape(-1, 4)]
tensor_str_counts = Counter(binary_strings)

labels = [label for label in tensor_str_counts.keys()]
counts = [count for count in tensor_str_counts.values()]

normalized_counts = [x / len(binary_strings) for x in counts]
uniform_counts = [1 / len(labels) for _ in labels]
ic(uniform_counts)
ic(normalized_counts)

loss = sqrt(mean_squared_error(normalized_counts, uniform_counts))
# ic(f"Root mean squared error: {loss}")

sorted_labels, sorted_counts = zip(*sorted(zip(labels, counts), key=lambda x: x[1]))
sorted_labels = list(sorted_labels)
sorted_counts = list(sorted_counts)

plt.figure(figsize=(15, 8))
sns.barplot(x=sorted_labels, y=sorted_counts)

plt.title("Count of Each Unique Class in binary_preds")
plt.xlabel("Unique Classes")
plt.ylabel("Count")
plt.xticks(rotation=25)
plt.show()
