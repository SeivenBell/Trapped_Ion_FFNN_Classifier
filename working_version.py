import os
import sys

sys.path.append(os.path.abspath("../"))

########################################################################################

import numpy as np

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.optim import Adam
from torch.optim import lr_scheduler
from collections import Counter
import h5py
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from torchinfo import summary

import h5py
import optuna

import copy

#######################################################################################

# Package parameters

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 30
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


########################################################################################


class Labelled_Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]
        self.labels = self.file["labels"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])
        # x = x - 200  # Subtracting 200 as in the original code
        y = torch.tensor(self.labels[idx])
        return x, y

    def close(self):
        self.file.close()


class Unlabelled_Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.measurements = self.file["measurements"]

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.measurements[idx])  # , dtype=torch.float32)
        # x = x - 200  # Subtracting 200 as in the original code
        return x

    def close(self):
        self.file.close()


# Load full dataset
full_dataset_path = "binary/fully_reformatted_dataset_standardized.h5"

full_dataset = Labelled_Dataset(full_dataset_path)
print(len(full_dataset))
data, labels = full_dataset[0]  # Get the first sample from the dataset
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 200
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Load halfpi dataset
halfpi_dataset_path = "binary/fully_reformatted_dataset_halfpi_standardized.h5"
halfpi_dataset = Unlabelled_Dataset(halfpi_dataset_path)
print(len(halfpi_dataset))  # PRINT
data = halfpi_dataset[0]  # Get the first sample from the dataset
print("halfpi Data shape:", data.shape)

halfpi_train_size = int((1 - val_ratio) * len(halfpi_dataset))
halfpi_val_size = len(halfpi_dataset) - halfpi_train_size

halfpi_train_dataset, halfpi_val_dataset = random_split(
    halfpi_dataset, [halfpi_train_size, halfpi_val_size]
)

halfpi_train_loader = DataLoader(
    halfpi_train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
halfpi_val_loader = DataLoader(
    halfpi_val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Example of getting input size from the dataset
input_sample, _ = full_dataset[0]  # Get the first sample from the dataset
input_size = input_sample.shape  # Get the shape of the first sample
print("Input size:", input_size)

print("Data loading is done")


########################################################################################


class IndexDependentDense(Module):
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, x):  # Defining the forward through the network
        # print("IndexDependentDense Shape of input 'X' tensor before reshape :", x.shape)
        x = x.float()
        x = x.reshape(-1, self.N, self.N_i)
        # print("IndexDependentDense Shape of input 'X' tensor after :", x.shape)
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
        # print("IndexDependentDense Shape of output 'Y' tensor in :", y.shape)
        if self.activation:
            y = self.activation(y)
        return y


class Encoder(Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o

        self.dense = IndexDependentDense(N, N_i, N_o, activation=F.relu)
        self.dense2 = IndexDependentDense(N, N_o, N_o, activation=F.relu)
        self.dense3 = IndexDependentDense(N, N_o, N_o, activation=F.relu)
        pass

    def forward(self, x):
        y = self.dense(x)
        y = self.dense2(y)
        y = self.dense3(y)
        return y

    pass


class Classifier(Module):
    def __init__(self, N, N_i, N_o):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.N_o = N_o

        self.dense = IndexDependentDense(
            N, N_i, N_o, activation=lambda x: F.log_softmax(x, dim=-1)
        )
        pass

    def forward(self, x):
        y = self.dense(x)
        return y

    pass


class Coupler(Module):
    def __init__(self, N, N_i):
        super().__init__()

        self.N = N
        self.N_i = N_i

        self.dense = nn.Linear(N * N_i, N * N_i)
        self.dense2 = nn.Linear(N * N_i, N * N_i)
        self.dense3 = nn.Linear(N * N_i, N * N_i)

        self._reset_parameters()

        pass

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.zeros_(p)
        pass

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1)
        y = self.dense(y)
        y = F.relu(y)
        y = y.reshape(*x.shape)
        return y

    pass


########################################################################################


class AbstractMultiIonReadout(Module):
    def __init__(self):
        super().__init__()

        pass

    def classify(self, x):
        return F.gumbel_softmax(logits=self(x), tau=1, hard=True)[..., -1:]

    def nllloss(self, x, y):
        _x = self(x)
        _y = F.one_hot(y.squeeze().long())

        return -(_x * _y).sum(dim=-1).mean()

    @staticmethod
    def _sia(y_pred, y_true):
        sia = (y_true == y_pred).to(dtype=torch.float32).mean()
        return sia * 100

    def sia(self, x, y):
        return self._sia(self.classify(x), y)

    @staticmethod
    def _aia(y_pred, y_true):
        aia = (y_true == y_pred).all(dim=-2).to(dtype=torch.float32).mean()
        return aia * 100

    def aia(self, x, y):
        return self._aia(self.classify(x), y)

    @staticmethod
    def _classification_report(y_pred, y_true):
        N = y_true.shape[-2]

        y = torch.cat([y_pred, y_true], dim=-1)
        y = (y * torch.flip(2 ** torch.arange(N), dims=(0,))[:, None]).sum(dim=(-2))

        uy, cy = torch.unique(y, dim=0, return_counts=True)
        uy, cy = uy.long(), cy.to(dtype=torch.float32)

        classification_report = torch.zeros((2**N, 2**N))

        classification_report[uy[:, 0], uy[:, 1]] = cy

        classification_report /= classification_report.sum((0, 1), keepdim=True)

        return classification_report

    def classification_report(self, x, y):
        return self._classification_report(self.classify(x), y)

    def corrloss(self, x):
        y = self.classify(x).squeeze()
        y = 2 * y - 1

        idcs = torch.triu_indices(y.shape[-1], y.shape[-1], 1)
        corr = (
            y[..., :, None] * y[..., None, :]
            - y[..., :, None].mean(0) * y[..., None, :].mean(0)
        )[..., idcs[0], idcs[1]]
        corr = torch.abs(corr.mean(0)).mean()

        mag = torch.abs(y.mean(0)).mean()

        return corr + mag

    @torch.no_grad()
    def counts(self, x):
        return torch.unique(self.classify(x).squeeze(), dim=0, return_counts=True)

    pass


class MultiIonReadout(AbstractMultiIonReadout):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier
        pass

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        y = self.encoder(y)
        y = self.classifier(y)
        return y


class EnhancedMultiIonReadout(AbstractMultiIonReadout):
    def __init__(self, mir, coupler):
        super().__init__()

        self.mir = copy.deepcopy(mir)
        self.coupler = coupler

        for p in self.mir.parameters():
            p.requires_grad_(False)

        pass

    def forward(self, x):
        y = x.reshape(*x.shape[:-2], -1).to(torch.float32)
        y = self.mir.encoder(y)
        y1 = self.coupler(y)
        y = y + y1
        y = self.mir.classifier(y)
        return y

    pass


########################################################################################
########################################################################################

device = torch.device("cpu")

N = 4
L_x = 5
L_y = 5
N_i = L_x * L_y
N_h = 256
N_o = 2

encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)

########################################################################################

print(
    summary(
        model,
        input_size=(batch_size, *input_size),  # Including batch size in the input size
        device=device,
    )
)
print("")


########################################################################################

N_epochs = 5
lr = 0.0003512337837381173  # Best hyperparameters:  {'lr': 0.0003912337837381173}
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

########################################################################################


step = 0
for epoch in range(N_epochs):
    running_loss = 0.0
    for i, databatch in enumerate(train_loader):
        optimizer.zero_grad()
        (xbatch, ybatch) = databatch
        xbatch = xbatch.to(device=device)
        ybatch = ybatch.to(device=device)

        batch_loss = model.train().nllloss(xbatch, ybatch)
        batch_loss.backward()

        running_loss += batch_loss

        optimizer.step()
    step += 1

    running_loss /= i + 1

    schedule.step()

    if epoch % log_every == 0 or epoch == N_epochs - 1:
        val_loss = torch.tensor(
            [
                model.eval().nllloss(
                    databatch[0].to(device=device), databatch[1].to(device=device)
                )
                for databatch in val_loader
            ]
        ).mean()

        sia = torch.tensor(
            [
                model.eval().sia(
                    databatch[0].to(device=device), databatch[1].to(device=device)
                )
                for databatch in val_loader
            ]
        ).mean()

    ########################################################################################

    print(
        "{:<180}".format(
            "\r"
            + "[{:<60}] ".format(
                "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1) + ">"
                if epoch + 1 < N_epochs
                else "=" * 60
            )
            + "{:<40}".format(
                "Epoch {}/{}: NLL Loss(Train) = {:.3g}, NLL Loss(Val) = {:.3g}, Accuracy(Val) = {:.3f}".format(
                    epoch + 1, N_epochs, running_loss, val_loss, sia
                )
            )
        ),
        end="",
    )
    sys.stdout.flush()

########################################################################################

coupler = Coupler(N, N_h)
enhanced_model = EnhancedMultiIonReadout(model, coupler)

########################################################################################

print(
    summary(
        enhanced_model,
        input_size=(batch_size, *input_size),  # Including batch size in the input size
        device=device,
    )
)
print("")

########################################################################################

N_epochs = 5
lr = 0.00016388181712790806  # was 1e-3
weight_decay = 5.6547937254492916e-5  # 4.6
optimizer = Adam(enhanced_model.parameters(), lr=lr, weight_decay=weight_decay)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

step = 0
for epoch in range(N_epochs):
    running_loss = 0.0
    for i, xbatch in enumerate(halfpi_train_loader):  # Correct unpacking
        optimizer.zero_grad()
        xbatch = xbatch.to(device=device)
        xbatch.requires_grad_(True)

        batch_loss = enhanced_model.train().corrloss(xbatch)
        batch_loss.backward()

        if batch_loss.isnan():
            break

        running_loss += batch_loss

        optimizer.step()
    step += 1

    running_loss /= i + 1

    schedule.step()

    if epoch % log_every == 0 or epoch == N_epochs - 1:
        val_loss = torch.tensor(
            [
                enhanced_model.eval().corrloss(databatch.to(device=device))
                for databatch in halfpi_val_loader
            ]
        ).mean()

        sia = torch.tensor(
            [
                enhanced_model.eval().sia(
                    databatch[0].to(device=device), databatch[1].to(device=device)
                )
                for databatch in val_loader
            ]
        ).mean()

        print(
            "{:<180}".format(
                "\r"
                + "[{:<60}] ".format(
                    "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1)
                    + ">"
                    if epoch + 1 < N_epochs
                    else "=" * 60
                )
                + "{:<40}".format(
                    "Epoch {}/{}: Correlation Loss(Train) = {:.3f}, Correlation Loss(Val) = {:.3f}, Accuracy(All Bright & Dark) = {:.3f}".format(
                        epoch + 1, N_epochs, running_loss, val_loss, sia
                    )
                )
            ),
            end="",
        )
        sys.stdout.flush()

################################################################################################
import torch
import matplotlib.pyplot as plt

# Ensure halfpi_data is defined correctly before this part
halfpi_data = torch.stack([halfpi_dataset[i] for i in range(len(halfpi_dataset))]).to(
    device
)


# Function to filter and compute the mean for specified states
def filter_and_compute_mean(data, model, states):
    with torch.no_grad():
        model.eval()
        predictions = (model(data) > 0.5).int()

        # Debugging: Print the shape and first few predictions
        print("Predictions shape:", predictions.shape)
        print("Predictions sample:", predictions[:10])

        # Flatten predictions and convert to binary strings
        binary_preds = [
            "".join(map(str, pred.cpu().numpy().flatten())) for pred in predictions
        ]

        # Debugging: Print the first few binary predictions
        print("Binary Predictions:", binary_preds[:10])

        selected_data = {state: [] for state in states}

        for i, pred in enumerate(binary_preds):
            if pred in states:
                selected_data[pred].append(data[i].cpu().numpy())

        mean_images = {}
        for state, imgs in selected_data.items():
            if imgs:
                imgs_tensor = torch.tensor(imgs)

                # Debugging: Print the shape of images for each state
                print(f"Images for state {state}: {imgs_tensor.shape}")

                combined_mean_image = imgs_tensor.mean(dim=0).numpy()
                mean_images[state] = combined_mean_image.reshape(
                    4, 5, 5
                )  # Combine in respect to bits

                # Debugging: Print the mean image shape
                print(f"Mean image shape for state {state}: {mean_images[state].shape}")
            else:
                mean_images[state] = None

                # Debugging: Print when no images are found for a state
                print(f"No images found for state {state}")

    return mean_images


states = ["0011", "0101", "1010", "0110", "1001", "1100"]
mean_images = filter_and_compute_mean(halfpi_data, enhanced_model, states)

# Plotting the mean images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, state in enumerate(states):
    ax = axes[idx // 3, idx % 3]
    mean_image = mean_images[state]
    if mean_image is not None:
        # Debugging: Print the state and mean image before plotting
        print(f"Plotting state {state} with mean image shape: {mean_image.shape}")

        # Concatenate 4 separate images into one state and plot it
        concatenated_image = mean_image.reshape(20, 5)
        ax.imshow(concatenated_image, cmap="viridis", aspect="auto")
        ax.set_title(f"{state}")
    else:
        ax.set_title(f"No matching images for State {state}")

fig.tight_layout()
plt.show()

# Optional: print the indices of selected images for further debugging
for state in states:
    if mean_images[state] is not None:
        print(f"State {state} has mean image with shape: {mean_images[state].shape}")
    else:
        print(f"No images found for state {state}")


################################################################################################
################################################################################################
f = enhanced_model.counts(halfpi_data)
f2 = model.counts(halfpi_data)

x = (
    f[0].to(dtype=torch.float, device=torch.device("cpu"))
    * (2 ** torch.arange(N)).flip(0)
).sum(-1)

y = f[1].to(dtype=torch.float, device=torch.device("cpu")) / len(halfpi_data)

x2 = (
    f2[0].to(dtype=torch.float, device=torch.device("cpu"))
    * (2 ** torch.arange(N)).flip(0)
).sum(-1)
y2 = f2[1].to(dtype=torch.float, device=torch.device("cpu")) / len(halfpi_data)


# Create a function to sort binary strings by the number of 1s
def sort_by_num_ones(bin_list):
    return sorted(bin_list, key=lambda x: (x.count("1"), x))


# Generate the binary strings for the x-axis
bin_strings = [
    ("{:0>" + "{}".format(N) + "}").format(str(bin(i))[2:]) for i in range(2**N)
]

# Sort the binary strings
sorted_bin_strings = sort_by_num_ones(bin_strings)

# Create a mapping from the sorted binary strings to their indices
sorted_indices = {bin_string: idx for idx, bin_string in enumerate(sorted_bin_strings)}

# Map the x and x2 values to the sorted indices
sorted_x = torch.tensor(
    [
        sorted_indices[("{:0>" + "{}".format(N) + "}").format(str(bin(int(val)))[2:])]
        for val in x
    ]
)
sorted_x2 = torch.tensor(
    [
        sorted_indices[("{:0>" + "{}".format(N) + "}").format(str(bin(int(val)))[2:])]
        for val in x2
    ]
)

# Sort the x and y values
sorted_x, sorted_y = zip(*sorted(zip(sorted_x, y.tolist())))
sorted_x2, sorted_y2 = zip(*sorted(zip(sorted_x2, y2.tolist())))

# Calculate the sum of distances to the uniform distribution
uniform_dist = 2**-N
sum_dist_enhanced = sum(abs(np.array(sorted_y) - uniform_dist))
sum_dist_original = sum(abs(np.array(sorted_y2) - uniform_dist))

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(
    sorted_x,
    sorted_y,
    "o--",
    label=f"Enhanced Model (Sum Dist: {sum_dist_enhanced:.4f})",
)
ax.plot(
    sorted_x2,
    sorted_y2,
    "o--",
    label=f"Original Model (Sum Dist: {sum_dist_original:.4f})",
)
ax.plot(range(2**N), 2**-N * np.ones(2**N), ":", label="Uniform Distribution")

ax.set_ylim([0, None])
ax.legend()

ax.set_xticks(range(2**N))
ax.set_xticklabels(sorted_bin_strings)
ax.xaxis.set_tick_params(rotation=90)

fig.tight_layout()
plt.show()

# Print the sum of distances to the uniform distribution
print(
    f"Sum of distances to uniform distribution (Enhanced Model): {sum_dist_enhanced:.4f}"
)
print(
    f"Sum of distances to uniform distribution (Original Model): {sum_dist_original:.4f}"
)
