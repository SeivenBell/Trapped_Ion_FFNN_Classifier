import os
import sys
import h5py
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.optim import Adam, lr_scheduler
import copy


# Define your model architecture (this should match the architecture used during training)
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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, x):  # Defining the forward through the network
        x = x.float()
        x = x.reshape(-1, self.N, self.N_i)
        y = torch.einsum("nij,...ni->...nj", self.W, x) + self.b
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


class Coupler(Module):
    def __init__(self, N, N_i):
        super().__init__()

        self.N = N
        self.N_i = N_i
        self.dense = nn.Linear(N * N_i, N * N_i)
        self.dense2 = nn.Linear(N * N_i, N * N_i)
        self.dense3 = nn.Linear(N * N_i, N * N_i)
        self._reset_parameters()

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


# Define the model parameters
N = 4
L_x = 5
L_y = 5
N_i = L_x * L_y
N_h = 256
N_o = 2

# Initialize the model
encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)



#################################################################################################################
# Load the pre-trained model weights
model_path = "outputs_dir/local_enchanthed_model.pth"
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the labelled dataset
labelled_dataset_path = "binary/new_reformatted_data.h5"
with h5py.File(labelled_dataset_path, "r") as f:
    x = f.get("measurements")[:]
    y = f.get("labels")[:]

# Convert to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# Ensure y is the correct shape and contains proper indices
y = y.squeeze().long()


# Define the accuracy function
def calculate_accuracy(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=-1)
        accuracy = (predictions == y).float().mean()
    return accuracy.item()


# Calculate and print accuracy
accuracy = calculate_accuracy(model, x, y)
print(f"Accuracy on the labelled dataset: {accuracy:.4f}")
