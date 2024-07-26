# main.py
import torch
from config import BATCH_SIZE, DEVICE
from data_loader import load_dataset
from model import Encoder, Classifier, MultiIonReadout, Coupler, EnhancedMultiIonReadout
from train_eval import train_model, evaluate_model
from utils import filter_and_compute_mean
import matplotlib.pyplot as plt

# Load data
train_loader = load_dataset(train_dataset, BATCH_SIZE)
val_loader = load_dataset(val_dataset, BATCH_SIZE, shuffle=False)
halfpi_train_loader = load_dataset(halfpi_train_dataset, BATCH_SIZE)
halfpi_val_loader = load_dataset(halfpi_val_dataset, BATCH_SIZE, shuffle=False)

# Define models
encoder = Encoder()
classifier = Classifier()
model = MultiIonReadout(encoder, classifier)
coupler = Coupler()
enhanced_model = EnhancedMultiIonReadout(model, coupler)

# Train model
train_model(model, train_loader, val_loader)
train_model(enhanced_model, halfpi_train_loader, halfpi_val_loader)

# Evaluate and plot mean images
halfpi_data = torch.stack([halfpi_dataset[i] for i in range(len(halfpi_dataset))]).to(
    DEVICE
)
states = ["0011", "0101", "1010", "0110", "1001", "1100"]
mean_images = filter_and_compute_mean(halfpi_data, enhanced_model, states)

# Plotting the mean images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, state in enumerate(states):
    ax = axes[idx // 3, idx % 3]
    mean_image = mean_images[state]
    if mean_image is not None:
        concatenated_image = mean_image.reshape(20, 5)
        ax.imshow(concatenated_image, cmap="viridis", aspect="auto")
        ax.set_title(f"{state}")
    else:
        ax.set_title(f"No matching images for State {state}")
fig.tight_layout()
plt.show()
