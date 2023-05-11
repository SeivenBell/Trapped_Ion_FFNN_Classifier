import h5py
import numpy as np
import matplotlib.pyplot as plt

fn = 'img.h5'
f = h5py.File(fn)

# Define the list of dataset keys to plot
dataset_keys = ['halfpi_0', 'halfpi_10', 'dark_1340', 'dark_1341', 'bright_1342', 'halfpi_1226', 'halfpi_1229', 'halfpi_1225', 'halfpi_1259', 'bright_1000', 'bright_0']

# Create a grid of subplots with 3 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)

# Loop over the dataset keys and plot each image in a separate subplot
for i, key in enumerate(dataset_keys):
    row = i // 4
    col = i % 4
    img = f[key][:]
    ax = axs[row, col]
    ax.imshow(img)
    ax.set_title(key)

    # Display gridlines on each subplot
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1)

    # Customize x and y axis ticks
    ax.set_xticks(np.arange(0, img.shape[1], step=5))  # Change step to desired interval for x-axis
    ax.set_yticks(np.arange(0, img.shape[0], step=5))  # Change step to desired interval for y-axis

    # If you want to show the tick labels, uncomment the following lines:
    ax.set_xticklabels(np.arange(0, img.shape[1], step=5))
    ax.set_yticklabels(np.arange(0, img.shape[0], step=5))

    # If you prefer to hide the tick labels, use the following lines instead:
    """ax.set_xticklabels([])
    ax.set_yticklabels([])"""

# Adjust the spacing between subplots and show the plot
fig.tight_layout()
plt.show()
