import h5py
import numpy as np
import matplotlib.pyplot as plt

fn = r'C:\Users\Seiven\Desktop\MY_MLmodels\ions2\binary\cropped_ions.h5'
f = h5py.File(fn)

# Define the list of dataset keys to plot
dataset_keys = ['dark_0', 'dark_10', 'dark_1340', 'dark_1341', 'dark_1342', 'dark_1226', 'dark_1229', 'dark_1225', 'dark_1259', 'dark_1000', 'dark_0']

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
