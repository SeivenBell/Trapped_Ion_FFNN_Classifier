import h5py
import numpy as np
import matplotlib.pyplot as plt
import tqdm

bright = []  # all bright state data
dark = []  # all dark state data
halfpi = []  # all possible states
with h5py.File("img.h5", "a") as f:
    for i in tqdm.tqdm(range(10000)):
        bright.append(np.array(f["bright_{}".format(i)]))
        dark.append(np.array(f["dark_{}".format(i)]))
        halfpi.append(np.array(f["halfpi_{}".format(i)]))

plt.imshow(np.mean(bright, axis=0) - np.mean(dark, axis=0))
plt.colorbar()
plt.show()
