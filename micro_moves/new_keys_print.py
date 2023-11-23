import h5py
from matplotlib import pyplot as plt
import numpy as np

def visualize_category(file_path, category, num_samples=10):
    with h5py.File(file_path, "r") as f:
        all_keys = list(f.keys())
        category_keys = [key for key in all_keys if category in key]
        random_keys = np.random.choice(category_keys, num_samples, replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, key in enumerate(random_keys):
            image_data = f[key][()]
            axes[i].imshow(image_data, cmap='gray')
            axes[i].set_title(key)
            axes[i].axis('off')
        plt.show()

# Example usage for the "halfpi" category:
file_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5"
visualize_category(file_path, "halfpi")
# fighting for my life to fix the problem 
