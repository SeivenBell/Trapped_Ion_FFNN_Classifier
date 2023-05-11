import h5py
import numpy as np
import matplotlib.pyplot as plt


def crop_ions(image, coords, crop_size=5):
    half_crop_size = crop_size // 2
    ion_crops = []
    for x, y in coords:
        crop = image[y-half_crop_size:y+half_crop_size+1, x-half_crop_size:x+half_crop_size+1]
        ion_crops.append(crop)
    return ion_crops


input_file_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/img.h5"
output_file_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/cropped_ions.h5"

ion_coordinates = [(15, 6), (23, 6), (30, 6), (38, 6)]

with h5py.File(input_file_path, "r") as input_file, h5py.File(output_file_path, "w") as output_file:
    keys = sorted(input_file.keys(), key=lambda x: int(x.split('_')[-1]))
    for key in keys:
        image = input_file[key][()]
        ion_crops = crop_ions(image, ion_coordinates)

        for i, ion_crop in enumerate(ion_crops):
            new_key = f"{key}_ion_{i}"
            output_file.create_dataset(new_key, data=ion_crop)


# Check the shapes and display the first few images
with h5py.File(output_file_path, "r") as output_file:
    keys = list(output_file.keys())
    print(f"Total number of cropped images: {len(keys)}")

    for i, key in enumerate(keys[:16]):
        image = output_file[key][()]
        print(f"{key}: {image.shape}")

        plt.subplot(4, 4, i + 1)
        plt.imshow(image, cmap='viridis')  # Change the colormap to your preference
        plt.axis('off')
        plt.title(key, fontsize=8)

    plt.show()



