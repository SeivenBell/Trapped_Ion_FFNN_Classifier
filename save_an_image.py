import os
import h5py
import numpy as np
from PIL import Image
import cv2


def save_image(filepath, keyS, output_pathS):
    with h5py.File(filepath, "r") as f:
        image_data = f[keyS][:]
        image_data = (image_data * 255).astype(np.uint8)
        cv2.imwrite(output_pathS, image_data)


data_path = r"C:\Users\Seiven\Desktop\MY_MLmodels\ions2\img.h5"
key = "bright_11"
output_path = r"C:\Users\Seiven\Desktop\MY_MLmodels\ions2\test_image.png"

save_image(data_path, key, output_path)
