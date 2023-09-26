import torch

print("loading...")
images = torch.load("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_combined_images.pt")
labels = torch.load("C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_combined_labels.pt")

print(images.size())
print(labels.size())