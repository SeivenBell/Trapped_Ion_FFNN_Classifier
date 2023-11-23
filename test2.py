import torch

import torch

# Load the file
file_path = "binary/halfpi.pt"
data = torch.load(file_path)

# Assuming data is a dictionary, we need to know its keys
print("Keys in loaded data:", data.keys())

# Replace 'your_key_here' with the actual key you want to inspect
# For example, if you're interested in checking the size of model weights, 
# you should find the key corresponding to those weights
key_to_inspect = 'your_key_here'  # Replace with the actual key
tensor = data[key_to_inspect]

# Check the size of the tensor
print("Size of the tensor:", tensor.size())

# Verify if it matches the expected size
expected_size = torch.Size([4, 256, 1])
if tensor.size() == expected_size:
    print("Tensor size matches the expected size.")
else:
    print("Tensor size does not match the expected size.")


#I want to check the size of the images and labels tensors, but I get the following error: