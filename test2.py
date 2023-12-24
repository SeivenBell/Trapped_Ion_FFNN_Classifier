import torch

# import torch

# # Load the file
file_path = "binary/halfpi.pt"
data = torch.load(file_path)
data_processed = data['images'].reshape(data['images'].size(0),100)
print(data_processed.size())

# # Assuming data is a dictionary, we need to know its keys
# print("Keys in loaded data:", data.keys())

# # Replace 'your_key_here' with the actual key you want to inspect
# # For example, if you're interested in checking the size of model weights, 
# # you should find the key corresponding to those weights
# key_to_inspect = 'halfpi_9935_ion_1'  # Replace with the actual key
# tensor = data[key_to_inspect]

# # Check the size of the tensor
# print("Size of the tensor:", tensor.size())

# # Verify if it matches the expected size
# expected_size = torch.Size([4, 256, 1])
# if tensor.size() == expected_size:
#     print("Tensor size matches the expected size.")
# else:
#     print("Tensor size does not match the expected size.")



# Load the file
# model_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/golden_WandB.pth"
# state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# # Print the shape of each tensor in the state dictionary
# for name, tensor in state_dict.items():
#     print(f"Shape of {name}: {tensor.size()}")


#I want to check the size of the images and labels tensors, but I get the following error:
# golden_WandB.pth
# Shape of encoder.dense.W: torch.Size([4, 25, 256])
# Shape of encoder.dense.b: torch.Size([4, 256])
# Shape of shared_encoder.dense.weight: torch.Size([256, 25])
# Shape of shared_encoder.dense.bias: torch.Size([256])
# Shape of classifier.dense.W: torch.Size([4, 512, 1])
# Shape of classifier.dense.b: torch.Size([4, 1])