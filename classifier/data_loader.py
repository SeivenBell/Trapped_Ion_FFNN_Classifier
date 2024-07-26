# Load full dataset
full_dataset_path = "binary/fully_reformatted_dataset_standardized.h5"

full_dataset = Labelled_Dataset(full_dataset_path)
print(len(full_dataset))
data, labels = full_dataset[0]  # Get the first sample from the dataset
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 200
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Load halfpi dataset
halfpi_dataset_path = "binary/fully_reformatted_dataset_halfpi_standardized.h5"
halfpi_dataset = Unlabelled_Dataset(halfpi_dataset_path)
print(len(halfpi_dataset))  # PRINT
data = halfpi_dataset[0]  # Get the first sample from the dataset
print("halfpi Data shape:", data.shape)

halfpi_train_size = int((1 - val_ratio) * len(halfpi_dataset))
halfpi_val_size = len(halfpi_dataset) - halfpi_train_size

halfpi_train_dataset, halfpi_val_dataset = random_split(
    halfpi_dataset, [halfpi_train_size, halfpi_val_size]
)

halfpi_train_loader = DataLoader(
    halfpi_train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
halfpi_val_loader = DataLoader(
    halfpi_val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Example of getting input size from the dataset
input_sample, _ = full_dataset[0]  # Get the first sample from the dataset
input_size = input_sample.shape  # Get the shape of the first sample
print("Input size:", input_size)

print("Data loading is done")
