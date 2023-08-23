# ions2
Creation of FFNN for trapped ions state classification 
exlanation:
What do we do in ionsFNNv2.py:

This code defines two classes and some training setup. Here's what each part does:

1. `IonDataset` class: This is a custom PyTorch `Dataset` that's used to load and preprocess your data. It reads a file in the HDF5 format, which is a format designed to store and organize large amounts of data. For each key in the file, it retrieves the associated image, converts it into a tensor, adds a dimension (using `unsqueeze(0)`), and appends it to the `data` list. It also determines the label for each image based on the key name, converts it into a tensor, and appends it to the `labels` list.

2. `FFNN` class: This is a simple feed-forward neural network (FFNN) model. It takes an input of size 5*5, applies a linear transformation followed by a ReLU activation function, and then applies another linear transformation. The output size is 2, representing the two classes of your classification problem (bright and dark states).

(The -1 in the view() method means that that dimension will be automatically calculated based on the size of the input and the other dimensions specified.)

The loss function has been changed to BCEWithLogitsLoss(), which is suitable for binary classification problems. It combines a Sigmoid layer and the BCELoss in one single class, which is more numerically stable than using them separately.


3. The rest of the code is setting up and preparing for training the model:
   - The paths to the data files are set.
   - The dataset is split into a training set and a validation set, with 80% of the data used for training and the rest for validation.
   - The training and validation data are loaded into `DataLoader`s, which are PyTorch utilities that help iterate over the data in batches.
   - The model is instantiated and moved to the GPU if one is available.
   - The loss function (cross-entropy loss) is set up with class weights to handle class imbalance.
   - The optimizer (Adam) and the learning rate scheduler (ReduceLROnPlateau) are set up.

4. The `validate` function: This function is used to evaluate the model on the validation set. It calculates the loss and the accuracy of the predictions, both overall and separately for the bright and dark classes. The `model.eval()` line tells PyTorch to set the model to evaluation mode, which turns off features like dropout that are used during training but not during evaluation. The `with torch.no_grad():` block tells PyTorch not to calculate gradients during the following operations, since we're not going to be doing backpropagation (which is not needed during evaluation).



SO, i moved from this:
class IonImagesDataset(Dataset):
    def __init__(self, file_path, ion_position, start_idx=0, end_idx=None):
        with h5py.File(file_path, "r") as f:
            # Filter keys based on ion_position and whether they are bright/dark or halfpi
            category = "bright" if end_idx <= BRIGHT_DARK_SIZE else "halfpi"
            self.keys = [key for key in f.keys() if f"ion_{ion_position}" in key and category in key][start_idx:end_idx]
            self.images = [np.array(f[key]) for key in self.keys]
            self.labels = [self.get_label(key) for key in self.keys]
            print(len(self.images))
            #print(f"Length of labels: {len(self.labels)}")  # Debugging print statement
            
            # Check that the lengths of images and labels are the same
            assert len(self.images) == len(self.labels), "Length mismatch between images and labels"
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = [
            torch.from_numpy(self.images[(idx + i) % len(self.images)]).float()
            for i in range(4)
        ]
        images = torch.stack(images, dim=0)
        return images, self.labels[idx] * torch.ones((4,1))
    
    def get_label(self, key):
        if "bright" in key:
            return 1
        elif "dark" in key:
            return 0
        elif "halfpi" in key:
            return 1  # or whatever logic you want to use to determine the label
        else:
            raise ValueError(f"Unknown label for key: {key}")


# Constants for the number of images per category
BRIGHT_DARK_SIZE = 40000
HALFPI_SIZE = 40000
TRAIN_SIZE = int(0.8 * BRIGHT_DARK_SIZE)
VAL_SIZE = BRIGHT_DARK_SIZE - TRAIN_SIZE

train_datasets = []
val_datasets = []
test_datasets = []
for ion_position in range(4):
    # Create a dataset for bright and dark images
    bright_dark_ds = IonImagesDataset(file_path, ion_position, start_idx=0, end_idx=BRIGHT_DARK_SIZE)
    # Split bright and dark dataset into training and validation
    train_ds = Subset(bright_dark_ds, range(0, TRAIN_SIZE))
    val_ds = Subset(bright_dark_ds, range(TRAIN_SIZE, BRIGHT_DARK_SIZE))
    # Create a dataset for halfpi images
    test_ds = IonImagesDataset(file_path, ion_position, start_idx=BRIGHT_DARK_SIZE, end_idx=2 * BRIGHT_DARK_SIZE)
    
    train_datasets.append(train_ds)
    val_datasets.append(val_ds)
    test_datasets.append(test_ds)

# Concatenate the datasets across ion positions
full_train_ds = ConcatDataset(train_datasets)
full_val_ds = ConcatDataset(val_datasets)
full_test_ds = ConcatDataset(test_datasets)


# Create DataLoaders for the training, validation, and testing datasets
batch_size = 1000
train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(full_val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(full_test_ds, batch_size=batch_size, shuffle=False, drop_last=True)


To way simpler way of loading data and I focused more on just bright and dark states to train. Will see how it turns out

Also, I still had a various problems with the getitem and other dataloader parts like:
It seems the error is still occurring because the elements inside the channels list are NumPy arrays, and torch.stack expects PyTorch Tensors. You can convert the NumPy arrays to PyTorch Tensors before stacking them.

Here's how you can modify the __getitem__ method to do that:


Certainly! Let's dive into the details of the model architecture, breaking down the various components and explaining them in layman's terms:

### 1. **`IndexDependentDense` Class (Custom Fully Connected Layer)**:
This is a custom building block that is similar to a regular fully connected (dense) layer in a neural network, but with a twist. It's designed to handle different weights for each index (ion position), allowing for more flexibility in learning the relationships between the features.

### 2. **`Encoder` Class (Encoding Part of the Model)**:
The encoder's job is to transform the input data into a new representation, making it easier for the model to understand and learn the patterns in the data. Think of it as translating a complex language into a simpler one that the model can better understand.

In this specific case, the encoder takes the images of ions and translates them into a new form that captures the essential information about the ions but in a more condensed and digestible format.

### 3. **`Classifier` Class (Classification Part of the Model)**:
After the encoder has transformed the data, the classifier takes over. Its job is to decide what category the data belongs to. In this case, it's trying to classify whether the image represents a bright or dark ion.

It does this by taking the transformed data from the encoder and passing it through another custom layer (similar to `IndexDependentDense`) followed by a sigmoid activation function. The sigmoid function ensures that the output is between 0 and 1, representing the probability of the image being a bright ion.

### 4. **`SharedEncoder` and `SharedClassifier` Classes**:
These classes are similar to the `Encoder` and `Classifier` but use standard linear (fully connected) layers. They are defined in the code but not used in the main model.

### 5. **`MultiIonReadout` Class (Main Model)**:
This class brings everything together. It's the main model that you'll be training, and it consists of the encoder and classifier mentioned earlier.

Here's how it works step by step:
   - **Input**: A batch of images, each representing an ion.
   - **Step 1 (Reshaping)**: The images are reshaped into a format that the model can work with.
   - **Step 2 (Encoding)**: The reshaped images are passed through the encoder, transforming them into a new representation.
   - **Step 3 (Classification)**: The transformed data is passed through the classifier, determining whether each image represents a bright or dark ion.
   - **Output**: A probability value for each image, representing the likelihood that it is a bright ion.

### 6. **`EnhancedMultiIonReadout` Class**:
This class is an enhanced version of `MultiIonReadout`, but it's not used in the current code. It seems to be intended for further experimentation or extension of the model.

### Summary:
The model's architecture is designed to process images of ions and determine whether they represent bright or dark ions. It does this through a series of transformations (encoding) followed by a classification step. The custom layers provide flexibility in handling different ion positions, and the combination of encoding and classification allows the model to learn complex patterns in the data.



some prints:

# def visualize_images(dataset, title):
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#     fig.suptitle(title)
#     axes = axes.flatten()  # Flatten the axes array
#     for i in range(4):
#         sample_idx = np.random.randint(len(dataset))
#         print(sample_idx, len(dataset))
#         images, labels = dataset[sample_idx]
#         print(dataset[sample_idx])
#         label = labels[0].item()  # Take the first element of the label tensor
#         axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
#         axes[i].set_title(f"Label: {label}")
#         axes[i].axis('off')
#     plt.show()


# def check_data_distribution(dataset, dataset_name):
#     labels_count = {0: 0, 1: 0}
#     for _, labels in dataset:
#         for label in labels.squeeze():  # Iterate through the batch of labels
#             labels_count[label.item()] += 1
#     print(f"{dataset_name} Labels Distribution: {labels_count}")

# visualize_images(train_dataset, "Training Dataset")
# visualize_images(val_dataset, "Validation Dataset")
# check_data_distribution(train_dataset, "Training Dataset")
# check_data_distribution(val_dataset, "Validation Dataset")
