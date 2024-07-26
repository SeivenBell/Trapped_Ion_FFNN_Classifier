import torch
import matplotlib.pyplot as plt

# Ensure halfpi_data is defined correctly before this part
halfpi_data = torch.stack([halfpi_dataset[i] for i in range(len(halfpi_dataset))]).to(
    device
)


# Function to filter and compute the mean for specified states
def filter_and_compute_mean(data, model, states):
    with torch.no_grad():
        model.eval()
        predictions = (model(data) > 0.5).int()

        # Debugging: Print the shape and first few predictions
        print("Predictions shape:", predictions.shape)
        print("Predictions sample:", predictions[:10])

        # Flatten predictions and convert to binary strings
        binary_preds = [
            "".join(map(str, pred.cpu().numpy().flatten())) for pred in predictions
        ]

        # Debugging: Print the first few binary predictions
        print("Binary Predictions:", binary_preds[:10])

        selected_data = {state: [] for state in states}

        for i, pred in enumerate(binary_preds):
            if pred in states:
                selected_data[pred].append(data[i].cpu().numpy())

        mean_images = {}
        for state, imgs in selected_data.items():
            if imgs:
                imgs_tensor = torch.tensor(imgs)

                # Debugging: Print the shape of images for each state
                print(f"Images for state {state}: {imgs_tensor.shape}")

                combined_mean_image = imgs_tensor.mean(dim=0).numpy()
                mean_images[state] = combined_mean_image.reshape(
                    4, 5, 5
                )  # Combine in respect to bits

                # Debugging: Print the mean image shape
                print(f"Mean image shape for state {state}: {mean_images[state].shape}")
            else:
                mean_images[state] = None

                # Debugging: Print when no images are found for a state
                print(f"No images found for state {state}")

    return mean_images


states = ["0011", "0101", "1010", "0110", "1001", "1100"]
mean_images = filter_and_compute_mean(halfpi_data, enhanced_model, states)

# Plotting the mean images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, state in enumerate(states):
    ax = axes[idx // 3, idx % 3]
    mean_image = mean_images[state]
    if mean_image is not None:
        # Debugging: Print the state and mean image before plotting
        print(f"Plotting state {state} with mean image shape: {mean_image.shape}")

        # Concatenate 4 separate images into one state and plot it
        concatenated_image = mean_image.reshape(20, 5)
        ax.imshow(concatenated_image, cmap="viridis", aspect="auto")
        ax.set_title(f"{state}")
    else:
        ax.set_title(f"No matching images for State {state}")

fig.tight_layout()
plt.show()

# Optional: print the indices of selected images for further debugging
for state in states:
    if mean_images[state] is not None:
        print(f"State {state} has mean image with shape: {mean_images[state].shape}")
    else:
        print(f"No images found for state {state}")


################################################################################################
################################################################################################
f = enhanced_model.counts(halfpi_data)
f2 = model.counts(halfpi_data)

x = (
    f[0].to(dtype=torch.float, device=torch.device("cpu"))
    * (2 ** torch.arange(N)).flip(0)
).sum(-1)

y = f[1].to(dtype=torch.float, device=torch.device("cpu")) / len(halfpi_data)

x2 = (
    f2[0].to(dtype=torch.float, device=torch.device("cpu"))
    * (2 ** torch.arange(N)).flip(0)
).sum(-1)
y2 = f2[1].to(dtype=torch.float, device=torch.device("cpu")) / len(halfpi_data)


# Create a function to sort binary strings by the number of 1s
def sort_by_num_ones(bin_list):
    return sorted(bin_list, key=lambda x: (x.count("1"), x))


# Generate the binary strings for the x-axis
bin_strings = [
    ("{:0>" + "{}".format(N) + "}").format(str(bin(i))[2:]) for i in range(2**N)
]

# Sort the binary strings
sorted_bin_strings = sort_by_num_ones(bin_strings)

# Create a mapping from the sorted binary strings to their indices
sorted_indices = {bin_string: idx for idx, bin_string in enumerate(sorted_bin_strings)}

# Map the x and x2 values to the sorted indices
sorted_x = torch.tensor(
    [
        sorted_indices[("{:0>" + "{}".format(N) + "}").format(str(bin(int(val)))[2:])]
        for val in x
    ]
)
sorted_x2 = torch.tensor(
    [
        sorted_indices[("{:0>" + "{}".format(N) + "}").format(str(bin(int(val)))[2:])]
        for val in x2
    ]
)

# Sort the x and y values
sorted_x, sorted_y = zip(*sorted(zip(sorted_x, y.tolist())))
sorted_x2, sorted_y2 = zip(*sorted(zip(sorted_x2, y2.tolist())))

# Calculate the sum of distances to the uniform distribution
uniform_dist = 2**-N
sum_dist_enhanced = sum(abs(np.array(sorted_y) - uniform_dist))
sum_dist_original = sum(abs(np.array(sorted_y2) - uniform_dist))

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(
    sorted_x,
    sorted_y,
    "o--",
    label=f"Enhanced Model (Sum Dist: {sum_dist_enhanced:.4f})",
)
ax.plot(
    sorted_x2,
    sorted_y2,
    "o--",
    label=f"Original Model (Sum Dist: {sum_dist_original:.4f})",
)
ax.plot(range(2**N), 2**-N * np.ones(2**N), ":", label="Uniform Distribution")

ax.set_ylim([0, None])
ax.legend()

ax.set_xticks(range(2**N))
ax.set_xticklabels(sorted_bin_strings)
ax.xaxis.set_tick_params(rotation=90)

fig.tight_layout()
plt.show()

# Print the sum of distances to the uniform distribution
print(
    f"Sum of distances to uniform distribution (Enhanced Model): {sum_dist_enhanced:.4f}"
)
print(
    f"Sum of distances to uniform distribution (Original Model): {sum_dist_original:.4f}"
)
s