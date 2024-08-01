import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_nn_performance(enhanced_model, model, halfpi_data, N):
    # Calculate counts
    f = enhanced_model.counts(halfpi_data)
    f2 = model.counts(halfpi_data)

    # Convert binary counts to decimal for plotting
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

    # Function to sort binary strings by the number of 1s
    def sort_by_num_ones(bin_list):
        return sorted(bin_list, key=lambda x: (x.count("1"), x))

    # Generate binary strings for x-axis
    bin_strings = [
        ("{:0>" + "{}".format(N) + "}").format(str(bin(i))[2:]) for i in range(2**N)
    ]

    # Sort binary strings
    sorted_bin_strings = sort_by_num_ones(bin_strings)

    # Create mapping from sorted binary strings to indices
    sorted_indices = {bin_string: idx for idx, bin_string in enumerate(sorted_bin_strings)}

    # Map x and x2 values to sorted indices
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

    # Sort x and y values
    sorted_x, sorted_y = zip(*sorted(zip(sorted_x, y.tolist())))
    sorted_x2, sorted_y2 = zip(*sorted(zip(sorted_x2, y2.tolist())))

    # Calculate sum of distances to uniform distribution
    uniform_dist = 2**-N
    sum_dist_enhanced = sum(abs(np.array(sorted_y) - uniform_dist))
    sum_dist_original = sum(abs(np.array(sorted_y2) - uniform_dist))

    # Plot the results
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
