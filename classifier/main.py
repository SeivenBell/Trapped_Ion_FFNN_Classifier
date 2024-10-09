import torch
from torch.utils.data import DataLoader, random_split
from datasets import LabelledDataset, UnlabelledDataset
from models import (
    Encoder,
    Classifier,
    MultiIonReadout,
    Coupler,
    EnhancedMultiIonReadout,
)
from train import train_model, train_enhanced_model
from config import config
from utils import plot_nn_performance, check_dataset
from torchviz import make_dot


def main():
    # Check presence and shapes of datasets
    print("Checking the full dataset:")
    full_dataset_path = config["full_dataset_path"]
    full_measurements_shape, full_labels_shape = check_dataset(full_dataset_path)

    print("\nChecking the halfpi dataset:")
    halfpi_dataset_path = config["halfpi_dataset_path"]
    halfpi_measurements_shape, _ = check_dataset(halfpi_dataset_path)

    # Load the datasets
    if full_measurements_shape and full_labels_shape:
        full_dataset = LabelledDataset(full_dataset_path)
    else:
        print("Error: Full dataset is not properly loaded.")
        return

    if halfpi_measurements_shape:
        halfpi_dataset = UnlabelledDataset(halfpi_dataset_path)
    else:
        print("Error: Halfpi dataset is not properly loaded.")
        return

    # Splitting the full_dataset into train, validation, and test sets
    total_size = len(full_dataset)
    test_ratio = 0.1
    val_ratio = 0.2
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * (total_size - test_size))
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize models
    N, N_i, N_h, N_o = 4, 25, 256, 2
    encoder = Encoder(N, N_i, N_h)
    classifier = Classifier(N, N_h, N_o)
    model = MultiIonReadout(encoder, classifier)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_params"]["lr"])
    schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)

    # Train base model
    train_model(
        model,
        train_loader,
        val_loader,
        device=config["device"],
        epochs=config["train_params"]["N_epochs"],
        lr=config["train_params"]["lr"],
        schedule=schedule,
        log_every=config["log_every"],
    )

    # Test the trained model on the test set
    test_sia = torch.tensor(
        [
            model.eval().sia(
                databatch[0].to(config["device"]), databatch[1].to(config["device"])
            )
            for databatch in test_loader
        ]
    ).mean()
    print(f"\nTest Accuracy: {test_sia:.3f}")

    # Enhance the model with Coupler
    coupler = Coupler(N, N_h)
    enhanced_model = EnhancedMultiIonReadout(model, coupler)

    # Prepare the halfpi dataset loaders
    halfpi_train_size = int((1 - val_ratio) * len(halfpi_dataset))
    halfpi_val_size = len(halfpi_dataset) - halfpi_train_size

    halfpi_train_dataset, halfpi_val_dataset = random_split(
        halfpi_dataset, [halfpi_train_size, halfpi_val_size]
    )

    halfpi_train_loader = DataLoader(
        halfpi_train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    halfpi_val_loader = DataLoader(
        halfpi_val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Define optimizer for the enhanced model
    enhanced_optimizer = torch.optim.Adam(
        enhanced_model.parameters(),
        lr=config["enhanced_train_params"]["lr"],
        weight_decay=config["enhanced_train_params"]["weight_decay"],
    )
    enhanced_schedule = torch.optim.lr_scheduler.ConstantLR(
        enhanced_optimizer, factor=1
    )

    # Train the enhanced model
    train_enhanced_model(
        enhanced_model,
        halfpi_train_loader,
        halfpi_val_loader,
        device=config["device"],
        epochs=config["enhanced_train_params"]["N_epochs"],
        lr=config["enhanced_train_params"]["lr"],
        weight_decay=config["enhanced_train_params"]["weight_decay"],
        schedule=enhanced_schedule,
        log_every=config["log_every"],
    )

    # Plot the performance of the models
    halfpi_data = torch.stack(
        [halfpi_dataset[i] for i in range(len(halfpi_dataset))]
    ).to(config["device"])
    plot_nn_performance(enhanced_model, model, halfpi_data, N)

    sample_input = torch.randn(1, N, N_i)
    output = model(sample_input)
    dot = make_dot(output, params=dict(list(model.named_parameters())))
    dot.format = "png"
    dot.render("model_architecture")


if __name__ == "__main__":
    main()
