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
from utils import plot_nn_performance
from config import config


def main():
    # Load datasets
    full_dataset = LabelledDataset(config["full_dataset_path"])
    train_size = int((1 - config["val_ratio"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    halfpi_dataset = UnlabelledDataset(config["halfpi_dataset_path"])
    halfpi_train_size = int((1 - config["val_ratio"]) * len(halfpi_dataset))
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

    device = torch.device(config["device"])
    input_sample, _ = full_dataset[0]
    input_size = input_sample.shape

    # Initialize models
    N, N_i, N_h, N_o = 4, input_size[0] * input_size[1], 256, 2
    encoder = Encoder(N, N_i, N_h)
    classifier = Classifier(N, N_h, N_o)
    model = MultiIonReadout(encoder, classifier)

    # Train base model
    train_model(
        model,
        train_loader,
        val_loader,
        device,
        config["train_params"]["N_epochs"],
        config["train_params"]["lr"],
    )

    # Enhance model
    coupler = Coupler(N, N_h)
    enhanced_model = EnhancedMultiIonReadout(model, coupler)
    train_enhanced_model(
        enhanced_model,
        halfpi_train_loader,
        halfpi_val_loader,
        device,
        config["enhanced_train_params"]["N_epochs"],
        config["enhanced_train_params"]["lr"],
        config["enhanced_train_params"]["weight_decay"],
    )

    halfpi_data = torch.stack([halfpi_dataset[i] for i in range(len(halfpi_dataset))])
    plot_nn_performance(enhanced_model, model, halfpi_data, N)

    # Save models
    torch.save(model.state_dict(), "model_tests.pth")
    torch.save(enhanced_model.state_dict(), "enhanced_model_tests.pth")


if __name__ == "__main__":
    main()
