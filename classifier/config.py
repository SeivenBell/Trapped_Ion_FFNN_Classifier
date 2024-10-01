config = {
    "full_dataset_path": r"/home/seiven/Documents/Github/Trapped_Ion_FFNN_Classifier/binary/combined_labelled_data.h5",
    "halfpi_dataset_path": r"/home/seiven/Documents/Github/Trapped_Ion_FFNN_Classifier/binary/combined_halfpi_data.h5",
    "batch_size": 250,
    "val_ratio": 0.2,
    "train_params": {"N_epochs": 25, "lr": 0.0003512337837381173},
    "enhanced_train_params": {
        "N_epochs": 15,
        "lr": 0.00016388181712790806,
        "weight_decay": 5.6547937254492916e-5,
    },
    "device": "cpu",
    "lr_schedule": "lr_schedule",
    "log_every": 1,
}
