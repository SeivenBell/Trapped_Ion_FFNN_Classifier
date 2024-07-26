# config.py
BATCH_SIZE = 200
N_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 5e-5
LOG_EVERY = 1
DEVICE = "cpu"

N_epochs = 5
lr = 0.0003512337837381173  # Best hyperparameters:  {'lr': 0.0003912337837381173}
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

N_epochs = 5
lr = 0.00016388181712790806  # was 1e-3
weight_decay = 5.6547937254492916e-5  # 4.6
optimizer = Adam(enhanced_model.parameters(), lr=lr, weight_decay=weight_decay)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1
