import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from config import config


def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    schedule = lr_scheduler.ConstantLR(optimizer, factor=1)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (xbatch, ybatch) in enumerate(train_loader):
            optimizer.zero_grad()
            xbatch = xbatch.to(device)
            ybatch = ybatch.to(device)
            batch_loss = model.train().nllloss(xbatch, ybatch)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss
        running_loss /= i + 1
        schedule.step()
        if epoch % config.log_every == 0 or epoch == epochs - 1:
            val_loss = torch.tensor(
                [
                    model.eval().nllloss(xbatch.to(device), ybatch.to(device))
                    for xbatch, ybatch in val_loader
                ]
            ).mean()
            print(
                f"Epoch {epoch+1}/{epochs}: Loss(Train)={running_loss:.3f}, Loss(Val)={val_loss:.3f}"
            )
    return model


def train_enhanced_model(
    enhanced_model, train_loader, val_loader, device, epochs, lr, weight_decay
):
    optimizer = Adam(enhanced_model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = lr_scheduler.ConstantLR(optimizer, factor=1)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, xbatch in enumerate(train_loader):
            optimizer.zero_grad()
            xbatch = xbatch.to(device)
            batch_loss = enhanced_model.train().corrloss(xbatch)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss
        running_loss /= i + 1
        schedule.step()
        if epoch % config.log_every == 0 or epoch == epochs - 1:
            val_loss = torch.tensor(
                [
                    enhanced_model.eval().corrloss(xbatch.to(device))
                    for xbatch in val_loader
                ]
            ).mean()
            print(
                f"Epoch {epoch+1}/{epochs}: Correlation Loss(Train)={running_loss:.3f}, Correlation Loss(Val)={val_loss:.3f}"
            )
    return enhanced_model
