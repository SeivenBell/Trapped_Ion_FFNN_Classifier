import sys
import numpy as np
import torch
from torch.optim import lr_scheduler


def train_model(
    model, train_loader, val_loader, device, epochs, lr, schedule, log_every
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    schedule = lr_scheduler.ConstantLR(optimizer, factor=1)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, databatch in enumerate(train_loader):
            optimizer.zero_grad()
            xbatch, ybatch = databatch
            xbatch = xbatch.to(device=device)
            ybatch = ybatch.to(device=device)

            batch_loss = model.train().nllloss(xbatch, ybatch)
            batch_loss.backward()

            running_loss += batch_loss

            optimizer.step()

        step += 1
        running_loss /= i + 1
        schedule.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            val_loss = torch.tensor(
                [
                    model.eval().nllloss(
                        databatch[0].to(device=device), databatch[1].to(device=device)
                    )
                    for databatch in val_loader
                ]
            ).mean()

            sia = torch.tensor(
                [
                    model.eval().sia(
                        databatch[0].to(device=device), databatch[1].to(device=device)
                    )
                    for databatch in val_loader
                ]
            ).mean()

            # Fancy progress display
            print(
                "{:<180}".format(
                    "\r"
                    + "[{:<60}] ".format(
                        "=" * ((np.floor((epoch + 1) / epochs * 60)).astype(int) - 1)
                        + ">"
                        if epoch + 1 < epochs
                        else "=" * 60
                    )
                    + "{:<40}".format(
                        "Epoch {}/{}: NLL Loss(Train) = {:.3g}, NLL Loss(Val) = {:.3g}, Accuracy(Val) = {:.3f}".format(
                            epoch + 1, epochs, running_loss, val_loss, sia
                        )
                    )
                ),
                end="",
            )
            sys.stdout.flush()


def train_enhanced_model(
    enhanced_model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    schedule,
    log_every,
):
    optimizer = torch.optim.Adam(
        enhanced_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    step = 0
    schedule = lr_scheduler.ConstantLR(optimizer, factor=1)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, xbatch in enumerate(train_loader):
            optimizer.zero_grad()
            xbatch = xbatch.to(device=device)
            xbatch.requires_grad_(True)

            batch_loss = enhanced_model.train().corrloss(xbatch)
            batch_loss.backward()

            if batch_loss.isnan():
                break

            running_loss += batch_loss

            optimizer.step()

        step += 1
        running_loss /= i + 1
        schedule.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            val_loss = torch.tensor(
                [
                    enhanced_model.eval().corrloss(databatch.to(device=device))
                    for databatch in val_loader
                ]
            ).mean()

            sia = torch.tensor(
                [
                    enhanced_model.eval().sia(
                        databatch[0].to(device=device), databatch[1].to(device=device)
                    )
                    for databatch in val_loader
                ]
            ).mean()

            # Fancy progress display
            print(
                "{:<180}".format(
                    "\r"
                    + "[{:<60}] ".format(
                        "=" * ((np.floor((epoch + 1) / epochs * 60)).astype(int) - 1)
                        + ">"
                        if epoch + 1 < epochs
                        else "=" * 60
                    )
                    + "{:<40}".format(
                        "Epoch {}/{}: Correlation Loss(Train) = {:.3f}, Correlation Loss(Val) = {:.3f}, Accuracy(All Bright & Dark) = {:.3f}".format(
                            epoch + 1, epochs, running_loss, val_loss, sia
                        )
                    )
                ),
                end="",
            )
            sys.stdout.flush()
