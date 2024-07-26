device = torch.device("cpu")

N = 4
L_x = 5
L_y = 5
N_i = L_x * L_y
N_h = 256
N_o = 2

encoder = Encoder(N, N_i, N_h)
classifier = Classifier(N, N_h, N_o)
model = MultiIonReadout(encoder, classifier)

########################################################################################

print(
    summary(
        model,
        input_size=(batch_size, *input_size),  # Including batch size in the input size
        device=device,
    )
)
print("")


########################################################################################

N_epochs = 5
lr = 0.0003512337837381173  # Best hyperparameters:  {'lr': 0.0003912337837381173}
optimizer = Adam(model.parameters(), lr=lr)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

########################################################################################


step = 0
for epoch in range(N_epochs):
    running_loss = 0.0
    for i, databatch in enumerate(train_loader):
        optimizer.zero_grad()
        (xbatch, ybatch) = databatch
        xbatch = xbatch.to(device=device)
        ybatch = ybatch.to(device=device)

        batch_loss = model.train().nllloss(xbatch, ybatch)
        batch_loss.backward()

        running_loss += batch_loss

        optimizer.step()
    step += 1

    running_loss /= i + 1

    schedule.step()

    if epoch % log_every == 0 or epoch == N_epochs - 1:
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

    ########################################################################################

    print(
        "{:<180}".format(
            "\r"
            + "[{:<60}] ".format(
                "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1) + ">"
                if epoch + 1 < N_epochs
                else "=" * 60
            )
            + "{:<40}".format(
                "Epoch {}/{}: NLL Loss(Train) = {:.3g}, NLL Loss(Val) = {:.3g}, Accuracy(Val) = {:.3f}".format(
                    epoch + 1, N_epochs, running_loss, val_loss, sia
                )
            )
        ),
        end="",
    )
    sys.stdout.flush()

########################################################################################

coupler = Coupler(N, N_h)
enhanced_model = EnhancedMultiIonReadout(model, coupler)

########################################################################################

print(
    summary(
        enhanced_model,
        input_size=(batch_size, *input_size),  # Including batch size in the input size
        device=device,
    )
)
print("")

########################################################################################

N_epochs = 5
lr = 0.00016388181712790806  # was 1e-3
weight_decay = 5.6547937254492916e-5  # 4.6
optimizer = Adam(enhanced_model.parameters(), lr=lr, weight_decay=weight_decay)
schedule_params = {"factor": 1}
schedule = lr_scheduler.ConstantLR(optimizer, **schedule_params)
log_every = 1

step = 0
for epoch in range(N_epochs):
    running_loss = 0.0
    for i, xbatch in enumerate(halfpi_train_loader):  # Correct unpacking
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

    if epoch % log_every == 0 or epoch == N_epochs - 1:
        val_loss = torch.tensor(
            [
                enhanced_model.eval().corrloss(databatch.to(device=device))
                for databatch in halfpi_val_loader
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

        print(
            "{:<180}".format(
                "\r"
                + "[{:<60}] ".format(
                    "=" * ((np.floor((epoch + 1) / N_epochs * 60)).astype(int) - 1)
                    + ">"
                    if epoch + 1 < N_epochs
                    else "=" * 60
                )
                + "{:<40}".format(
                    "Epoch {}/{}: Correlation Loss(Train) = {:.3f}, Correlation Loss(Val) = {:.3f}, Accuracy(All Bright & Dark) = {:.3f}".format(
                        epoch + 1, N_epochs, running_loss, val_loss, sia
                    )
                )
            ),
            end="",
        )
        sys.stdout.flush()
