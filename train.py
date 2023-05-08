import torch
import config


def train(model, data_loader, optimizer, criterion):
    model.train()

    epoch_loss = 0

    for src, trg in data_loader:

        optimizer.zero_grad()

        output = model(src, trg, config.teacher_forcing_ratio)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)

        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def evaluate(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for src, trg in dataloader:
            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
