import torch


def run(
    model,
    model_loss,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    n_epoch=200,
    batch_size=32,
    sampling=False,
    scheduler=None,
):
    train_errors = []
    test_errors = []
    test_mapes = []
    n_train = len(x_train)

    for epoch in range(n_epoch):
        model.train()
        # print("### epoch:", epoch)
        # Shuffle training data
        if sampling:
            weights = 0.25 + 0.75 * (1 + torch.cos(torch.pi * x_train.flatten())) / 2

            idx = torch.multinomial(
                weights,
                num_samples=n_train,
                replacement=True,
            )
        else:
            idx = torch.randperm(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = idx[start : start + batch_size]

            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = model_loss(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_idx.size(0)

        # Continue for (epoch)
        train_errors.append([epoch, epoch_loss / n_train])

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_val)
            test_error = model_loss(y_pred, y_val).item()
            test_mape = torch.abs((y_val - y_pred) / y_val).mean().item()
        test_errors.append([epoch, test_error])
        test_mapes.append([epoch, test_mape])

        if scheduler is not None:
            scheduler.step()

    return (train_errors, test_errors, test_mapes)
