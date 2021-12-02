import torch


def train_model(model, train_loader, loss_fct, optimizer):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  # Perform a single forward pass.
        loss = loss_fct(out, data['article'].y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def eval_model(model, test_loader):
    model.eval()
    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data['article'].y).sum())  # Check against ground-truth labels.
    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.


def train_eval_model(model, train_loader, test_loader, loss_fct, optimizer, num_epochs=1):
    for epoch in range(1, num_epochs+1):
        train_model(model=model, train_loader=train_loader, loss_fct=loss_fct, optimizer=optimizer)
        train_acc = eval_model(model, train_loader)
        test_acc = eval_model(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
