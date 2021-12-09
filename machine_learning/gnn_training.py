import numpy as np
import torch
from sklearn.metrics import classification_report


def train_model(model, train_loader, loss_fct, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  # Perform a single forward pass.
        loss = loss_fct(out, data['article'].y)  # Compute the loss.
        #print(10*'*', batch_idx, 10*'*')
        #print(data['article'].y)
        #print(out)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def eval_model(model, test_loader, print_classification_report=False):
    model.eval()
    correct = 0
    true_y = []
    pred_y = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_y.append(pred.cpu().detach().numpy())
        correct += int((pred == data['article'].y).sum())  # Check against ground-truth labels.
        true_y.append(data['article'].y.cpu().detach().numpy())
    if print_classification_report:
        print(classification_report(np.concatenate(true_y), np.concatenate(pred_y), digits=5))
    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.


def train_eval_model(model, train_loader, test_loader, loss_fct, optimizer, num_epochs=1):
    for epoch in range(1, num_epochs+1):
        train_model(model=model, train_loader=train_loader, loss_fct=loss_fct, optimizer=optimizer)
        train_acc = eval_model(model, train_loader)
        if epoch == num_epochs:
            test_acc = eval_model(model, test_loader, print_classification_report=True)
        else:
            test_acc = eval_model(model, test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
