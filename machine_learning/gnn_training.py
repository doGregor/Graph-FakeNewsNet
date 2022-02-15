import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, loss_fct, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data.to(DEVICE)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  # Perform a single forward pass.
        loss = loss_fct(out, data['article'].y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def eval_model(model, test_loader, print_classification_report=False):
    model.eval()
    correct = 0
    true_y = []
    pred_y = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(DEVICE)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_y.append(pred.cpu().detach().numpy())
        correct += int((pred == data['article'].y).sum())  # Check against ground-truth labels.
        true_y.append(data['article'].y.cpu().detach().numpy())
    if print_classification_report:
        print(classification_report(np.concatenate(true_y), np.concatenate(pred_y), digits=5))
    return accuracy_score(np.concatenate(true_y), np.concatenate(pred_y)), precision_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), recall_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), f1_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro')


def train_eval_model(model, train_loader, test_loader, loss_fct, optimizer, num_epochs=1, verbose=1):
    model.to(DEVICE)
    for epoch in range(1, num_epochs+1):
        train_model(model=model, train_loader=train_loader, loss_fct=loss_fct, optimizer=optimizer)
        train_acc, train_p, train_r, train_f1 = eval_model(model, train_loader)
        if epoch == num_epochs:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader, print_classification_report=True)
            return test_acc, test_p, test_r, test_f1
        else:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader)
            if verbose == 1:
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
