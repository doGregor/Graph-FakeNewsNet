import numpy as np
import torch
from random import shuffle
from torch_geometric.data import Batch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_temporal_model(model, train_batches, loss_fct, optimizer, shuffle_batches=True):
    model.train()
    y_true = []
    y_pred = []
    if shuffle_batches:
        shuffle(train_batches)
    for batch in train_batches:
        cost = 0
        h_dict = None
        for idx_snap in range(0, len(batch[0])):
            graph_list = []
            for graph_sequence in batch:
                graph_list.append(graph_sequence[idx_snap])
            snapshot_batch = Batch.from_data_list(graph_list)
            snapshot_batch.to(DEVICE)
            y_hat, h_dict = model(snapshot_batch.x_dict, snapshot_batch.edge_index_dict,
                                  snapshot_batch.batch_dict, h_dict)
            cost += loss_fct(y_hat, torch.flatten(snapshot_batch['article'].y))
            # cost = loss_fct(y_hat, torch.flatten(snapshot_batch['article'].y))
            if idx_snap == len(batch[0]) - 1:
                y_pred.append(y_hat.argmax(dim=1).cpu().detach().numpy())
                y_true.append(snapshot_batch['article'].y.cpu().detach().numpy())
        cost /= len(batch[0])
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    return y_true, y_pred


def evaluate_temporal_model(model, test_batches):
    model.eval()
    true_y = []
    pred_y = []
    for batch in test_batches:
        h_dict = None
        for idx_snap in range(0, len(batch[0])):
            graph_list = []
            for graph_sequence in batch:
                graph_list.append(graph_sequence[idx_snap])
            snapshot_batch = Batch.from_data_list(graph_list)
            snapshot_batch.to(DEVICE)
            y_hat, h_dict = model(snapshot_batch.x_dict, snapshot_batch.edge_index_dict,
                                  snapshot_batch.batch_dict, h_dict)
        pred = y_hat.argmax(dim=1)
        pred_y.append(pred.cpu().detach().numpy())
        true_y.append(snapshot_batch['article'].y.cpu().detach().numpy())
    return true_y, pred_y


def train_eval_temporal_model(model, train_batches, test_batches, loss_fct, optimizer, num_epochs=20, verbose=1):
    for epoch in range(num_epochs):
        model.to(DEVICE)
        true_train, pred_train = train_temporal_model(model=model, train_batches=train_batches,
                                                      loss_fct=loss_fct, optimizer=optimizer)
        true_test, pred_test = evaluate_temporal_model(model=model, test_batches=test_batches)

        true_train = np.concatenate(true_train)
        pred_train = np.concatenate(pred_train)
        true_test = np.concatenate(true_test)
        pred_test = np.concatenate(pred_test)

        if epoch + 1 == num_epochs:
            print(classification_report(true_test, pred_test, digits=5))

            test_acc = accuracy_score(true_test, pred_test)
            test_p = precision_score(true_test, pred_test, average='macro')
            test_r = recall_score(true_test, pred_test, average='macro')
            test_f1 = f1_score(true_test, pred_test, average='macro')

            return test_acc, test_p, test_r, test_f1
        else:
            if verbose == 1:
                train_acc = accuracy_score(true_train, pred_train)
                train_f1 = f1_score(true_train, pred_train, average='macro')

                test_acc = accuracy_score(true_test, pred_test)
                test_f1 = f1_score(true_test, pred_test, average='macro')

                print_epoch = epoch + 1
                print(f'Epoch: {print_epoch:03d},',
                      f'Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f},',
                      f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
