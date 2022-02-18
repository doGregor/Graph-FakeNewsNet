import os
import sys
from tqdm import tqdm
from random import shuffle
from sklearn.model_selection import KFold
from operator import itemgetter
from sklearn.utils import compute_class_weight
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data_preprocessing.graph_structure import *
from machine_learning.gnn_models import *
from machine_learning.gnn_training import *
from data_preprocessing.load_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    'dataset': 'gossipcop',
    'setting': 'all-data',
    'batch_size': 16,
    'hidden_dim': 64,
    'learning_rate': 0.00008,
    'weight_decay': 0.00005,
    'epochs': 20
}


def load_graph_by_path(file_name, dataset='gossipcop', setting='all_data'):
    if setting in ['tweets', 'tweets_users', 'tweets_users_retweets', 'tweets_users_timeline', 'all_data'] and \
            dataset in ['gossipcop', 'politifact']:
        path = "../data/graphs_" + dataset + "/" + setting + "/" + file_name + ".pickle"
        with open(path, 'rb') as handle:
            return pickle.load(handle)


ids_true, ids_fake = get_news_ids('gossipcop')

relevant_real = []
relevant_fake = []

path_to_graphs = '../data/graphs_' + CONFIG['dataset'] + '/' + CONFIG['setting'] + '/'
for graph_id in os.listdir(path_to_graphs):
    if graph_id.endswith('pickle'):
        graph_id = graph_id.split('.')[0]
        graph = load_graph_by_path(graph_id, CONFIG['dataset'], CONFIG['setting'])['graph']
        if graph['user'].x.size()[0] >= 5 and graph['tweet'].x.size()[0] >= 5 and \
                (0 == graph['tweet', 'cites', 'article'].edge_index.size()[1] or
                 graph['tweet', 'cites', 'article'].edge_index.size()[1] >= 5) and \
                (0 == graph['user', 'posts', 'tweet'].edge_index.size()[1] or
                 graph['user', 'posts', 'tweet'].edge_index.size()[1] >=5) and \
                (0 == graph['tweet', 'retweets', 'tweet'].edge_index.size()[1] or
                 graph['tweet', 'retweets', 'tweet'].edge_index.size()[1] >=5):
            if graph_id in list(ids_true):
                relevant_real.append(T.NormalizeFeatures()(graph))
            elif graph_id in list(ids_fake):
                relevant_fake.append(T.NormalizeFeatures()(graph))

print("Number or real news graphs:", len(relevant_real))
print("Number of fake news graphs:", len(relevant_fake))

all_graphs = relevant_real + relevant_fake
shuffle(all_graphs)

kf = KFold(n_splits=5)
kf.get_n_splits(all_graphs)

train_splits = []
test_splits = []

for train_index, test_index in kf.split(all_graphs):
    print(30*"*")
    X_train, X_test = itemgetter(*train_index)(all_graphs), itemgetter(*test_index)(all_graphs)
    print("Num train:", len(X_train), "Num test:", len(X_test))
    train_splits.append(X_train)
    test_splits.append(X_test)


# ###################################### #
# ################ SAGE ################ #
# ###################################### #


acc_all = []
p_all = []
r_all = []
f1_all = []

for idx, val in enumerate(train_splits):
    X_train = val
    X_test = test_splits[idx]

    y_tensors = []
    for graph in val:
        y_tensors.append(graph['article'].y)

    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.asarray([0, 1]),
                                                      y=torch.cat(y_tensors).cpu().detach().numpy()),
                                 dtype=torch.float32)
    class_weights.to(device)

    train_loader = DataLoader(X_train, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(X_test, batch_size=CONFIG['batch_size'], shuffle=False)

    model = GraphSAGE(hidden_channels=CONFIG['hidden_dim'], out_channels=2, metadata=relevant_fake[0].metadata())

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)
    acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                                  loss_fct=criterion, optimizer=optimizer, num_epochs=CONFIG['epochs'],
                                                  verbose=0)
    acc_all.append(acc)
    p_all.append(precision)
    r_all.append(recall)
    f1_all.append(f1)

print("ACC SAGE", acc_all, sum(acc_all) / len(acc_all))
print("P SAGE", p_all, sum(p_all) / len(p_all))
print("R SAGE", r_all, sum(r_all) / len(r_all))
print("F1 SAGE", f1_all, sum(f1_all) / len(f1_all))


# ###################################### #
# ################ GAT ################# #
# ###################################### #


acc_all = []
p_all = []
r_all = []
f1_all = []

for idx, val in enumerate(train_splits):
    X_train = val
    X_test = test_splits[idx]

    y_tensors = []
    for graph in val:
        y_tensors.append(graph['article'].y)

    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.asarray([0, 1]),
                                                      y=torch.cat(y_tensors).cpu().detach().numpy()),
                                 dtype=torch.float32)
    class_weights.to(device)

    train_loader = DataLoader(X_train, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(X_test, batch_size=CONFIG['batch_size'], shuffle=False)

    model = GAT(hidden_channels=CONFIG['hidden_dim'], out_channels=2, metadata=relevant_fake[0].metadata())

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)
    acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                                  loss_fct=criterion, optimizer=optimizer, num_epochs=CONFIG['epochs'],
                                                  verbose=0)
    acc_all.append(acc)
    p_all.append(precision)
    r_all.append(recall)
    f1_all.append(f1)

print("ACC GAT", acc_all, sum(acc_all) / len(acc_all))
print("P GAT", p_all, sum(p_all) / len(p_all))
print("R GAT", r_all, sum(r_all) / len(r_all))
print("F1 GAT", f1_all, sum(f1_all) / len(f1_all))


# ###################################### #
# ################ HGT ################# #
# ###################################### #


acc_all = []
p_all = []
r_all = []
f1_all = []

for idx, val in enumerate(train_splits):
    X_train = val
    X_test = test_splits[idx]

    y_tensors = []
    for graph in val:
        y_tensors.append(graph['article'].y)

    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.asarray([0, 1]),
                                                      y=torch.cat(y_tensors).cpu().detach().numpy()),
                                 dtype=torch.float32)
    class_weights.to(device)

    train_loader = DataLoader(X_train, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(X_test, batch_size=CONFIG['batch_size'], shuffle=False)

    model = HGT(hidden_channels=CONFIG['hidden_dim'], out_channels=2, metadata=relevant_fake[0].metadata())

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(device)
    acc, precision, recall, f1 = train_eval_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                                  loss_fct=criterion, optimizer=optimizer, num_epochs=CONFIG['epochs'],
                                                  verbose=0)
    acc_all.append(acc)
    p_all.append(precision)
    r_all.append(recall)
    f1_all.append(f1)

print("ACC", acc_all, sum(acc_all) / len(acc_all))
print("P", p_all, sum(p_all) / len(p_all))
print("R", r_all, sum(r_all) / len(r_all))
print("F1", f1_all, sum(f1_all) / len(f1_all))