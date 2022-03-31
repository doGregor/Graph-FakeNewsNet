from temporal.temporal_layers import *
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feature_dict, metadata, num_hidden_nodes, num_output_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = HeteroGCLSTM(node_feature_dict, num_hidden_nodes, metadata)
        self.linear = torch.nn.Linear(num_hidden_nodes*3, num_output_features)

    def forward(self, x_dict, edge_index_dict, batch_dict, h_dict, c_dict):
        h_0, c_0 = self.recurrent(x_dict, edge_index_dict, h_dict, c_dict)
        
        h = {key: val.relu() for key, val in h_0.items()}
        h = {key: global_mean_pool(val, batch_dict[key]) for key, val in h.items()}
        h = torch.cat([h['article'], h['tweet'], h['user']], dim=1)
        h = self.linear(h)
        
        return h, h_0, c_0
