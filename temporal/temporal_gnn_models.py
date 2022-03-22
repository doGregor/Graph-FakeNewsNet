from temporal.temporal_layers import *
import torch.nn.functional as F
import torch


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feature_dict, metadata, num_hidden_nodes, num_output_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = HeteroGCLSTM(node_feature_dict, num_hidden_nodes, metadata)
        self.linear = torch.nn.Linear(num_hidden_nodes, num_output_features)

    def forward(self, x_dict, edge_index_dict):
        h, c = self.recurrent(x_dict, edge_index_dict)
        h = F.relu(h['article'])
        h = self.linear(h)
        return h
