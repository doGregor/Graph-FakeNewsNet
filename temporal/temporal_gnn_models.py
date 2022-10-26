from temporal.temporal_layers import *
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool, HGTConv
from torch.nn import Transformer
import math


class HeteroLSTMGCN(torch.nn.Module):
    def __init__(self, node_feature_dict, metadata, num_hidden_nodes, num_output_features):
        super(HeteroLSTMGCN, self).__init__()
        self.recurrent = HeteroGCLSTM(node_feature_dict, num_hidden_nodes, metadata)
        self.linear = torch.nn.Linear(num_hidden_nodes*3, num_output_features)

    def forward(self, x_dict, edge_index_dict, batch_dict, h_dict, c_dict):
        h_0, c_0 = self.recurrent(x_dict, edge_index_dict, h_dict, c_dict)
        
        h = {key: val.relu() for key, val in h_0.items()}
        h = {key: global_mean_pool(val, batch_dict[key]) for key, val in h.items()}
        h = torch.cat([h['article'], h['tweet'], h['user']], dim=1)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.linear(h)
        
        return h, h_0, c_0


class HeteroGRUGCN(torch.nn.Module):
    def __init__(self, num_hidden_nodes, metadata, num_output_features):
        super(HeteroGRUGCN, self).__init__()
        self.recurrent = HeteroGConvGRU(num_hidden_nodes, metadata)
        self.linear = torch.nn.Linear(num_hidden_nodes * 3, num_output_features)

    def forward(self, x_dict, edge_index_dict, batch_dict, h_dict):
        h_0 = self.recurrent(x_dict, edge_index_dict, h_dict)
        
        h = {key: val.relu() for key, val in h_0.items()}
        h = {key: global_mean_pool(val, batch_dict[key]) for key, val in h.items()}
        h = torch.cat([h['article'], h['tweet'], h['user']], dim=1)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.linear(h)

        return h, h_0


class HeteroGATGRUModel(torch.nn.Module):
    def __init__(self, num_hidden_nodes, metadata, num_output_features, num_attention_heads=3):
        super(HeteroGATGRUModel, self).__init__()
        self.recurrent = HeteroGATGRU(num_hidden_nodes, metadata, num_attention_heads)
        self.linear = torch.nn.Linear(num_hidden_nodes * 3 * num_attention_heads, num_output_features)

    def forward(self, x_dict, edge_index_dict, batch_dict, h_dict):
        h_0 = self.recurrent(x_dict, edge_index_dict, h_dict)

        h = {key: val.relu() for key, val in h_0.items()}
        h = {key: global_mean_pool(val, batch_dict[key]) for key, val in h.items()}
        h = torch.cat([h['article'], h['tweet'], h['user']], dim=1)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.linear(h)

        return h, h_0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerGAT(torch.nn.Module):
    def __init__(self,
                 num_timesteps,
                 num_graph_features,
                 num_output_features,
                 metadata,
                 num_attention_heads=1,
                 num_layers=2,
                 num_hidden_gnn=64,
                 num_output_gnn=8,
                 num_transformer_heads=8,
                 num_encoders=3,
                 num_decoders=3):
        super(TransformerGAT, self).__init__()

        self.graph_convs = nn.ModuleList()
        for i in range(num_timesteps):
            convs = nn.ModuleList()
            convs.append(HGTConv(-1, num_hidden_gnn, metadata, num_attention_heads))
            convs.append(HGTConv(-1, num_output_gnn, metadata, num_attention_heads))
            self.graph_convs.append(convs)

        self.pos_encoder = PositionalEncoding(d_model=num_graph_features)
        self.Transformer = Transformer(d_model=num_graph_features,
                                       num_encoder_layers=num_encoders,
                                       num_decoder_layers=num_decoders,
                                       nhead=num_transformer_heads)

        self.linear = torch.nn.Linear(num_timesteps * num_graph_features, num_output_features)

    def forward(self, snapshot_batch):

        hidden_representations = []
        for idx, t_batch in enumerate(snapshot_batch):
            conv_layers = self.graph_convs[idx]
            batch_x_dict = t_batch.x_dict
            for conv in conv_layers:
                batch_x_dict = conv(batch_x_dict, t_batch.edge_index_dict)
                batch_x_dict = {key: x.relu() for key, x in batch_x_dict.items()}
            batch_x_dict = {key: global_mean_pool(x, t_batch.batch_dict[key]) for key, x in batch_x_dict.items()}
            x = torch.cat([batch_x_dict['article'], batch_x_dict['tweet'], batch_x_dict['user']], dim=1)
            hidden_representations.append(x)
            
        x_features = torch.stack((hidden_representations))

        x_features = self.pos_encoder(x_features)
        x_features = self.Transformer(x_features, x_features)
        x_features = x_features.relu()

        x_features = torch.moveaxis(x_features, 0, 1)
        x_features = torch.reshape(x_features, (x_features.shape[0], x_features.shape[1] * x_features.shape[2]))

        x_features = self.linear(x_features)

        return x_features
