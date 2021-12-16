from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, GATConv, SAGEConv, RGCNConv, HGTConv, Linear
from torch_geometric.nn import global_mean_pool


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers=2):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #conv = HeteroConv({
            #    ('user', 'posts', 'tweet'): SAGEConv((-1, -1), hidden_channels),
            #    ('tweet', 'cites', 'article'): SAGEConv((-1, -1), hidden_channels),
            #    ('article', 'rev_cites', 'tweet'): SAGEConv((-1, -1), hidden_channels),
            #    ('tweet', 'rev_posts', 'user'): SAGEConv((-1, -1), hidden_channels)
            #}, aggr='sum')
            conv = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in metadata[1]})
            self.convs.append(conv)

        self.lin = torch.nn.Linear(hidden_channels*3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
        x = torch.cat([x_dict['article'], x_dict['tweet'], x_dict['user']], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers=2, num_attention_heads=3):
        super(GAT, self).__init__()
        torch.manual_seed(12345)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({edge_type: GATConv((-1, -1), hidden_channels, heads=num_attention_heads,
                                                  add_self_loops=False) for edge_type in metadata[1]})
            self.convs.append(conv)

        # self.lin1 = torch.nn.Linear(hidden_channels*3*num_attention_heads, hidden_channels*3)
        # self.lin2 = torch.nn.Linear(hidden_channels*3, out_channels)
        self.lin = torch.nn.Linear(hidden_channels*3*num_attention_heads, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
        x = torch.cat([x_dict['article'], x_dict['tweet'], x_dict['user']], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.lin1(x)
        # x = self.lin2(x)
        x = self.lin(x)

        return x


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers=2, num_attention_heads=1):
        super(HGT, self).__init__()
        torch.manual_seed(12345)

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_attention_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels*3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x_dict = {key: global_mean_pool(x, batch_dict[key]) for key, x in x_dict.items()}
        x = torch.cat([x_dict['article'], x_dict['tweet'], x_dict['user']], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x
