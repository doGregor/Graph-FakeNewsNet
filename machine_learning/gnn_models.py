from torch_geometric.loader import DataLoader
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_layers=2):
        super(GCN, self).__init__()
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
