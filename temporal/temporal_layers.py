import torch
from torch.nn import Parameter
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.inits import glorot


class HeteroGCLSTM(torch.nn.Module):
    r"""An implementation similar to the Integrated Graph Convolutional Long Short Term
        Memory Cell for heterogeneous Graphs.
        Args:
            in_channels_dict (dict of keys=str and values=int): Dimension of each node's input features.
            out_channels (int): Number of output features.
            metadata (tuple): Metadata on node types and edge types in the graphs. Can be generated via PyG method
                :obj:`snapshot.metadata()` where snapshot is a single HeteroData object.
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
    """

    def __init__(
            self,
            in_channels_dict: dict,
            out_channels: int,
            metadata: tuple,
            bias: bool = True
    ):
        super(HeteroGCLSTM, self).__init__()

        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.metadata = metadata
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_i = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                      out_channels=self.out_channels,
                                                      bias=self.bias) for edge_type in self.metadata[1]})

        self.W_i = {node_type: Parameter(torch.Tensor(in_channels, self.out_channels)).to('cuda') 
                    for node_type, in_channels in self.in_channels_dict.items()}
        self.b_i = {node_type: Parameter(torch.Tensor(1, self.out_channels)).to('cuda') 
                    for node_type in self.in_channels_dict}

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_f = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                      out_channels=self.out_channels,
                                                      bias=self.bias) for edge_type in self.metadata[1]})

        self.W_f = {node_type: Parameter(torch.Tensor(in_channels, self.out_channels)).to('cuda') 
                    for node_type, in_channels in self.in_channels_dict.items()}
        self.b_f = {node_type: Parameter(torch.Tensor(1, self.out_channels)).to('cuda') 
                    for node_type in self.in_channels_dict}

    def _create_cell_state_parameters_and_layers(self):
        self.conv_c = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                      out_channels=self.out_channels,
                                                      bias=self.bias) for edge_type in self.metadata[1]})

        self.W_c = {node_type: Parameter(torch.Tensor(in_channels, self.out_channels)).to('cuda') 
                    for node_type, in_channels in self.in_channels_dict.items()}
        self.b_c = {node_type: Parameter(torch.Tensor(1, self.out_channels)).to('cuda') 
                    for node_type in self.in_channels_dict}

    def _create_output_gate_parameters_and_layers(self):
        self.conv_o = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                      out_channels=self.out_channels,
                                                      bias=self.bias) for edge_type in self.metadata[1]})

        self.W_o = {node_type: Parameter(torch.Tensor(in_channels, self.out_channels)).to('cuda') 
                    for node_type, in_channels in self.in_channels_dict.items()}
        self.b_o = {node_type: Parameter(torch.Tensor(1, self.out_channels)).to('cuda') 
                    for node_type in self.in_channels_dict}

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        for key in self.W_i:
            glorot(self.W_i[key])
        for key in self.W_f:
            glorot(self.W_f[key])
        for key in self.W_c:
            glorot(self.W_c[key])
        for key in self.W_o:
            glorot(self.W_o[key])
        for key in self.b_i:
            glorot(self.b_i[key])
        for key in self.b_f:
            glorot(self.b_f[key])
        for key in self.b_c:
            glorot(self.b_c[key])
        for key in self.b_o:
            glorot(self.b_o[key])

    def _set_hidden_state(self, x_dict, h_dict):
        if h_dict is None:
            h_dict = {node_type: torch.zeros(X.shape[0], self.out_channels).to('cuda') for node_type, X in x_dict.items()}
        return h_dict

    def _set_cell_state(self, x_dict, c_dict):
        if c_dict is None:
            c_dict = {node_type: torch.zeros(X.shape[0], self.out_channels).to('cuda') for node_type, X in x_dict.items()}
        return c_dict

    def _calculate_input_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        i_dict = {node_type: torch.matmul(X, self.W_i[node_type]) for node_type, X in x_dict.items()}
        i_dict = {node_type: I + self.conv_i(h_dict, edge_index_dict)[node_type] for node_type, I in i_dict.items()}
        i_dict = {node_type: I + self.b_i[node_type] for node_type, I in i_dict.items()}
        i_dict = {node_type: torch.sigmoid(I) for node_type, I in i_dict.items()}
        return i_dict

    def _calculate_forget_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        f_dict = {node_type: torch.matmul(X, self.W_f[node_type]) for node_type, X in x_dict.items()}
        f_dict = {node_type: F + self.conv_f(h_dict, edge_index_dict)[node_type] for node_type, F in f_dict.items()}
        f_dict = {node_type: F + self.b_f[node_type] for node_type, F in f_dict.items()}
        f_dict = {node_type: torch.sigmoid(F) for node_type, F in f_dict.items()}
        return f_dict

    def _calculate_cell_state(self, x_dict, edge_index_dict, h_dict, c_dict, i_dict, f_dict):
        t_dict = {node_type: torch.matmul(X, self.W_c[node_type]) for node_type, X in x_dict.items()}
        t_dict = {node_type: T + self.conv_c(h_dict, edge_index_dict)[node_type] for node_type, T in t_dict.items()}
        t_dict = {node_type: T + self.b_c[node_type] for node_type, T in t_dict.items()}
        t_dict = {node_type: torch.tanh(T) for node_type, T in t_dict.items()}
        c_dict = {node_type: f_dict[node_type] * C + i_dict[node_type] * t_dict[node_type] for node_type, C in c_dict.items()}
        return c_dict

    def _calculate_output_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        o_dict = {node_type: torch.matmul(X, self.W_o[node_type]) for node_type, X in x_dict.items()}
        o_dict = {node_type: O + self.conv_o(h_dict, edge_index_dict)[node_type] for node_type, O in o_dict.items()}
        o_dict = {node_type: O + self.b_o[node_type] for node_type, O in o_dict.items()}
        o_dict = {node_type: torch.sigmoid(O) for node_type, O in o_dict.items()}
        return o_dict

    def _calculate_hidden_state(self, o_dict, c_dict):
        h_dict = {node_type: o_dict[node_type] * torch.tanh(C) for node_type, C in c_dict.items()}
        return h_dict

    def forward(
        self,
        x_dict,
        edge_index_dict,
        h_dict=None,
        c_dict=None,
    ):
        """
        Making a forward pass. If the hidden state and cell state
        matrix dicts are not present when the forward pass is called these are
        initialized with zeros.
        Arg types:
            * **x_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensors)* - Node features dicts. Can
                be obtained via PyG method :obj:`snapshot.x_dict` where snapshot is a single HeteroData object.
            * **edge_index_dict** *(Dictionary where keys=Tuples and values=PyTorch Long Tensors)* - Graph edge type
                and index dicts. Can be obtained via PyG method :obj:`snapshot.edge_index_dict`.
            * **h_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor, optional)* - Node type and
                hidden state matrix dict for all nodes.
            * **c_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor, optional)* - Node type and
                cell state matrix dict for all nodes.
        Return types:
            * **h_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor)* - Node type and
                hidden state matrix dict for all nodes.
            * **c_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor)* - Node type and
                cell state matrix dict for all nodes.
        """

        h_dict = self._set_hidden_state(x_dict, h_dict)
        c_dict = self._set_cell_state(x_dict, c_dict)
        i_dict = self._calculate_input_gate(x_dict, edge_index_dict, h_dict, c_dict)
        f_dict = self._calculate_forget_gate(x_dict, edge_index_dict, h_dict, c_dict)
        c_dict = self._calculate_cell_state(x_dict, edge_index_dict, h_dict, c_dict, i_dict, f_dict)
        o_dict = self._calculate_output_gate(x_dict, edge_index_dict, h_dict, c_dict)
        h_dict = self._calculate_hidden_state(o_dict, c_dict)
        return h_dict, c_dict


class HeteroGConvGRU(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        metadata: tuple,
        bias: bool = True
    ):
        super(HeteroGConvGRU, self).__init__()

        self.out_channels = out_channels
        self.metadata = metadata,
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

        self.conv_h_z = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

        self.conv_h_r = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

        self.conv_h_h = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                    out_channels=self.out_channels,
                                    bias=self.bias) for edge_type in self.metadata[1]})

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, x_dict, h_dict):
        if h_dict is None:
            h_dict = {node_type: torch.zeros(X.shape[0], self.out_channels).to('cuda') for node_type, X in x_dict.items()}
        return h_dict

    def _calculate_update_gate(self, x_dict, edge_index_dict, h_dict):
        z_dict = self.conv_x_z(x_dict, edge_index_dict)
        z_dict = {node_type: Z + self.conv_h_z(h_dict, edge_index_dict)[node_type] for node_type, Z in z_dict.items()}
        z_dict = {node_type: torch.sigmoid(Z) for node_type, Z in z_dict.items()}
        return z_dict

    def _calculate_reset_gate(self, x_dict, edge_index_dict, h_dict):
        r_dict = self.conv_x_r(x_dict, edge_index_dict)
        r_dict = {node_type: R + self.conv_h_r(h_dict, edge_index_dict)[node_type] for node_type, R in r_dict.items()}
        r_dict = {node_type: torch.sigmoid(R) for node_type, R in r_dict.items()}
        return r_dict

    def _calculate_candidate_state(self, x_dict, edge_index_dict, h_dict, r_dict):
        h_tilde_dict = self.conv_x_h(x_dict, edge_index_dict)
        h_tilde_dict = {node_type: h_tilde for node_type, h_tilde in h_tilde_dict.items()}
        h_r_dict = {node_type: H * r_dict[node_type] for node_type, H in h_dict.items()}
        h_tilde_dict = {node_type: h_tilde + self.conv_h_h(h_r_dict, edge_index_dict) for node_type, h_tilde in h_tilde_dict.items()}
        h_tilde_dict = {node_type: torch.tanh(h_tilde) for node_type, h_tilde in h_tilde_dict.items()}
        return h_tilde_dict

    def _calculate_hidden_state(self, z_dict, h_dict, h_tilde_dict):
        h_dict = {node_type: z_dict[node_type] * H + (1 - z_dict[node_type]) * h_tilde_dict[node_type] for node_type, H in h_dict.items()}
        return h_dict

    def forward(
        self,
        x_dict,
        edge_index_dict,
        h_dict=None,
    ):
        h_dict = self._set_hidden_state(x_dict, h_dict)
        z_dict = self._calculate_update_gate(x_dict, edge_index_dict, h_dict)
        r_dict = self._calculate_reset_gate(x_dict, edge_index_dict, h_dict)
        h_tilde_dict = self._calculate_candidate_state(x_dict, edge_index_dict, h_dict, r_dict)
        h_dict = self._calculate_hidden_state(z_dict, h_dict, h_tilde_dict)
        return h_dict
