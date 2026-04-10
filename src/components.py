import torch
import torch.nn as nn

EEG_GRAPH_NEIGHBORS = {
    'Fp1': ['Fpz', 'F3', 'F7'],
    'Fp2': ['Fpz', 'F4', 'F8'],
    'Fpz': ['Fp1', 'Fp2', 'Fz'],

    'F7':  ['Fp1', 'F3', 'T3'],
    'F3':  ['Fp1', 'F7', 'Fz', 'C3'],
    'Fz':  ['Fpz', 'F3', 'F4', 'Cz'],
    'F4':  ['Fp2', 'F8', 'Fz', 'C4'],
    'F8':  ['Fp2', 'F4', 'T4'],

    'T3':  ['F7', 'C3', 'T5'],
    'C3':  ['F3', 'Cz', 'P3', 'T3'],
    'Cz':  ['Fz', 'C3', 'C4', 'Pz'],
    'C4':  ['F4', 'Cz', 'P4', 'T4'],
    'T4':  ['F8', 'C4', 'T6'],

    'T5':  ['T3', 'P3', 'O1'],
    'P3':  ['C3', 'Pz', 'O1', 'T5'],
    'Pz':  ['Cz', 'P3', 'P4', 'O1', 'O2'],
    'P4':  ['C4', 'Pz', 'O2', 'T6'],
    'T6':  ['T4', 'P4', 'O2'],

    'O1':  ['P3', 'Pz', 'T5'],
    'O2':  ['P4', 'Pz', 'T6'],
}

EEG_CHANNELS_ORDER = ['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EEGCONVEncoder(nn.Module):
    def __init__(self, in_channels= 20, out_channels=64, kernel_size=(2,1), dropout=0.1, pool = 'max'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size, padding=(kernel_size[0]//2, 0))
        self.bn1 = nn.BatchNorm2d(out_channels//2)

        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size, padding=(kernel_size[0]//2, 0))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size[0]//2, 0))
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.pool = pool

    def forward(self, x, **kwargs):
        # x: [B, F, 20, T]
        x = self.conv1(x)   # [B, F_out, N, T]
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)   # [B, F_out, N, T]
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)   # [B, F_out, N, T]
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        if self.pool == 'max':
            x = F.adaptive_max_pool2d(x, (1, None))  # [B, F_out, 1, T]
        elif self.pool == 'mean':
            x = F.adaptive_avg_pool2d(x, (1, None))  # [B, F_out, 1, T]

        x = x.squeeze(2).transpose (1, 2)  # [B, T, F_out]
        return x


class EEGGATEncoder(nn.Module):
    """
    Input:  x [B, F_in, N, T]
    Output: y [B, F_out, 1, T]
    where N = number of EEG electrodes
    """
    def __init__(self, in_features, hidden_features=32, out_features=64, dropout=0.1, pool='mean'):
        super().__init__()
        self.pool = pool

        self.encoder = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=(1,1)),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gat1 = GATConv(hidden_features, hidden_features // 2, heads=2, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_features)

        self.gat2 = GATConv(hidden_features, hidden_features // 2, heads=2, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_features)

        self.gat3 = GATConv(hidden_features, out_features, heads=1, concat=True)
        self.bn3 = nn.BatchNorm1d(out_features)

        self.dropout = nn.Dropout(dropout)

    def _batch_edge_index(self, edge_index, batch_size, num_nodes, device):
        E = edge_index.size(1)
        offsets = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_nodes
        edge_index = edge_index.view(1, 2, E) + offsets
        edge_index = edge_index.permute(1, 0, 2).reshape(2, batch_size * E)
        return edge_index

    def _gnn_block(self, x, edge_index, conv, bn):
        # x: [G, N, F]
        G, N, Fdim = x.shape
        x = x.reshape(G * N, Fdim)
        edge_index = self._batch_edge_index(edge_index, G, N, x.device)

        x = conv(x, edge_index)
        x = bn(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = x.view(G, N, -1)
        return x

    def forward(self, x, edge_index):
        # x: [B, F_in, N, T], usually N = 19 for EEG channels
        x = self.encoder(x)  # [B, F_hidden, N, T]

        B, F_in, N, T = x.shape
        # turn each time slice into one graph
        x = x.permute(0, 3, 2, 1).contiguous()   # [B, T, N, F_in]
        x = x.view(B * T, N, F_in)               # [B*T, N, F_in]

        x = self._gnn_block(x, edge_index, self.gat1, self.bn1)
        x = self._gnn_block(x, edge_index, self.gat2, self.bn2)
        x = self._gnn_block(x, edge_index, self.gat3, self.bn3)


        # pool across electrodes
        if self.pool == 'mean':
            x = x.mean(dim=1)                    # [B*T, F_out]
        elif self.pool == 'max':
            x = x.max(dim=1).values              # [B*T, F_out]
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        # reshape back to time sequence
        x = x.view(B, T, -1)                     # [B, T, F_out]

        return x
    
def build_eeg_edge_index(channel_order, neighbor_dict, add_self_loops=True):
    ch_to_idx = {ch: i for i, ch in enumerate(channel_order)}
    edges = set()

    for ch, nbrs in neighbor_dict.items():
        i = ch_to_idx[ch]
        for nbr in nbrs:
            j = ch_to_idx[nbr]
            edges.add((i, j))
            edges.add((j, i))   # undirected graph

    if add_self_loops:
        for i in range(len(channel_order)):
            edges.add((i, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index


def testEEGGAT(EEG_channels_ORDER, input_feature_dim):
    edge_index = build_eeg_edge_index(EEG_channels_ORDER, EEG_GRAPH_NEIGHBORS)

    model = EEGGATEncoder(
        in_features=input_feature_dim,   # not number of electrodes
        hidden_features=32,
        out_features=64,
        dropout=0.1
    )
    num_paras = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, input_feature_dim, len(EEG_channels_ORDER), 10)   # [B, F_in, N, T]
    y = model(x, edge_index.to(x.device))
    print(f"output shape: {y.shape}")   # [B, T, F_out]
    print(f"number of parameters: {num_paras}")

def testEEGCONV(input_feature_dim):
    model = EEGCONVEncoder(
        in_channels=input_feature_dim,
        out_channels=64,
        kernel_size=(2,1),
        dropout=0.1,
        pool='max'
    )
    num_paras = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, input_feature_dim, 19, 10)   # [B, F_in, N, T]
    y = model(x)
    print(f"output shape: {y.shape}")   # [B, T, F_out]
    print(f"number of parameters: {num_paras}")


if __name__ == "__main__":
    EEG_channels_ORDER = [
    'C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2',
    'Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']
    testEEGGAT(EEG_channels_ORDER, input_feature_dim=16)
    testEEGCONV(input_feature_dim=16)