from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GATConv, global_mean_pool
import warnings
from torch_geometric.nn import global_mean_pool

# Ignore specific warnings
warnings.filterwarnings("ignore")


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, input_size):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(input_size, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        # self.conv4 = GCNConv(input_size, 64)
        self.lin = Linear(hidden_channels, 3)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x