import torch
from torch_geometric.nn import GCNConv, GATConv, pool
import torch.nn as nn


class GCNEncoder(nn.Module):
  
  def __init__(self, in_channels, hidden_size, out_channels, dropout):
    super(GCNEncoder, self).__init__()
    self.conv1 = GCNConv(in_channels, hidden_size)
    self.conv2 = GCNConv(hidden_size, out_channels)
    self.dropout = nn.Dropout(dropout)

  # Our model will take the feature matrix X and the edge list
  # representation of the graph as inputs.
  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index).relu()
    x = self.dropout(x)
    return self.conv2(x, edge_index)

class GATEncoder(torch.nn.Module):
  
  def __init__(self, in_channels, hidden_sizes, out_channels, heads):
    super(GATEncoder, self).__init__()
    self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_sizes[0], heads=heads[0])
    self.convs = nn.ModuleList()
    if len(hidden_sizes)==1:
      self.conv_out = GATConv(in_channels=hidden_sizes[0]*heads[0], out_channels=out_channels, heads=heads[0])
    else:
      for i in range(1, len(hidden_sizes)):
        self.convs.append(GATConv(in_channels=hidden_sizes[i-1]*heads[i-1], out_channels=hidden_sizes[i], heads=heads[i]))
      self.conv_out = GATConv(in_channels=hidden_sizes[-1]*heads[-2], out_channels=out_channels, heads=heads[-1])
      
  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index).relu()
    if len(self.convs) == 0:
      return self.conv_out(x, edge_index)
    for conv in self.convs:
      x = conv(x, edge_index).relu()
    x = self.conv_out(x, edge_index)
    return x

class SingleLayerGATEncoder(torch.nn.Module):

  def __init__(self, in_channels, out_channels, heads):
    super(SingleLayerGATEncoder, self).__init__()
    self.conv = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads)
      
  def forward(self, x, edge_index):
    x = self.conv(x, edge_index)

    return x


class GATPooling(nn.Module):

  def __init__(self, in_channels, hidden_size, out_channels):
    super(GATPooling, self).__init__()
    self.encoder1 = GCNConv(in_channels=in_channels, out_channels=hidden_size)
    self.pool1 = pool.SAGPooling(in_channels=hidden_size, ratio=0.5)
    self.encoder2 = GCNConv(in_channels=hidden_size, out_channels=out_channels)
    self.pool2 = pool.SAGPooling(in_channels=out_channels, ratio=0.5)
  
  def forward(self, x, edge_index, batch):
    x = self.encoder1(x, edge_index).relu()
    x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
    x = self.encoder2(x, edge_index).relu()
    x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
    x = pool.global_mean_pool(x, batch)
    return x

# class MeanMaxPooling(nn.Module):

#   def __init__(self):
#     super(MeanMaxPooling, self).__init__()
  
#   def forward(self, x, edge_index, batch):
#     x1 = self.pool1(x, batch)
#     x2 = self.pool2(x, batch)
#     return torch.cat([x1, x2], dim=1)

class GCNClassifier(nn.Module):
  
  def __init__(self, in_channels, hidden_sizes, out_channels):
    super(GCNClassifier, self).__init__()
    self.conv1 = GCNConv(in_channels, hidden_sizes[0])
    self.convs = nn.ModuleList()
    for i in range(1, len(hidden_sizes)):
      self.convs.append(GCNConv(hidden_sizes[i-1], hidden_sizes[i]))
    self.conv_out = GCNConv(hidden_sizes[-1], out_channels)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index).relu()
    for conv in self.convs:
      x = conv(x, edge_index).relu()
    return self.conv_out(x, edge_index)
  
class Classifier(nn.Module):

  def __init__(self, in_channels, hidden_sizes, out_channels):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(in_channels, hidden_sizes[0])
    self.fcs = nn.ModuleList()
    for i in range(1, len(hidden_sizes)):
      self.fcs.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
    self.out = nn.Linear(hidden_sizes[-1], out_channels)
  
  def forward(self, x, edge_index):
    x = self.fc1(x).relu()
    if len(self.fcs) == 0:
      return self.out(x)
    for fc in self.fcs:
      x = fc(x).relu()
    return self.out(x)