import torch
from torch_geometric.nn import GCNConv, GATConv, pool
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

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
  
  def forward(self, x):
    x = self.fc1(x).relu()
    if len(self.fcs) == 0:
      return self.out(x)
    for fc in self.fcs:
      x = fc(x).relu()
    return self.out(x)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    # Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Classification(nn.Module):

  def __init__(self, encoder_parameter, pooling_parameter, classifier_parameter):
    super(Classification, self).__init__()
    self.encoder = GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3])
    self.pooling = GATPooling(pooling_parameter[0], pooling_parameter[1], pooling_parameter[2])
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
  
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.encoder(x, edge_index)
    x = self.pooling(x, edge_index, batch)
    x = self.classifier(x)
    return x

class ClassificationSimplifiedPooling(nn.Module):

  def __init__(self, encoder_parameter, classifier_parameter):
    super(ClassificationSimplifiedPooling, self).__init__()
    self.encoder = GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3])
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])

  def forward(self, x, edge_index, batch):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.encoder(x, edge_index)
    
    x1 = pool.global_mean_pool(x, batch)
    x2 = pool.global_max_pool(x, batch)
    
    x = torch.cat([x1, x2], dim=1)
    x = self.classifier(x)
    return x

class ClassificationSimplifiedPoolingExplain(nn.Module):

  def __init__(self, encoder_parameter, classifier_parameter):
    super(ClassificationSimplifiedPoolingExplain, self).__init__()
    self.encoder = GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3])
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])

  def forward(self, x, edge_index, batch):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.encoder(x, edge_index)
    
    x1 = pool.global_mean_pool(x, batch)
    x2 = pool.global_max_pool(x, batch)
    
    x = torch.cat([x1, x2], dim=1)
    x = self.classifier(x)
    return F.softmax(x)

class DANN(nn.Module):

  def __init__(self, encoder_parameter, classifier_parameter, lambda_=1.0):
    super(DANN, self).__init__()
    self.encoder = GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3])
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
    # self.discriminator = combined_models.Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
    # self.gr1 = combined_models.GradientReversalLayer(lambda_)

  def forward(self, x, edge_index, batch):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.encoder(x, edge_index)
    
    x1 = pool.global_mean_pool(x, batch)
    x2 = pool.global_max_pool(x, batch)
    
    x = torch.cat([x1, x2], dim=1)
    x = self.classifier(x)
    # batch_output = self.discriminator(self.gr1(x))
    # return x, batch_output
    return x

class GraphDANN(nn.Module):

  def __init__(self, encoder_parameter, classifier_parameter):
    super(GraphDANN, self).__init__()
    self.encoder = GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3])
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
    self.discriminator = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
    # self.gr1 = combined_models.GradientReversalLayer(lambda_)
  
  def forward(self, x, edge_index, batch, lambd=2.0):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.encoder(x, edge_index)
    
    x1 = pool.global_mean_pool(x, batch)
    x2 = pool.global_max_pool(x, batch)
    
    x = torch.cat([x1, x2], dim=1)
    label_pred = self.classifier(x)
    reverse_x = grad_reverse(x, lambd)
    batch_pred = self.discriminator(reverse_x)
    # batch_output = self.discriminator(self.gr1(x))
    # return x, batch_output
    return label_pred, batch_pred

class MultiDANN(nn.Module):

  def __init__(self, encoder_parameters_l, classifier_parameter, n_experiments, dropout=0.3):
    super(MultiDANN, self).__init__()
    self.encoders = nn.ModuleList()
    for encoder_parameter in encoder_parameters_l:
      self.encoders.append(GATEncoder(encoder_parameter[0], encoder_parameter[1], encoder_parameter[2], encoder_parameter[3]))
    self.classifier = Classifier(classifier_parameter[0], classifier_parameter[1], classifier_parameter[2])
    self.discriminator = Classifier(classifier_parameter[0], classifier_parameter[1], n_experiments)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, edge_index, batch, set_idx, lambd=2.0, return_latent=False):
    
    x = self.encoders[set_idx](x, edge_index)
    # x = self.dropout(x)
    x1 = pool.global_mean_pool(x, batch)
    x2 = pool.global_max_pool(x, batch)

    x = torch.cat([x1, x2], dim=1)
    x = F.normalize(x, dim=1)
    label_pred = self.classifier(x)
    reverse_x = grad_reverse(x, lambd)
    batch_pred = self.discriminator(reverse_x)
    if return_latent:
      return label_pred, batch_pred, x
    else:
      return label_pred, batch_pred

class CenterLoss(nn.Module):
  def __init__(self, num_classes, feat_dim, device):
    super().__init__()
    self.num_classes = num_classes
    self.feat_dim = feat_dim
    self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

  def forward(self, z, labels):
    centers_batch = self.centers[labels]
    loss = ((z - centers_batch) ** 2).sum(dim=1).mean()
    return loss

def dann_lambda(step, max_steps):
  p = step / max_steps
  return 2. / (1. + np.exp(-10 * p)) - 1

def freeze(module):
  for p in module.parameters():
    p.requires_grad = False