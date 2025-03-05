import math
import copy
import torch
import torch.nn as nn
from math import cos, pi
import torch.nn.functional as f


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 GECs_dimension: int,
                 hidden_nodes: int,
                 dropout_rate: float):
        """
        Feed forward neural network layer.

        Args:
            GECs_dimension (int): The last dimension of GEC size(978).
            hidden_nodes (int): Number of nodes in each layers for Feed forward neural network.
            dropout_rate (float, optional): Dropout layer ratio of Feed forward neural network.
        """

        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(GECs_dimension, hidden_nodes)
        self.w2 = nn.Linear(hidden_nodes, GECs_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w2(self.dropout(f.leaky_relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, dimension=978, eps=1e-6):
        """
        LayerNorm module.

        Args:
            dimension (int, optional): The dimension of norm.Defaults to 978
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(dimension))
        self.b2 = nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, dropout_rate):
        """
        SublayerConnection module.

        Args:
            dropout_rate (float): Dropout rate of each sub-layer.
        """
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, sublayer):
        """
        Forward pass logic.

        Args:
            x (Tensor): The input of the previous layer or sub-layer.
            sublayer (nn.Module): The sublayer function in the sublayer connection.
        """

        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, dropout=None):
    """
    Implementation of attention mechanism.

    Args:
        query: A tensor of queries.
        key: A tensor of key.
        value: A tensor of values.
        dropout (nn.Dropout): The dropout layer.Default is None
    """

    d_k = query.size(-1)
    scores = torch.as_tensor(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k))

    p_attn = f.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value)


class ColumnAttention(nn.Module):
    def __init__(self, dropout_rate):
        """
        column-wise attention.

        Args:
            dropout_rate (float): Dropout layer ratio of column-wise attention.
        """
        super(ColumnAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value):
        """
        Forward pass logic.

        Args:
            query (Tensor): A tensor of queries.
            key (Tensor): A tensor of key.
            value (Tensor): A tensor of values.
        """
        query = torch.transpose(query, -1, -2)
        key = torch.transpose(key, -1, -2)
        value = torch.transpose(value, -1, -2)

        col_attn = attention(query, key, value, dropout=self.dropout)

        col_attn = torch.transpose(col_attn, -1, -2)

        return col_attn


def clones(module, num_clone):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clone)])


class RowAttention(nn.Module):
    def __init__(self, heads, GECs_dimension, dropout_rate=0.1):
        """
        Multi-head attention.

        Args:
            heads (int): The number of heads.
            GECs_dimension (int): The last dimension of GEC size(978).
            dropout_rate (float): The dropout layer.Default is None
        """

        super(RowAttention, self).__init__()

        assert GECs_dimension % heads == 0

        self.d_k = GECs_dimension // heads
        self.heads = heads
        self.linears = clones(nn.Linear(GECs_dimension, GECs_dimension), 4)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = \
            [model(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        x = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        return self.linears[-1](x)


class Generator(nn.Module):
    def __init__(self,
                 GECs_dimension: int,
                 hidden_nodes: int,
                 dropout_rate: float):
        """
        Generator.

        Args:
            GECs_dimension (int): The last dimension of GEC size(978).
            hidden_nodes (int): Number of nodes in each layer.
            dropout_rate (float, optional): Dropout layer ratio.
        """

        super(Generator, self).__init__()
        self.w1 = nn.Linear(GECs_dimension, hidden_nodes)
        self.w2 = nn.Linear(hidden_nodes, GECs_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output_item = self.w2(self.dropout(torch.tanh(self.w1(x))))
        output = output_item[:, 0, :]
        return output


def adjust_learning_rate(optimizer, warmup_epoch, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    """
    Cosine preheating mechanism regulates learning rate.

    Args:
        optimizer (nn.Module): Adam optimizer.
        warmup_epoch (int): Number of warm-up epochs.
        current_epoch (int): Current epoch.
        max_epoch (int): Number of epochs for model training.
        lr_min (float, optional): Minimum learning rate. The default is 0
        lr_max (float, optional): Max learning rate. The default is 0.1
        warmup (bool, optional)
    """
    warmup_epoch = warmup_epoch if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
