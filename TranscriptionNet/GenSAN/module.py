import torch.nn as nn
from GenSAN.utils import clones, SublayerConnection, LayerNorm


class BlockLayer(nn.Module):
    def __init__(self, GECs_dimension, col_attn, row_attn, feed_forward, dropout_rate):
        """
        Single transformer encoder block.

        Args:
            GECs_dimension (int): The last dimension of GEC size(978).
            col_attn (nn.Module): attention object for column attention.
            row_attn (nn.Module): Multi-head attention object for row attention.
            feed_forward (nn.Module): Feed forward neural network object.
            dropout_rate (float, optional): Dropout layer ratio.
        """
        super(BlockLayer, self).__init__()
        self.GECs_dimension = GECs_dimension
        self.col_attn = col_attn
        self.row_attn = row_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(dropout_rate), 3)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.row_attn(x, x, x))

        x = self.sublayer[1](x, lambda x: self.col_attn(x, x, x))

        output = self.sublayer[2](x, self.feed_forward)
        return output


class Blocks(nn.Module):
    def __init__(self, layer, blocks):
        """
        Transformer encoder blocks.

        Args:
            layer (nn.Module): Single transformer encoder block.
            blocks (int): Number of transformer encoder units for GenSAN.
        """

        super(Blocks, self).__init__()
        self.layers = clones(layer, blocks)
        self.norm = LayerNorm()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class BlocksRecycling(nn.Module):
    def __init__(self, block_layers, recycles):
        """
        Transformer encoder blocks with recycling

        Args:
            block_layers (nn.Module): Single transformer encoder block.
            recycles (int): Recycle times of Transformer encoder blocks
        """

        super(BlocksRecycling, self).__init__()
        self.models = clones(block_layers, recycles)
        self.norm = LayerNorm()

    def forward(self, x):
        for model in self.models:
            x = model(x)
            x = self.norm(x)
            x += x
        return x


