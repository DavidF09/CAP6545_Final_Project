import copy
import torch.nn as nn
from GenSAN.utils import ColumnAttention, RowAttention, PositionwiseFeedForward, Generator
from GenSAN.module import BlockLayer, Blocks, BlocksRecycling


class GenSAN(nn.Module):
    def __init__(self, util, generator):
        """
        GenSAN model

        Args:
            util (nn.Module): Transformer encoder blocks with recycling.
            generator (nn.Module): Generator block.
        """
        super(GenSAN, self).__init__()
        self.util = util
        self.generator = generator

    def forward(self, x):
        output = self.util(x)
        return self.generator(output)


def GenSAN_model(blocks=3, GECs_dimension=978, hidden_nodes=1024, heads=6, dropout_rate=0.2, recycles=1):
    """
    GenSAN model instantiate.

    Args:
        blocks (int): Number of transformer encoder units for GenSAN
        GECs_dimension (int): The last dimension of GEC size(978).
        hidden_nodes (int): Number of nodes in each feed forward neural network layer
        heads (int): Number of attention heads of row-wise self-attention block
        dropout_rate (float): Dropout layer ratio of GenSAN
        recycles (int): Recycle times of GenSAN model
    """

    c = copy.deepcopy

    col_attn = ColumnAttention(dropout_rate)
    row_attn = RowAttention(heads, GECs_dimension, dropout_rate)

    ff = PositionwiseFeedForward(GECs_dimension, hidden_nodes, dropout_rate)

    model = GenSAN(
        BlocksRecycling(Blocks(BlockLayer(GECs_dimension, c(col_attn), c(row_attn), c(ff), dropout_rate), blocks),
                        recycles),
        Generator(GECs_dimension, hidden_nodes, dropout_rate))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
