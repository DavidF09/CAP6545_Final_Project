import torch
import torch.nn as nn


class TranscriptionNet_Hyperparameters(object):
    """Defines the default TranscriptionNet config parameters."""

    def __init__(self):
        # FunDNN Model
        self.FunDNN_layers = 5  # Number of layers for FunDNN
        self.FunDNN_epochs = 1000  # Number of epochs for FunDNN model training
        self.FunDNN_batch_size = 32  # Number of GECs in each batch
        self.FunDNN_hidden_nodes = 1024  # Number of nodes in each layers for FunDNN
        self.FunDNN_dropout_rate = 0.1  # Dropout layer ratio of FunDNN
        self.FunDNN_activation_func = nn.LeakyReLU  # Activation function between each layer of FunDNN
        self.FunDNN_learning_rate = 0.00035  # Adadelta optimizer learning rate
        self.FunDNN_RNAi_path = "example_data/datasets/RNAi/"  # RNAi GECs and node feature path
        self.FunDNN_OE_path = "example_data/datasets/RNAi/"  # OE GECs and node feature path
        self.FunDNN_CRISPR_path = "example_data/datasets/RNAi/"  # CRISPR GECs and node feature path
        self.FunDNN_save_path = "example_data/result/"

        # GenSAN Model
        self.GenSAN_heads = 2  # Number of attention heads of row-wise self-attention block
        self.GenSAN_blocks = 3  # Number of transformer encoder units for GenSAN
        self.GenSAN_recycles = 3  # Recycle times of GenSAN model
        self.GenSAN_epochs = 110  # Number of epochs for GenSAN model training
        self.GenSAN_warmup_epochs = 5  # Number of warm-up epochs for GenSAN model training
        self.GenSAN_batch_size = 32  # Number of GECs in each batch
        self.GenSAN_GECs_dimension = 978  # The last dimension of GEC size(978).
        self.GenSAN_hidden_nodes = 1024  # Number of nodes in each feed forward neural network layer
        self.GenSAN_dropout_rate = 0.05  # Dropout layer ratio of GenSAN
        self.GenSAN_learning_rate = 0.0000045  # Adam optimizer learning rate
        self.GenSAN_weight_decay = 1e-5  # Adam optimizer weight_decay
        self.GenSAN_save_path = "example_data/result/"

        self.PMSELoss_beta = 0.1  # Weight hyperparameter of the combination of mse and pcc


class Device:
    """Returns the currently used device by calling `Device()`.

    Returns:
        str: Either "cuda" or "cpu".
    """

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls) -> str:
        return cls._device
