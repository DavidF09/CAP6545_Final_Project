import torch
import torch.nn as nn
from config_parser import Device
from FunDNN.train_function import train, test_evaluate, feature_predict


class FunDNN(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_nodes: int,
                 activate_func: nn.Module,
                 dropout_rate: float):
        """
        The FunDNN model.

        Args:
            num_layers (int): Number of layers for FunDNN.
            hidden_nodes (int): Number of nodes in each layers for FunDNN.
            activate_func (nn.Module): Activation function between each layer of FunDNN.
            dropout_rate (float, optional): Dropout layer ratio of FunDNN.
        """

        super(FunDNN, self).__init__()
        layers = [nn.Linear(512, hidden_nodes), activate_func(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 3):
            layers.extend([nn.Linear(hidden_nodes, hidden_nodes), activate_func(), nn.Dropout(dropout_rate)])
        layers.extend([nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_nodes, 978))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass logic.

        Args:
            x (Tensor): 2D tensor of node features.Each row is a node, each column is a feature.

        Returns:
            Tensor: 2D tensor of .
        """
        return self.model(x)


def run_model(num_layers, hidden_nodes, activate_func, dropout_rate,
              learning_rate, epochs, train_dataloader, valid_dataloader, beta,
              feature_test, gecs_test, save_path, node_feature, name):
    """
    FunDNN model training process.

    Args:
        num_layers (int): Number of layers for FunDNN.
        hidden_nodes (int): Number of nodes in each layers for FunDNN.
        activate_func (nn.Module): Activation function between each layer of FunDNN.
        dropout_rate (float): Dropout layer ratio of FunDNN
        learning_rate (float): Adadelta optimizer learning rate
        epochs (int): Number of epochs for FunDNN model training
        train_dataloader: Training set batch data.
        valid_dataloader: Validation set batch data.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).
        feature_test (tensor): Test set of node features.
        gecs_test (ndarray): Test set of GECs data.
        save_path: Path to save the trained model.
        node_feature: All network nodes embedded features.
        name: GECs type(RNAi, OE or CRISPR)

    Returns:
        best_model (nn.Module): The trained FunDNN model with the lowest validation loss.
    """

    model = FunDNN(num_layers=num_layers,
                   hidden_nodes=hidden_nodes,
                   activate_func=activate_func,
                   dropout_rate=dropout_rate)
    model = model.to(Device())

    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    best_model = train(epochs=epochs,
                       model=model,
                       train_dataloader=train_dataloader,
                       valid_dataloader=valid_dataloader,
                       optimizer=optimizer,
                       beta=beta)

    test_evaluate(best_model, feature_test, gecs_test)

    # torch.save(best_model, save_path + "FunDNN best model.pt")

    pre_GECs = feature_predict(best_model=best_model,
                               node_feature=node_feature,
                               save_path=save_path,
                               name=name)
    return pre_GECs
