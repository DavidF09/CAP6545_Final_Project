import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler


def load_data(input_path, file_name):
    """
    Preprocesses input node features and GECs dataSets.

    Args:
        input_path: Paths to input DataSets.
        file_name (str): node features or GECs dataSets file name.

    Returns:
        Tensor: 2D tensor of train, valid and test sets. Each row is a node, each column is a feature.
    """

    file_path = input_path + file_name
    with open(file_path, 'rb') as f:
        datasets_dict = pickle.load(f)  

    if file_name == "feature_dict.pkl":
        scaler = StandardScaler()
        train_tensor = torch.FloatTensor(scaler.fit_transform(datasets_dict['train'].sort_index().values))
        valid_tensor = torch.FloatTensor(scaler.transform(datasets_dict['valid'].sort_index().values))
        test_tensor = torch.FloatTensor(scaler.transform(datasets_dict['test'].sort_index().values))

        print('Node feature dimension:\ntrain data:{}\nvalid data:{}\ntest data:{}\n'
              .format(train_tensor.shape, valid_tensor.shape, test_tensor.shape))
        return train_tensor, valid_tensor, test_tensor

    elif file_name == "GECs_dict.pkl":
        train_tensor = torch.FloatTensor(datasets_dict['train'].sort_index().values)
        valid_tensor = torch.FloatTensor(datasets_dict['valid'].sort_index().values)
        test_array = datasets_dict['test'].sort_index()

        print('GECS data dimension:\ntrain data:{}\nvalid data:{}\ntest data:{}\n'
              .format(train_tensor.shape, valid_tensor.shape, test_array.shape))
        return train_tensor, valid_tensor, test_array


def get_dataloader(batch_size, node_train, gecs_train, node_valid, gecs_valid):
    """
    Build batched data for training and validation sets.

    Args:
        batch_size (int): Batch size hyperparameter(see Class TranscriptionNet_Hyperparameters).
        node_train (tensor): Tensor data for training sets of node features.
        node_valid (tensor): Tensor data for testing sets of node features.
        gecs_train (tensor): Tensor data for training sets of GECs data.
        gecs_valid (tensor): Tensor data for testing sets of GECs data.

    Returns:
        Dataloader of train and valid sets.
    """
    train_dataset = TensorDataset(node_train, gecs_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(node_valid, gecs_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader




