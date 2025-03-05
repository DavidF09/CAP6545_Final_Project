import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_GECs(input_path, file_name):
    """
    Preprocesses input GECs dataSets.

    Args:
        input_path: Paths to input DataSets.
        file_name (str): GECs dataSets folder name.

    Returns:
        Dataframe: train, valid and test sets. Each row is a node, each column is a feature.
    """
    file_path = input_path + file_name
    with open(file_path, 'rb') as f:
        datasets_dict = pickle.load(f)

    train = datasets_dict["train"].sort_index()
    valid = datasets_dict["valid"].sort_index()
    test = datasets_dict["test"].sort_index()
    return train, valid, test


def GECs_combine(true_GECs, predict_GECs):
    """
    Combines true and predicted GECs dataSets.

    Args:
        true_GECs (DataFrame): True GECs dataSets.
        predict_GECs (DataFrame): Predicted GECs dataSets.

    Returns:
        Dataframe: Combined dataSets.
    """
    predict = predict_GECs[~ predict_GECs.index.isin(true_GECs.index)]
    predict.columns = true_GECs.columns
    combine = pd.concat([true_GECs, predict], axis=0).sort_index()
    return combine


def get_datasets(GECs, pre_GECs, combine1, combine2):
    """
    Gets train, valid and test sets.

    Args:
        GECs (DataFrame): True GECs dataSets(train, valid or test).
        pre_GECs (DataFrame): Predicted GECs dataSets.
        combine1 (DataFrame): Combined GECs dataSets.
        combine2 (DataFrame): Combined GECs dataSets.

    Returns:
        ndarray: train, valid or test sets.
    """

    index = GECs.index.sort_values()

    # input
    pre_gecs_item = pre_GECs[pre_GECs.index.isin(index)].sort_index()
    combine1_item = combine1[combine1.index.isin(index)].sort_index()
    combine2_item = combine2[combine2.index.isin(index)].sort_index()

    datasets = []
    for i in range(len(index)):
        single_pre_gecs = pre_gecs_item[pre_gecs_item.index == index[i]].values
        single_combine1 = combine1_item[combine1_item.index == index[i]].values
        single_combine2 = combine2_item[combine2_item.index == index[i]].values
        single_data = np.concatenate((single_pre_gecs, single_combine1, single_combine2), axis=0)
        datasets.append(single_data)

    datasets_array = np.array(datasets)
    return datasets_array


def datasets_scaled(trainSets, validSets, testSets):
    """
    Standardization of training, validation, and testing data sets.
    """

    train_gecs, train_cmap, train_gene = trainSets.shape
    valid_gecs, valid_cmap, valid_gene = validSets.shape
    test_gecs, test_cmap, test_gene = testSets.shape

    trainSets_item = trainSets.reshape((train_gecs, train_cmap * train_gene))
    validSets_item = validSets.reshape((valid_gecs, valid_cmap * valid_gene))
    testSets_item = testSets.reshape((test_gecs, test_cmap * test_gene))

    scaler = StandardScaler()
    train_scaled_item = scaler.fit_transform(trainSets_item)
    valid_scaled_item = scaler.transform(validSets_item)
    test_scaled_item = scaler.transform(testSets_item)

    train_scaled = train_scaled_item.reshape((train_gecs, train_cmap, train_gene))
    valid_scaled = valid_scaled_item.reshape((valid_gecs, valid_cmap, valid_gene))
    test_scaled = test_scaled_item.reshape((test_gecs, test_cmap, test_gene))

    train = torch.FloatTensor(train_scaled)
    valid = torch.FloatTensor(valid_scaled)
    test = torch.FloatTensor(test_scaled)
    return train, valid, test


def get_pre_GECs(pre_GECs, combine1, combine2):
    """
    GenSAN model prediction data

    Args:
        pre_GECs (DataFrame): Predicted GECs dataSets.
        combine1 (DataFrame): Combined GECs dataSets.
        combine2 (DataFrame): Combined GECs dataSets.

    Returns:
        Tensor: 3D Tensor of prediction data.
    """

    index = pre_GECs.index.sort_values()

    pre_GECs = pre_GECs.sort_index()
    combine1 = combine1.sort_index()
    combine2 = combine2.sort_index()

    predict_data = []
    for i in range(len(index)):
        pre_GECs_item = pre_GECs[pre_GECs.index == index[i]].values
        combine1_item = combine1[combine1.index == index[i]].values
        combine2_item = combine2[combine2.index == index[i]].values
        predict_data_item = np.concatenate((pre_GECs_item, combine1_item, combine2_item), axis=0)
        predict_data.append(predict_data_item)

    predict_data = torch.FloatTensor(np.array(predict_data))

    return predict_data


def GenSAN_preprocessor(true_GECs1, true_GECs2, predict_GECs1, predict_GECs2, pre_GECS, input_path, file_name):
    """
    GenSAN data preprocessing
    """

    combine_GECs1 = GECs_combine(true_GECs1, predict_GECs1)
    combine_GECs2 = GECs_combine(true_GECs2, predict_GECs2)

    train_GECs, valid_GECs, test_GECs = load_GECs(input_path, file_name)

    train = get_datasets(train_GECs, pre_GECS, combine_GECs1, combine_GECs2)
    valid = get_datasets(valid_GECs, pre_GECS, combine_GECs1, combine_GECs2)
    test = get_datasets(test_GECs, pre_GECS, combine_GECs1, combine_GECs2)

    train_data, valid_data, test_data = datasets_scaled(train, valid, test)

    print('pre-GECS dimension:\ntrain data:{}\nvalid data:{}\ntest data:{}\n'
              .format(train_data.shape, valid_data.shape, test_data.shape))
    
    return train_data, valid_data, test_data, combine_GECs1, combine_GECs2
