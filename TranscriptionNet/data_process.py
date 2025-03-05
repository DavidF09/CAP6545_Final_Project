import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_rawdata(input_path):
    """
    Load raw data(node feature and true GECs)
    """
    datasets = []
    file_paths = os.listdir(input_path)
    for file in file_paths:
        file_path = os.path.join(input_path, file)
        data = pd.read_csv(file_path, sep=",", index_col=0).sort_index()
        datasets.append(data)

    RNAi = datasets[3].sort_index()
    OE = datasets[2].sort_index()
    CRISPR = datasets[0].sort_index()
    node_feature = datasets[1].sort_index()
    return node_feature, RNAi, OE, CRISPR


def train_test_val_split(GECs, ratio_train, ratio_test, ratio_valid):
    """
    Split the data into train, test, and validation sets.
    """
    train, middle = train_test_split(GECs, train_size=ratio_train, test_size=ratio_test + ratio_valid, random_state=0)
    ratio = ratio_valid / (1 - ratio_train)
    test, validation = train_test_split(middle, test_size=ratio, random_state=0)

    train = train.sort_index()
    test = test.sort_index()
    validation = validation.sort_index()
    return train, test, validation


def save_datasets(df_dict, file_path):
    """
    Save the datasets as pickle files.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(df_dict, f)


def datasets_split(GECs, feature, save_path):
    """
    Datasets split and data scaler.
    """
    GECs_filter = GECs[GECs.index.isin(feature.index)]
    GECs_train, GECs_test, GECs_valid = train_test_val_split(GECs_filter, 0.7, 0.2, 0.1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    GECs_train_scaled = scaler.fit_transform(GECs_train.values)
    GECs_test_scaled = scaler.fit_transform(GECs_test.values)
    GECs_valid_scaled = scaler.fit_transform(GECs_valid.values)
    
    GECs_train = pd.DataFrame(GECs_train_scaled, index=GECs_train.index, columns=GECs_filter.columns).sort_index()
    GECs_test = pd.DataFrame(GECs_test_scaled, index=GECs_test.index, columns=GECs_filter.columns).sort_index()
    GECs_valid = pd.DataFrame(GECs_valid_scaled, index=GECs_valid.index, columns=GECs_filter.columns).sort_index()

    feature_train = feature[feature.index.isin(GECs_train.index)].sort_index()
    feature_test = feature[feature.index.isin(GECs_test.index)].sort_index()
    feature_valid = feature[feature.index.isin(GECs_valid.index)].sort_index()

    GECs_dict = {'train': GECs_train, 'valid': GECs_valid, 'test': GECs_test}
    feature_dict = {'train': feature_train, 'valid': feature_valid, 'test': feature_test}

    save_datasets(GECs_dict, save_path + 'GECs_dict.pkl')
    save_datasets(feature_dict, save_path + 'feature_dict.pkl')

    return scaler
