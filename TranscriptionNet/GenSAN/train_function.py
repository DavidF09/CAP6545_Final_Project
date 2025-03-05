import time
import copy
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config_parser import Device
from PMSELoss import PMSELoss
from GenSAN.utils import adjust_learning_rate
from GenSAN.model import GenSAN_model


def train_function(model, train_dataloader, optimizer, beta):
    """
    GenSAN model training function.

    Args:
        model : GenSAN initialization model.
        train_dataloader : Training set batch data.
        optimizer : Adam optimizer.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).

    Returns:
        Float: Combination loss, mse loss and pearson_loss of train sets.
    """

    train_loss = 0
    mse_loss = 0
    pcc_loss = 0
    num_batches = len(train_dataloader)

    model.train()
    for node_feature, gecs_data in train_dataloader:
        node_feature = node_feature.to(Device())
        gecs_data = gecs_data.to(Device())

        optimizer.zero_grad()
        predict_gecs = model(node_feature)
        loss, mse, pcc = PMSELoss(gecs_data, predict_gecs, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            mse_loss += mse.item()
            pcc_loss += pcc.item()

    train_loss /= num_batches
    mse_loss /= num_batches
    pcc_loss /= num_batches
    return train_loss, mse_loss, pcc_loss


def valid_function(model, valid_dataloader, beta):
    """
    GenSAN model validation function.

    Args:
        model : The GenSAN model after training on the training set.
        valid_dataloader : Validation set batch data.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).

    Returns:
        Float: Combination loss, mse loss and pearson_loss of valid sets.
    """

    valid_loss = 0
    mse_loss = 0
    pcc_loss = 0
    num_batches = len(valid_dataloader)

    model.eval()
    with torch.no_grad():
        for node_feature, gecs_data in valid_dataloader:
            node_feature = node_feature.to(Device())
            gecs_data = gecs_data.to(Device())

            predict_gecs = model(node_feature)
            loss, mse, pcc = PMSELoss(gecs_data, predict_gecs, beta)
            valid_loss += loss.item()
            mse_loss += mse.item()
            pcc_loss += pcc.item()

    valid_loss /= num_batches
    mse_loss /= num_batches
    pcc_loss /= num_batches
    return valid_loss, mse_loss, pcc_loss


def test_evaluate(best_model, pre_gecs_test, gecs_test):
    """
    GenSAN model test evaluation function.

    Args:
        best_model : The GenSAN model after all iterations of training.
        pre_gecs_test (tensor): Test set of pre-GECs.
        gecs_test (ndarray): Test set of GECs data.
    """

    feature_test = pre_gecs_test.to(Device())
    feature_test_predict = best_model(feature_test).cpu().detach().numpy()

    # feature_test_predict_df = pd.DataFrame(feature_test_predict, index=net_test.index)
    # feature_test_predict_df.to_csv(save_path + "feature_test_predict.csv", index=True)

    d = []
    pcc = []
    for i in range(feature_test_predict.shape[0]):
        pearson = np.corrcoef(feature_test_predict[i], gecs_test[i])[0, 1]
        item_d, _ = stats.ks_2samp(feature_test_predict[i], gecs_test[i])
        pcc.append(pearson)
        d.append(item_d)
    abs_pcc = abs(np.array(pcc))
    abs_pcc_mean = abs_pcc.mean()
    d_mean = np.array(d).mean()

    mse = mean_squared_error(gecs_test, feature_test_predict)

    print('=' * 30)
    print('test evaluate result:\nAverage pcc: {}\nAverage mse: {}\nAverage D: {}'
          .format(abs_pcc_mean, mse, d_mean))
    print('=' * 30)

    # return abs_pearson


def plot_loss_figure(epochs, train_loss, valid_loss):
    """
    Draw the training loss value image

    Args:
        epochs (int): Number of epochs for FunDNN model training
        train_loss (list): List of training set loss values.
        valid_loss (list): List of valid set loss values.
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7, 7), dpi=144)

    plt.title('Loss figure')
    plt.plot(range(epochs), train_loss, color='red', linestyle='--', label='train loss', linewidth=2)
    plt.plot(range(epochs), valid_loss, color='dodgerblue', linestyle='-', label='valid loss', linewidth=2)
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.show()


def train(epochs, model, train_dataloader, valid_dataloader, optimizer, beta, warmup_epoch, learning_rate):
    """
    Train the GenSAN model.

    Args:
        epochs (int): Number of epochs for GenSAN model training
        model (nn.Module): GenSAN model
        train_dataloader : Training set batch data.
        valid_dataloader : Validation set batch data.
        optimizer : Adam optimizer.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).
        warmup_epoch (int): Number of warm-up epochs for GenSAN model training
        learning_rate (float): The initial learning rate.
    Returns:
        best_model (nn.Module): The trained GenSAN model with the lowest validation loss.
    """

    min_loss = float("inf")
    best_model = None

    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        tra_loss, tra_mse_loss, tra_cor_loss = train_function(model=model,
                                                              train_dataloader=train_dataloader,
                                                              optimizer=optimizer,
                                                              beta=beta)
        val_loss, val_mse_loss, val_cor_loss = valid_function(model=model,
                                                              valid_dataloader=valid_dataloader,
                                                              beta=beta)
        train_losses.append(tra_loss)
        valid_losses.append(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            best_model = copy.deepcopy(model)

        adjust_learning_rate(optimizer=optimizer, warmup_epoch=warmup_epoch, current_epoch=epoch, max_epoch=epochs,
                             lr_min=0, lr_max=learning_rate, warmup=True)

        print('end of epoch:{:3d} | time:{:5.2f}s | train loss:{:5.5f} | valid loss:{:5.5f} | train MseLoss:{:5.5f} | '
              'train PccLoss:{:5.5f} | valid MseLoss:{:5.5f} | valid PccLoss:{:5.5f}'
              .format(epoch, (time.time() - epoch_start_time), tra_loss, val_loss, tra_mse_loss, tra_cor_loss,
                      val_mse_loss, val_cor_loss))

    # plot_loss_figure(epochs, train_losses, valid_losses)

    return best_model


def feature_predict(input_matrix, best_model, pre_GECs, scaler, save_path, name, length=64):
    """
    Predict the GECs and inverse MinMaxScaler

    Args:
        input_matrix (tensor): 3D tensor composed of pre-GECs of RNAi, OE, and CRISPR.
        best_model (nn.Module): The trained GenSAN model with the lowest validation loss.
        length (int): Divide the length of input_matrix.Defaults to 64.
        pre_GECs (dataframe): Single pre-GECs data(RNAi, OE or CRISPR).
        scaler: MinMaxScaler object.
        save_path : Path to save predict GECs.
        name (str): GECs type(RNAi, OE or CRISPR)
    Returns:
        predict_GECs (dataframe): predict GECs.
    """

    sub_matrix = [input_matrix[i:i + length, :, :] for i in range(0, len(input_matrix), length)]
    predict_GECs = []
    for sub in range(len(sub_matrix)):
        inputs = sub_matrix[sub].to(Device())
        sub_predict_GECs = pd.DataFrame(best_model(inputs).cpu().detach().numpy())
        predict_GECs.append(sub_predict_GECs)
    predict_GECs = pd.concat(predict_GECs, ignore_index=True)

    inverse_predict_GECs = scaler.inverse_transform(predict_GECs.values)
    inverse_predict_GECs = pd.DataFrame(inverse_predict_GECs, index=pre_GECs.index)
    inverse_predict_GECs.to_csv(save_path + name + "_predict_GECs.csv", index=True, sep=",")

    print('\npredict finish:\npredict GECs:{}\n'.format(inverse_predict_GECs.shape))
    return predict_GECs


def run_GenSAN_model(blocks, GECs_dimension, hidden_nodes, heads, dropout_rate, recycles,
                     learning_rate, weight_decay,
                     epochs, train_dataloader, valid_dataloader, beta, warmup_epoch,
                     pre_gecs_test, gecs_test, save_path,
                     input_matrix, length, pre_GECs, scaler, name):
    """
    FunDNN model training process.

    Args:
        blocks (int): Number of transformer encoder units for GenSAN.
        GECs_dimension (int): The last dimension of GEC size(978).
        hidden_nodes (int): Number of nodes in each feed forward neural network layer.
        heads (int): Number of attention heads of row-wise self-attention block.
        dropout_rate (float): Dropout layer ratio of GenSAN.
        recycles (int): Recycle times of GenSAN model.
        learning_rate (float): Adam optimizer initial learning rate.
        weight_decay (float): Adam optimizer weight decay parameter.
        epochs (int): Number of epochs for GenSAN model training
        train_dataloader : Training set batch data.
        valid_dataloader : Validation set batch data.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).
        warmup_epoch (int):Number of warm-up epochs for GenSAN model training
        pre_gecs_test (tensor): Test set of pre-GECs..
        gecs_test (ndarray): Test set of GECs data.
        save_path : Path to save predict GECs and best model.
        input_matrix (tensor): 3D tensor composed of pre-GECs of RNAi, OE, and CRISPR.
        length (int): Divide the length of input_matrix
        pre_GECs (dataframe): Single pre-GECs data(RNAi, OE or CRISPR).
        scaler: MinMaxScaler object.
        name (str): GECs type(RNAi, OE or CRISPR)

    Returns:
        best_model (nn.Module): The trained GenSAN model with the lowest validation loss.
    """

    model = GenSAN_model(blocks=blocks,
                         GECs_dimension=GECs_dimension,
                         hidden_nodes=hidden_nodes,
                         heads=heads,
                         dropout_rate=dropout_rate,
                         recycles=recycles)
    model = model.to(Device())

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    best_model = train(epochs=epochs,
                       model=model,
                       train_dataloader=train_dataloader,
                       valid_dataloader=valid_dataloader,
                       optimizer=optimizer,
                       beta=beta,
                       warmup_epoch=warmup_epoch,
                       learning_rate=learning_rate)

    test_evaluate(best_model, pre_gecs_test, gecs_test)

    # torch.save(best_model, save_path + "FunDNN best model.pt")

    predict_GECs = feature_predict(input_matrix=input_matrix,
                                   length=length,
                                   best_model=best_model,
                                   pre_GECs=pre_GECs,
                                   scaler=scaler,
                                   save_path=save_path,
                                   name=name)
    return predict_GECs
