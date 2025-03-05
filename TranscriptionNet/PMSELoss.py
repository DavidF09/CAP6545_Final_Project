import torch
import torch.nn as nn


def PMSELoss(true, predict, beta):
    """
    Loss function combining the mean squared error (MSE) and Pearson correlation losses.

    Args:
        true (tensor): Tensor data of true GECs data.
        predict (tensor): Tensor data of GECs predicted by the model.
        beta (float): Weight hyperparameter of the combination of mse and pcc(see Class TranscriptionNet_Hyperparameters).

    Returns:
        Float: Combination loss, mse loss and pearson_loss.
    """

    error = true - predict
    sqr_error = torch.square(error)
    mse_loss = torch.mean(sqr_error)

    m_pred = torch.mean(predict, dim=1, keepdim=True)
    m_true = torch.mean(true, dim=1, keepdim=True)
    pred_m, true_m = predict - m_pred, true - m_true

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pearson_loss = torch.mean(1 - cos(pred_m, true_m))

    loss = (1 - beta) * mse_loss + beta * pearson_loss
    return loss, mse_loss, pearson_loss
