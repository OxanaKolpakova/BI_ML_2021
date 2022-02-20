import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i] == 0:
            TN += 1
        elif y_pred[i] == y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        else:
            FP += 1
        
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precison = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / len(y_true)
    return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = 1 - (np.sum(np.square(y_true - y_pred) / np.sum(np.square(y_true - np.mean(y_true)))))
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    MSE = (1 / y_pred.shape[0]) * (np.sum(np.square(y_true - y_pred)))
    return MSE



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    MAE = (1 / y_pred.shape[0]) * np.sum(np.abs(y_true - y_pred))
    return MAE
    