import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import mlflow
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_positive_rate, false_negative_rate

def predictive_equality_difference(y_true, y_pred, sensitive_attr):
    """
    Predictive equality difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Predictive equality difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fp_rate(y_true_0, y_pred_0) - fp_rate(y_true_1, y_pred_1))

def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    """
    Equal opportunity difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Equal opportunity difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fn_rate(y_true_0, y_pred_0) - fn_rate(y_true_1, y_pred_1))

def tp_rate(y_true, y_pred):
    """
    True positive rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        True positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)

def fp_rate(y_true, y_pred):
    """
    False positive rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        False positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fp + tn) == 0:
        return 0
    return fp / (fp + tn)

def fn_rate(y_true, y_pred):
    """
    False negative rate

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels

    Returns
    -------
    float
        False negative rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fn + tp) == 0:
        return 0
    return fn / (fn + tp)

def eq_odds_difference(y_true, y_pred, sensitive_attr):
    """
    Equalized odds difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Equalized odds difference
    """
    # TPR difference

    tpr_0 = tp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    tpr_1 = tp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    tpr_diff = abs(tpr_0 - tpr_1)

    # FPR difference
    fpr_0 = fp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    fpr_1 = fp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    fpr_diff = abs(fpr_0 - fpr_1)

    return max(tpr_diff, fpr_diff)

def auc_difference(y_true, y_pred_proba, sensitive_attr):
    """
    AUC difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        AUC difference
    """
    try:
        auc_0 = roc_auc_score(y_true.loc[sensitive_attr == 0], y_pred_proba.loc[sensitive_attr == 0])
    except ValueError:
        auc_0 = 0
    try:
        auc_1 = roc_auc_score(y_true.loc[sensitive_attr == 1], y_pred_proba.loc[sensitive_attr == 1])
    except ValueError:
        auc_1 = 0

    return auc_0 - auc_1

def evaluate(y_test:pd.Series, y_pred_proba, sensitive_attr):
    """
    Calculate and log evaluation metrics to MLflow

    Parameters
    ----------
    y_test : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    y_pred_proba : numpy.ndarray
        Predicted positive label probabilities
    sensitive_attr : pd.Series
        Sensitive attribute
    """
    if len(set(y_pred_proba)) > 1 and len(y_test.unique()) > 1:
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
        mlflow.log_metric("auc_difference", auc_difference(y_test, pd.Series(y_pred_proba, index=y_test.index), sensitive_attr))
    
    for thresh in [0.2, 0.5, 0.8]:
        y_pred = pd.Series(np.where(y_pred_proba > thresh, 1, 0), index=y_test.index)

        mlflow.log_metric(f"accuracy_{thresh}", accuracy_score(y_test, y_pred))
        mlflow.log_metric(f"demographic_parity_difference_{thresh}", demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attr))
        mlflow.log_metric(f"equalized_odds_difference_{thresh}", eq_odds_difference(y_test, y_pred, sensitive_attr))
        mlflow.log_metric(f"predictive_equality_difference_{thresh}", predictive_equality_difference(y_test, y_pred, sensitive_attr))
        mlflow.log_metric(f"equal_opportunity_difference_{thresh}", equal_opportunity_difference(y_test, y_pred, sensitive_attr))

def evaluate_correction(y:pd.Series, y_train_corrected:pd.Series, y_test_corrected:pd.Series):
    """
    Evaluate the similarity of the corrected labels to the original ones

    Parameters
    ----------
    y : pd.Series
        Original labels
    y_train_corrected : pd.Series
        Corrected training labels
    y_test_corrected : pd.Series
        Corrected test labels

    Returns
    -------
    float
        Accuracy
    float
        False positive rate
    float
        False negative rate
    """
    original_labels = y.sort_index()
    corrected_labels = pd.concat([y_train_corrected, y_test_corrected]).sort_index()
    
    acc = accuracy_score(original_labels.values, corrected_labels.values)
    fpr = fp_rate(original_labels.values, corrected_labels.values)
    fnr = fn_rate(original_labels.values, corrected_labels.values)

    return acc, fpr, fnr