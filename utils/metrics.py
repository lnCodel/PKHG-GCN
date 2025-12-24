# ============================
# Imports
# ============================

from math import log10
import os
import glob
import csv

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import t
import scipy.stats

import pingouin as pg

# Sklearn metrics
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

# Medical image related metrics
from medpy.metric.binary import sensitivity, specificity


# ============================
# Global flag
# ============================

# a = 1: input is logits (need argmax)
# a = 0: input is already predicted labels
a = 1


# ============================
# Image quality metric
# ============================

def PSNR(mse, peak=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        mse (float): Mean Squared Error
        peak (float): Maximum possible signal value

    Returns:
        float: PSNR value in dB
    """
    return 10 * log10((peak ** 2) / mse)


# ============================
# Utility class
# ============================

class AverageMeter(object):
    """
    Computes and stores the current value and running average.
    Commonly used during training to track loss or accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update meter with new value.

        Args:
            val (float): current value
            n (int): number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================
# Binary classification accuracy
# ============================

def accuracy(preds, labels):
    """
    Compute accuracy, sensitivity, and specificity.

    Args:
        preds (ndarray): model outputs (logits or labels)
        labels (ndarray): ground truth labels

    Returns:
        correct_num (int): number of correct predictions
        acc (float): accuracy
        spe (float): specificity
        sen (float): sensitivity
        spe_sum (int): true negatives count
        sen_sum (int): true positives count
    """
    x = np.sum(labels)                 # number of positive samples
    y = len(labels) - x                # number of negative samples

    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    spe = specificity(pred, labels)
    sen = sensitivity(pred, labels)

    correct_prediction = np.equal(pred, labels).astype(np.float32)

    sen_sum = round(x * sen)
    spe_sum = round(y * spe)

    return (
        np.sum(correct_prediction),
        np.mean(correct_prediction),
        spe,
        sen,
        spe_sum,
        sen_sum
    )


# ============================
# Save prediction probabilities
# ============================

def sa(sub_ids, preds, fold, n):
    """
    Save subject-level predictions and probabilities to CSV.

    Args:
        sub_ids (list): subject IDs
        preds (ndarray): logits
        fold (int): fold index
        n (int): offset index
    """
    base_path = r"/home/lining/Data/Res/"
    save_path = os.path.join(base_path, f"I_{fold}.csv")

    A = np.argmax(preds, 1)
    pos_probs = softmax(preds, axis=1)

    info = {
        "TEST_ID": [],
        "pred": [],
        "prob_0": [],
        "prob_1": []
    }

    for i in range(len(preds)):
        info["TEST_ID"].append(sub_ids[n - 100 + i])
        info["pred"].append(A[i])
        info["prob_0"].append(pos_probs[i][0])
        info["prob_1"].append(pos_probs[i][1])

    df = pd.DataFrame(info)
    df.to_csv(save_path)


# ============================
# Confidence interval
# ============================

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval for mean.

    Args:
        data (array-like)

    Returns:
        (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    z = 1.96

    lower_bound = mean - z * (std / np.sqrt(n))
    upper_bound = mean + z * (std / np.sqrt(n))

    return lower_bound, upper_bound


# ============================
# Save logits and labels
# ============================

def save_logit(preds, labels, named, region, asdw):
    """
    Save raw logits and labels to CSV files.
    """
    base_path = r"/homeb/lining/Data/experiment/KSR_logit123"
    save_path = os.path.join(base_path, named, str(asdw))
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f"{region}.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        for row in preds:
            writer.writerow(row)

    with open(os.path.join(save_path, f"{region}_label.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])


# ============================
# Save final predictions
# ============================

def save(preds, labels, local, named, region, asdw):
    """
    Save predicted labels with subject IDs.
    """
    preds = np.argmax(preds, 1)

    base_path = r"/homeb/lining/Data/experiment/Pro"
    save_path = os.path.join(base_path, named, str(asdw))
    os.makedirs(save_path, exist_ok=True)

    save_path_csv = os.path.join(save_path, f"{region}.csv")

    with open(save_path_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(["ids", f"{region}_true", f"{region}_pred"])

        for i in range(len(preds)):
            writer.writerow([local[i], labels[i], preds[i]])


# ============================
# Confusion matrix visualization
# ============================

def show_graph(conf_matrix_norm, conf_matrix):
    """
    Visualize normalized confusion matrix.
    """
    plt.rcParams.update({'font.size': 12})
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    class_labels = [
        'Others\n[0-6]',
        'Mild lesions\n[7-9]',
        'Normal tissue\n[10]'
    ]

    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.gca().xaxis.set_ticks_position('top')

    thresh = conf_matrix_norm.max() / 2.
    for i, j in np.ndindex(conf_matrix_norm.shape):
        plt.text(
            j, i,
            f"{conf_matrix_norm[i, j] * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            color="white" if conf_matrix_norm[i, j] > thresh else "black"
        )

    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    plt.show()


# ============================
# Per-class specificity
# ============================

def specificity_per_class1(y_true, y_pred):
    """
    Compute specificity for each class in multi-class classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    specificity = []

    for i in range(cm.shape[0]):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

    return np.array(specificity)


# ============================
# Intraclass Correlation Coefficient (ICC)
# ============================

def to_icc(preds, labels):
    """
    Compute ICC between predictions and labels.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    matrix = confusion_matrix(labels, pred)

    pred_list, real_list = [], []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix[i][j]
            pred_list.extend([j] * value)
            real_list.extend([i] * value)

    icc = icc_caculate(pred_list, real_list)
    return icc["ICC"][5]


def icc_caculate(pred_list, real_list):
    """
    Calculate ICC using pingouin.
    """
    ids = list(range(len(pred_list))) * 2
    judges = ["pred"] * len(pred_list) + ["real"] * len(real_list)
    scores = pred_list + real_list

    df = pd.DataFrame({
        "id": ids,
        "judge": judges,
        "score": scores
    })

    return pg.intraclass_corr(
        data=df,
        targets="id",
        raters="judge",
        ratings="score"
    )


# ============================
# Micro-averaged specificity
# ============================

def spe_three(y_true, y_pred):
    """
    Compute micro-averaged specificity for multi-class classification.
    """
    cm = confusion_matrix(y_true, y_pred)

    TN, FP = 0, 0
    for i in range(cm.shape[0]):
        TN += np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        FP += np.sum(cm[:, i]) - cm[i, i]

    return TN / (TN + FP)


# ============================
# AUC from saved CSV
# ============================

def auc_la():
    """
    Load saved prediction CSV and compute AUC, sensitivity, specificity, and kappa.
    """
    base_path = r"/home/lining/GCN/M3/CKSA/data/Result"
    save_path = os.path.join(base_path, "All_I.csv")

    if not os.path.exists(save_path):
        return 0, 0, 0, 0

    df = pd.read_csv(save_path)

    auc_out = roc_auc_score(df["TRUE"], df["prob_1"])
    se = sensitivity(df["pred"], df["TRUE"])
    sp = specificity(df["pred"], df["TRUE"])
    ka = cohen_kappa_score(df["pred"], df["TRUE"])

    os.remove(save_path)
    return auc_out, se, sp, ka


# ============================
# AUC / PRF / Kappa helpers
# ============================

def auc(preds, labels):
    """
    Compute AUC using predicted labels.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    try:
        return roc_auc_score(labels, pred)
    except:
        return 0


def prf(preds, labels):
    """
    Compute precision, recall, and F1-score.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    p, r, f, _ = precision_recall_fscore_support(labels, pred, average="binary")
    return [p, r, f]


def interval1_std(data):
    """
    Compute 95% confidence interval radius.
    """
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    interval = t.interval(0.95, len(data) - 1, loc=mean, scale=std_err)
    return (interval[1] - interval[0]) / 2


def kappa(preds, GT):
    """
    Compute Cohen's Kappa score.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    return cohen_kappa_score(pred, GT)
