from math import log10
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import glob
import csv

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)
from scipy.special import softmax
from scipy.stats import t
import scipy.stats
import pingouin as pg

from medpy.metric.binary import sensitivity, specificity

# Global flag controlling whether inputs are logits or class labels
a = 1


def PSNR(mse, peak=1.):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    """
    return 10 * log10((peak ** 2) / mse)


class AverageMeter(object):
    """
    Computes and stores the current value, sum, and running average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, labels):
    """
    Compute accuracy, specificity, sensitivity, and related statistics.
    """
    x = np.sum(labels)
    y = len(labels) - x

    # Convert logits to predicted labels if required
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


def sa(sub_ids, preds, fold, n):
    """
    Save subject-level predictions and class probabilities to a CSV file.
    """
    base_path = r"/home/lining/Data/Res/"
    save_path = os.path.join(base_path, f"I_{fold}.csv")

    # Predicted class labels
    A = np.argmax(preds, 1)

    # Softmax probabilities
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


def compute_confidence_interval(data):
    """
    Compute the 95% confidence interval of the mean.
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    z = 1.96  # Z-score for 95% confidence interval

    lower_bound = mean - z * (std / np.sqrt(n))
    upper_bound = mean + z * (std / np.sqrt(n))

    return lower_bound, upper_bound


def save(preds, labels, local, named, region, asdw):
    """
    Save prediction results for a specific region into a CSV file.
    """
    preds = np.argmax(preds, 1)

    base_path = r"/homeb/lining/Data/experiment/KSR"
    save_path = os.path.join(base_path, named)

    # Create directories if they do not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, str(asdw))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_csv = os.path.join(save_path, f"{region}.csv")

    with open(save_path_csv, mode="a", newline='') as file:
        writer = csv.writer(file)

        # Write header if file is empty
        if file.tell() == 0:
            header = ["ids", f"{region}_true", f"{region}_pred"]
            writer.writerow(header)

        # Write prediction rows
        for i in range(len(preds)):
            row = [local[i], labels[i], preds[i]]
            writer.writerow(row)


def Over_all(named, asdw, a):
    """
    Aggregate regional prediction CSV files and compute overall metrics.
    """
    base_path = r"/homeb/lining/Data/experiment/KSR"
    save_path = os.path.join(base_path, named, str(asdw))
    save_path1 = os.path.join(save_path, f"{named}_All")

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    # Collect all region CSV files
    csv_files = glob.glob(os.path.join(save_path, '*.csv'))

    # Sort and remove duplicate IDs in each CSV file
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.sort_values(by='ids', ascending=True)
        df = df.drop_duplicates(subset='ids', keep='first')
        df.to_csv(csv_file, index=False)

    # Merge all region CSV files
    combined_data = pd.DataFrame()
    column_name = ["ids"]

    for csv_file in csv_files:
        base = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)

        column_name.append(f"{base[:-4]}_true")
        column_name.append(f"{base[:-4]}_pred")

        if not combined_data.empty:
            df = df.iloc[:, 1:]

        combined_data = pd.concat([combined_data, df], axis=1, ignore_index=True)

    save_path_new = os.path.join(save_path1, "All.csv")
    combined_data.columns = column_name
    combined_data.to_csv(save_path_new, index=False)

    print(f'Merged CSV files saved to {save_path_new}')

    # Load merged data for evaluation
    df = pd.read_csv(save_path_new).iloc[:, 1:].values

    # Separate true and predicted labels
    a_idx = list(range(0, 20, 2))
    b_idx = list(range(1, 20, 2))

    true = df[:, a_idx]
    true1 = true.flatten()
    true = np.sum(true, axis=1)

    pred = df[:, b_idx]
    pred1 = pred.flatten()
    pred = np.sum(pred, axis=1)

    # Compute ICC and binary classification metrics
    print(asdw, "1")
    print(to_icc(preds=(10 - pred), labels=(10 - true)))

    pred[pred < 6] = 0
    pred[pred >= 6] = 1
    true[true < 6] = 0
    true[true >= 6] = 1

    print("Binary classification:", accuracy(pred, true))
    print("All regions:", accuracy(pred1, true1))
    print(auc(pred, true))
    print()


def auc_la():
    """
    Load prediction results from CSV and compute evaluation metrics.
    """
    base_path = r"/home/lining/GCN/M3/CKSA/data/Result"
    save_path = os.path.join(base_path, "All_I.csv")

    if not os.path.exists(save_path):
        return 0, 0, 0

    df = pd.read_csv(save_path)

    auc_out = roc_auc_score(df["TRUE"], df["prob_1"])
    se = sensitivity(df["pred"], df["TRUE"])
    sp = specificity(df["pred"], df["TRUE"])
    ka = cohen_kappa_score(df["pred"], df["TRUE"])

    os.remove(save_path)
    return auc_out, se, sp, ka


def auc(preds, labels, is_logit=True):
    """
    Compute AUC score from predictions and labels.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    try:
        auc_out = roc_auc_score(labels, pred)
    except:
        auc_out = 0

    return auc_out


def prf(preds, labels, is_logit=True):
    """
    Compute precision, recall, and F1-score.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    p, r, f, s = precision_recall_fscore_support(
        labels, pred, average='binary'
    )
    return [p, r, f]


def interval1_std(data):
    """
    Compute half-width of the confidence interval based on t-distribution.
    """
    mean = np.mean(data)
    std_data = np.std(data) / np.sqrt(len(data))
    confidence = 0.95
    n = len(data)
    dof = n - 1

    interval = t.interval(confidence, dof, loc=mean, scale=std_data)
    radius = (interval[1] - interval[0]) / 2

    return radius


def kappa(preds, GT):
    """
    Compute Cohen's Kappa coefficient.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    ka = cohen_kappa_score(pred, GT)
    return ka


def to_icc(preds, labels):
    """
    Compute Intraclass Correlation Coefficient (ICC) from predictions and labels.
    """
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)

    matrix = confusion_matrix(labels, pred)

    pred_list = []
    real_list = []

    # Expand confusion matrix into label lists
    leng = len(matrix)
    for i in range(leng):
        for j in range(leng):
            value = matrix[i][j]
            pred_list.extend([j] * value)
            real_list.extend([i] * value)

    icc = icc_caculate(pred_list, real_list)
    icc_value = icc["ICC"][5]

    return icc_value


def icc_caculate(pred_list, real_list):
    """
    Calculate ICC using Pingouin based on prediction and ground-truth lists.
    """
    id_list = list(range(len(pred_list)))
    id_list.extend(range(len(real_list)))

    judge = ['pre'] * len(pred_list)
    judge.extend(['real'] * len(pred_list))

    score_list = pred_list.copy()
    score_list.extend(real_list)

    dic = {
        "id": id_list,
        "judge": judge,
        "score": score_list
    }

    excel = pd.DataFrame(dic)
    icc = pg.intraclass_corr(
        data=excel,
        targets='id',
        raters='judge',
        ratings='score'
    )

    return icc
