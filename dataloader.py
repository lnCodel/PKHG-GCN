import os.path
import pandas as pd
import numpy as np
import torch
import random
import copy
import scipy.io as sio

import data.utils as Reader
from data import batchprocessing as ba

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN

# Mapping from region name to index
index = {
    "M1": 0, "M2": 1, "M3": 2, "M4": 3, "M5": 4,
    "M6": 5, "I": 6, "C": 7, "L": 8, "IC": 9, "ALL": 10
}

# Number of selected features per region
fnum = {
    "M1": 200, "M2": 200, "M3": 250, "M4": 250, "M5": 300,
    "M6": 250, "I": 250, "C": 300, "L": 250, "IC": 300
}

# Threshold scores per region
score = {
    "M1": 0.70, "M2": 0.65, "M3": 0.75, "M4": 0.65, "M5": 0.70,
    "M6": 0.70, "I": 0.65, "C": 0.70, "L": 0.70, "IC": 0.65
}


class dataloader():
    """
    Data loader and preprocessing class for multimodal graph-based learning.
    """

    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2
        self.dis_lis = []   # Disease-related features
        self.sub_lis = []   # Subject-related features
        self.hea_lis = []   # Healthy/control features
        self.y_lis = []     # Label lists

    def load_data(self, region):
        """
        Load multimodal data for a given region.
        Returns disease features, healthy features, labels, left/right info, and subject IDs.
        """
        subject_IDs = Reader.get_ids(path="")
        self.subject_IDs = subject_IDs

        # Load labels for different regions/scores
        self.y_M1 = Reader.get_subject_lable(subject_IDs, "M1")
        self.y_lis.append(self.y_M1)
        self.y_M2 = Reader.get_subject_lable(subject_IDs, "M2")
        self.y_lis.append(self.y_M2)
        self.y_M3 = Reader.get_subject_lable(subject_IDs, "M3")
        self.y_lis.append(self.y_M3)
        self.y_M4 = Reader.get_subject_lable(subject_IDs, "M4")
        self.y_lis.append(self.y_M4)
        self.y_M5 = Reader.get_subject_lable(subject_IDs, "M5")
        self.y_lis.append(self.y_M5)
        self.y_M6 = Reader.get_subject_lable(subject_IDs, "M6")
        self.y_lis.append(self.y_M6)
        self.y_I = Reader.get_subject_lable(subject_IDs, "I")
        self.y_lis.append(self.y_I)
        self.y_C = Reader.get_subject_lable(subject_IDs, "C")
        self.y_lis.append(self.y_C)
        self.y_L = Reader.get_subject_lable(subject_IDs, "L")
        self.y_lis.append(self.y_L)
        self.y_IC = Reader.get_subject_lable(subject_IDs, "IC")
        self.y_lis.append(self.y_IC)
        self.y_all = Reader.get_subject_lable(subject_IDs, "Bleed6")
        self.y_lis.append(self.y_all)

        # Load network features for each region
        self.M1_dis, self.M1_hea = Reader.get_networks(subject_IDs, "M1")
        self.dis_lis.append(self.M1_dis)
        self.hea_lis.append(self.M1_hea)

        self.M2_dis, self.M2_hea = Reader.get_networks(subject_IDs, "M2")
        self.dis_lis.append(self.M2_dis)
        self.hea_lis.append(self.M2_hea)

        self.M3_dis, self.M3_hea = Reader.get_networks(subject_IDs, "M3")
        self.dis_lis.append(self.M3_dis)
        self.hea_lis.append(self.M3_hea)

        self.M4_dis, self.M4_hea = Reader.get_networks(subject_IDs, "M4")
        self.dis_lis.append(self.M4_dis)
        self.hea_lis.append(self.M4_hea)

        self.M5_dis, self.M5_hea = Reader.get_networks(subject_IDs, "M5")
        self.dis_lis.append(self.M5_dis)
        self.hea_lis.append(self.M5_hea)

        self.M6_dis, self.M6_hea = Reader.get_networks(subject_IDs, "M6")
        self.dis_lis.append(self.M6_dis)
        self.hea_lis.append(self.M6_hea)

        self.I_dis, self.I_hea = Reader.get_networks(subject_IDs, "I")
        self.dis_lis.append(self.I_dis)
        self.hea_lis.append(self.I_hea)

        self.C_dis, self.C_hea = Reader.get_networks(subject_IDs, "C")
        self.dis_lis.append(self.C_dis)
        self.hea_lis.append(self.C_hea)

        self.L_dis, self.L_hea = Reader.get_networks(subject_IDs, "L")
        self.dis_lis.append(self.L_dis)
        self.hea_lis.append(self.L_hea)

        self.IC_dis, self.IC_hea = Reader.get_networks(subject_IDs, "IC")
        self.dis_lis.append(self.IC_dis)
        self.hea_lis.append(self.IC_hea)

        # Load hemisphere information
        self.Right = Reader.get_subject_lable(subject_IDs, "Right")
        self.Left = Reader.get_subject_lable(subject_IDs, "Left")

        # Select region-specific data
        self.disease = copy.copy(self.dis_lis[index[region]])
        self.sub = copy.copy(self.hea_lis[index[region]])
        self.y = copy.copy(self.y_lis[index[region]])

        return self.disease, self.sub, self.y, self.Left, self.Right, subject_IDs

    def abb(self, A, B):
        """
        Concatenate two feature matrices along the feature dimension.
        """
        x_data = []
        for i in range(A.shape[0]):
            arr = np.concatenate((A[i], B[i]))
            x_data.append(arr)
        return np.array(x_data)

    def data_split(self, n_folds):
        """
        Perform stratified K-fold cross-validation split.
        """
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.disease, self.y))
        return cv_splits

    def post_nega(self, train_ind, y):
        """
        Separate positive and negative sample indices.
        """
        post_ind = []
        nega_ind = []
        for i in train_ind:
            if y[i] == 0.0:
                nega_ind.append(i)
            if y[i] == 1.0:
                post_ind.append(i)
        return np.asarray(post_ind), np.asarray(nega_ind)

    def forest(self, feature1, labels, train_ind, fnum):
        """
        Perform feature selection using Random Forest with grid search.
        """
        feature1 = ba.normalization1(feature1)
        train_dis, y_dis = feature1[train_ind], labels[train_ind]

        # Grid search for optimal number of trees
        rfc = RandomForestClassifier()
        num_estimator = {'n_estimators': range(50, 400, 50)}
        gs1 = GridSearchCV(rfc, num_estimator, scoring='roc_auc', cv=3)
        gs1.fit(train_dis, y_dis)

        # Grid search for maximum depth
        maxdepth = {'max_depth': range(3, 10, 1)}
        gs2 = GridSearchCV(
            RandomForestClassifier(n_estimators=gs1.best_estimator_.n_estimators),
            maxdepth, scoring='roc_auc', cv=3
        )
        gs2.fit(train_dis, y_dis)

        # Grid search for minimum samples split
        minsamples = {'min_samples_split': range(2, 50, 2)}
        gs3 = GridSearchCV(
            RandomForestClassifier(
                max_depth=gs2.best_estimator_.max_depth,
                n_estimators=gs1.best_estimator_.n_estimators
            ),
            minsamples, scoring='roc_auc', cv=3
        )
        gs3.fit(train_dis, y_dis)

        # Train Random Forest with optimal parameters
        best_rfc = RandomForestClassifier(
            max_depth=gs2.best_estimator_.max_depth,
            min_samples_split=gs3.best_estimator_.min_samples_split,
            n_estimators=gs1.best_estimator_.n_estimators
        )
        best_rfc.fit(train_dis, y_dis)

        # Rank features by importance
        importance1 = best_rfc.feature_importances_
        sorted_id = sorted(range(len(importance1)), key=lambda k: importance1[k], reverse=True)

        # Select top features
        feature2 = feature1[:, sorted_id][:, :fnum]
        return feature2

    def select(self, train_ind, k, region):
        """
        Select and optionally fuse features from multiple regions.
        """
        fnum1 = {
            "M1": 15, "M2": 20, "M3": 15, "M4": 20, "M5": 25,
            "M6": 20, "I": 20, "C": 25, "L": 20, "IC": 25
        }

        atten_dis = self.forest(self.dis_lis[index[region]], self.y_lis[index[region]], train_ind, fnum[region])
        atten_hea = self.forest(self.hea_lis[index[region]], self.y_lis[index[region]], train_ind, fnum[region])

        if k == 0:
            return atten_dis, atten_hea

        if k == 1:
            els = {
                "M1": [1, 3, 6], "M2": [0, 2, 4, 6], "M3": [1, 5],
                "M4": [0, 6, 5], "M5": [1, 3, 5, 6],
                "M6": [2, 4], "I": [0, 1, 3, 4, 8],
                "C": [8, 9], "L": [6, 7], "IC": [7, 8]
            }
            for i in els[region]:
                atten_dis = self.abb(
                    self.forest(self.dis_lis[i], self.y_lis[i], train_ind, fnum1[region]),
                    atten_dis
                )
                atten_hea = self.abb(
                    self.forest(self.hea_lis[i], self.y_lis[i], train_ind, fnum1[region]),
                    atten_hea
                )
            return atten_dis, atten_hea

        if k == 2:
            els = {
                "M1": [1, 4, 3, 6, 8], "M2": [4, 6], "M3": [1, 4, 5, 6],
                "M4": [1, 4, 0, 5, 6, 8], "M5": [1, 6],
                "M6": [1, 2, 4, 6], "I": [1, 4, 8],
                "C": [1, 4, 6, 8], "L": [6, 7], "IC": [4, 6, 8, 7]
            }
            for i in els[region]:
                atten_dis = self.abb(
                    self.forest(self.dis_lis[i], self.y_lis[i], train_ind, fnum1[region]),
                    atten_dis
                )
                atten_hea = self.abb(
                    self.forest(self.hea_lis[i], self.y_lis[i], train_ind, fnum1[region]),
                    atten_hea
                )
            return atten_dis, atten_hea

    def get_node_features(self, train_ind, test_ind, k, region, da):
        """
        Prepare node features for the graph neural network.
        """
        sam = 0.5
        if region in ["M2", "M5", "I", "L"]:
            sam = 1

        self.node_ftr_dis, self.node_ftr_hea = self.select(train_ind, k, region)

        return self.node_ftr_dis, self.node_ftr_hea, self.y, train_ind, test_ind

    def get_node_features_load(self, k, region, da, fold):
        """
        Load precomputed node features and train/test indices from disk.
        """
        base_path = os.path.join("", region, str(fold))

        self.node_ftr_dis = sio.loadmat(os.path.join(base_path, "L.mat"))["feature"]
        self.node_ftr_hea = sio.loadmat(os.path.join(base_path, "R.mat"))["feature"]
        self.node_ftr_sub = sio.loadmat(os.path.join(base_path, "sub.mat"))["feature"]

        train_ind = pd.read_csv(os.path.join(base_path, "train.csv"), header=None).values.flatten()
        test_ind = pd.read_csv(os.path.join(base_path, "test.csv"), header=None).values.flatten()

        self.node_ftr_all = self.node_ftr_sub
        return self.node_ftr_dis, self.node_ftr_hea, self.node_ftr_all, self.y, train_ind, test_ind

    def Data_Aug(self, train_ind, test_ind, sam):
        """
        Apply data augmentation using ADASYN.
        """
        train = self.node_ftr[train_ind]
        test = self.node_ftr[test_ind]
        train_label = self.y[train_ind]
        test_label = self.y[test_ind]

        A = ADASYN(sampling_strategy=sam)
        train, train_label = A.fit_resample(train, train_label)

        # Shuffle augmented training data
        idx = list(range(len(train)))
        random.shuffle(idx)
        train = train[idx]
        train_label = train_label[idx]

        # Combine training and test data
        train = np.vstack((train, test))
        train_label = np.hstack((train_label, test_label))

        t = len(train_label) - len(test_label)
        train_ind = np.arange(t)
        test_ind = np.arange(t, len(train_label))

        self.node_ftr = train
        return train_ind, test_ind, train_label
