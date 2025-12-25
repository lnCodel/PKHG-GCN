# Import core model
from PKHG_GCN import PKHG

# Import hyperparameter configuration
from opt import *

# Import evaluation metrics and utility functions
from utils.metrics import (
    accuracy, auc, prf, save, auc_la, sa,
    to_icc, kappa, Over_all, interval1_std
)

# Import data loader
from dataloader import dataloader


# Excel read/write libraries
import xlrd
import xlwt
from xlutils.copy import copy

# PyTorch functional API
import torch.nn.functional as F


# ==========================
# Excel writing utilities
# ==========================
def write_excel_xls(path, sheet_name, value):
    """
    Create a new Excel file and write data into it.
    """
    index = len(value)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)

    for i in range(index):
        for j in range(len(value[i])):
            sheet.write(i, j, value[i][j])

    workbook.save(path)


def write_excel_xls_append(path, value):
    """
    Append data to an existing Excel file.
    """
    index = len(value)
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    rows_old = worksheet.nrows

    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)

    for i in range(index):
        for j in range(len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])

    new_workbook.save(path)
    print("Excel file updated successfully.")


# ==========================
# Main execution
# ==========================
if __name__ == '__main__':

    # Model name identifier
    named = "PKHG-GCN"

    # Control switch for global evaluation
    a = 1

    # Run overall evaluation if enabled
    if a == 0:
        for asdw in [0.55]:
            Over_all(named, asdw)

    else:
        # Graph-level hyperparameter
        for asdw in [0.55]:

            # Initialize Excel file
            book_name_xls = f'gl={asdw}.xls'
            sheet_name_xls = 'results'

            # Table header
            value_title = [[
                "Model", "Fusion_Method", "Data_Augmentation",
                "ACC", "ACC_std", "AUC", "AUC_std",
                "Sensitivity", "Sensitivity_std",
                "Specificity", "Specificity_std",
                "Precision", "Precision_std",
                "Recall", "Recall_std",
                "F1", "F1_std",
                "Kappa", "Kappa_std",
                "ICC", "ICC_std"
            ]]

            write_excel_xls(book_name_xls, sheet_name_xls, value_title)

            # Learning rate for each region
            lr = {"M1": 0.1, "M2": 0.1, "M3": 0.1, "M4": 0.1,
                  "M5": 0.1, "M6": 0.1, "I": 0.1,
                  "C": 0.1, "L": 0.1, "IC": 0.1, "ALL": 0.1}

            # Edge dropout rate
            edropout = {"M1": 0.4, "M2": 0.3, "M3": 0.4, "M4": 0.4,
                        "M5": 0.4, "M6": 0.3, "I": 0.4,
                        "C": 0.4, "L": 0.4, "IC": 0.4, "ALL": 0.4}

            # Node dropout rate
            dropout = {"M1": 0.1, "M2": 0.1, "M3": 0.01, "M4": 0.01,
                       "M5": 0.1, "M6": 0.1, "I": 0.1,
                       "C": 0.1, "L": 0.1, "IC": 0.1, "ALL": 0.1}

            # Number of training epochs
            num_iter = {key: 1000 for key in lr}

            # Regions to evaluate
            regions = ["M1", "M2", "M3", "M4", "M5", "M6", "L", "I", "C", "IC"]

            # Fusion strategy index
            xy = [1]

            # Data augmentation flag
            for da in [0]:
                for region in regions:
                    for k in xy:

                        # Initialize configuration
                        opt = OptInit().initialize()

                        print('Loading dataset...')
                        dl = dataloader()

                        # Load raw data
                        disease, sub, y, L, R, ids = dl.load_data(region)

                        # K-fold cross validation
                        n_folds = 10
                        cv_splits = dl.data_split(n_folds)

                        # Metric containers
                        corrects = np.zeros(n_folds)
                        accs = np.zeros(n_folds)
                        aucs = np.zeros(n_folds)
                        prfs = np.zeros((n_folds, 3))
                        sens = np.zeros(n_folds)
                        spes = np.zeros(n_folds)
                        kas = np.zeros(n_folds)
                        iccs = np.zeros(n_folds)

                        # ==========================
                        # Cross-validation loop
                        # ==========================
                        for fold in range(n_folds):
                            print(f"\n===== Fold {fold}, Region {region} =====")

                            train_ind, test_ind1 = cv_splits[fold]
                            test_ids = ids[test_ind1]

                            # Load graph features
                            node_ftr_dis, node_ftr_hea, node_ftr_all, y, train_ind, test_ind = \
                                dl.get_node_features_load(k, region, da, fold)

                            # Build edge inputs
                            edge_index, edgenet_input, edge_labels, train_labels, edge_mask = \
                                dl.get_PAE_inputs(region, train_ind)

                            # Normalize edge features
                            edgenet_input = (edgenet_input - edgenet_input.mean(0)) / edgenet_input.std(0)

                            # Initialize model
                            model = PKHG(
                                node_ftr_dis.shape[1],
                                opt.num_classes,
                                dropout[region],
                                edge_dropout=edropout[region],
                                hgc=opt.hgc,
                                lg=opt.lg,
                                edgenet_input_dim=edgenet_input.shape[1] // 2,
                                lg1=opt.lg1,
                                gl=asdw
                            ).to(opt.device)

                            # Loss and optimizer
                            loss_fn = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(
                                model.parameters(),
                                lr=lr[region],
                                weight_decay=opt.wd
                            )

                            # Tensor conversion
                            features_cuda_dis = torch.tensor(node_ftr_dis).float().to(opt.device)
                            features_cuda_hea = torch.tensor(node_ftr_hea).float().to(opt.device)
                            edge_index = torch.tensor(edge_index).long().to(opt.device)
                            edgenet_input = torch.tensor(edgenet_input).float().to(opt.device)
                            labels = torch.tensor(y).long().to(opt.device)
                            edge_mask = torch.tensor(edge_mask).long().to(opt.device)
                            edge_labels = torch.tensor(edge_labels).long().to(opt.device)
                            one_hot_edges = F.one_hot(edge_labels)

                            # ==========================
                            # Training function
                            # ==========================
                            def train():
                                model.train()
                                best_f1 = 0

                                for epoch in range(num_iter[region]):
                                    optimizer.zero_grad()

                                    node_logits, edge_weights, _, val = model(
                                        features_cuda_dis,
                                        features_cuda_hea,
                                        edge_index,
                                        edgenet_input,
                                        edge_mask
                                    )

                                    # Select top-k informative nodes
                                    num_size = int(len(train_ind) * 0.1)
                                    topk_index = dl.find_k_largest_indices(
                                        num_size, train_ind, test_ind, [-1], val
                                    )

                                    # Classification + edge loss
                                    loss_cls = loss_fn(node_logits[train_ind], labels[train_ind])
                                    loss_edge = dl.nn_loss_k(
                                        edge_weights, one_hot_edges,
                                        edge_mask[topk_index], train_labels
                                    )

                                    loss = 0.9 * loss_cls + 0.1 * loss_edge
                                    loss.backward()
                                    optimizer.step()

                                    # Evaluation
                                    logits_test = node_logits[test_ind].detach().cpu().numpy()
                                    acc, _, _, _, _, _ = accuracy(logits_test, y[test_ind])
                                    f1 = prf(logits_test, y[test_ind])[2]

                                    if f1 > best_f1:
                                        best_f1 = f1
                                        torch.save(model.state_dict(), fold_model_path)

                            # ==========================
                            # Evaluation function
                            # ==========================
                            def evaluate():
                                model.load_state_dict(torch.load(fold_model_path))
                                model.eval()

                                node_logits, _, _, _ = model(
                                    features_cuda_dis,
                                    features_cuda_hea,
                                    edge_index,
                                    edgenet_input,
                                    edge_mask
                                )

                                logits_test = node_logits[test_ind].detach().cpu().numpy()
                                corrects[fold], accs[fold], spe, sen, _, _ = accuracy(
                                    logits_test, y[test_ind]
                                )

                                aucs[fold] = auc(logits_test, y[test_ind])
                                prfs[fold] = prf(logits_test, y[test_ind])
                                sens[fold] = sen
                                spes[fold] = spe
                                kas[fold] = kappa(logits_test, y[test_ind])
                                iccs[fold] = to_icc(logits_test, y[test_ind])

                            if opt.train == 1:
                                train()
                            else:
                                evaluate()

                        # ==========================
                        # Aggregate results
                        # ==========================
                        value = [
                            region, str(k), str(da),
                            str(np.mean(accs)), str(np.std(accs)),
                            str(np.mean(aucs)), str(np.std(aucs)),
                            str(np.mean(sens)), str(np.std(sens)),
                            str(np.mean(spes)), str(np.std(spes)),
                            str(np.mean(prfs[:, 0])), str(np.std(prfs[:, 0])),
                            str(np.mean(prfs[:, 1])), str(np.std(prfs[:, 1])),
                            str(np.mean(prfs[:, 2])), str(np.std(prfs[:, 2])),
                            str(np.mean(kas)), str(np.std(kas)),
                            str(np.mean(iccs)), str(np.std(iccs))
                        ]

                        write_excel_xls_append(book_name_xls, [value])
