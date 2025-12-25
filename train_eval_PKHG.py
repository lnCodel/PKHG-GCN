# Import necessary modules and libraries
from PKHG_GCN import PKHG  # Import the main PKHG-GCN model
from opt import *  # Import configuration options and hyperparameters
from utils.metrics import accuracy, auc, prf, save, to_icc, kappa, Over_all, interval1_std, save_logit  # Import evaluation metrics
from dataloader import dataloader  # Import data loading utilities
import warnings
warnings.filterwarnings("ignore")  # Suppress warning messages
import time
import torch
import torch.nn.functional as F  # Import neural network functions
import numpy as np
from thop import profile
import torch

# Define the number of regions (nodes in the graph)
region_number = 10
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch
from thop import profile

def compute_flops_and_params(
    model,
    x_dis,
    x_hea,
    edge_index,
    edge_attr,
    edge_mask,
    x,
    device=None
):
    """
    Compute FLOPs and number of parameters for a GNN model
    using a single forward pass.

    Args:
        model (torch.nn.Module): GNN model
        x_dis (Tensor): node features (distance branch)
        x_hea (Tensor): node features (head branch)
        edge_index (Tensor): graph edge index [2, E]
        edge_attr (Tensor): edge attributes
        edge_mask (Tensor): edge mask
        x (Tensor): auxiliary node features
        device (torch.device, optional): cpu or cuda

    Returns:
        flops (float): number of FLOPs
        params (float): number of trainable parameters
    """

    model.eval()

    if device is not None:
        model = model.to(device)
        x_dis = x_dis.to(device)
        x_hea = x_hea.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        edge_mask = edge_mask.to(device)
        x = x.to(device)

    with torch.no_grad():
        flops, params = profile(
            model,
            inputs=(x_dis, x_hea, edge_index, edge_attr, edge_mask, x),
            verbose=False
        )

    return flops, params

# Main execution block
if __name__ == '__main__':
    # Model configuration and naming
    named = "PKHG"  # Model identifier for saving results
    
    # Control flow for different execution modes
    a = 1
    if a == 0:
        # Mode 0: Execute Over_all function for comprehensive evaluation
        for asdw in [1]:
            Over_all(named, asdw, a)
    else:
        # Mode 1: Main training and evaluation pipeline
        for asdw in [0.45]:  # Hyperparameter loop
            execution_time = 0
            start_time = time.time()  # Start timing execution
            
            # Hyperparameter exploration loop
            for lx in [6, 8, 12, 14]:
                # Define hyperparameters for different brain regions
                lr = {"M1": 0.1, "M2": 0.1, "M3": 0.1, "M4": 0.1, "M5": 0.1, "M6": 0.1, "I": 0.1, "C": 0.1, "L": 0.1, "IC": 0.1, "ALL": 0.1}
                edropout = {"M1": 0.4, "M2": 0.3, "M3": 0.4, "M4": 0.4, "M5": 0.4, "M6": 0.3, "I": 0.4, "C": 0.4, "L": 0.4, "IC": 0.4, "ALL": 0.4}
                dropout = {"M1": 0.1, "M2": 0.1, "M3": 0.1, "M4": 0.1, "M5": 0.1, "M6": 0.1, "I": 0.1, "C": 0.1, "L": 0.1, "IC": 0.1, "ALL": 0.1}
                num_iter = {"M1": 1000, "M2": 1000, "M3": 1000, "M4": 1000, "M5": 1000, "M6": 1000, "I": 1000, "C": 1000, "L": 1000, "IC": 1000, "ALL": 500}
                
                # Define brain regions to process
                regions = ["M1", "M2", "M3", "M4", "M5", "M6", "L", "I", "C", "IC"]
                regions1 = ["ALL"]
                params = {"M1": 0.1, "M2": 0.1, "M3": 0.1, "M4": 0.1, "M5": 0.1, "M6": 0.1, "I": 0.1, "C": 0.1, "L": 0.1, "IC": 0.1}
                xy = [1]  # Additional parameter configuration
                
                # Main training loop
                for da in [1]:  # Data augmentation flag
                    for region in regions:  # Iterate through each brain region
                        for k in xy:  # Parameter variation
                            if da == 1 or da == 0:
                                # Initialize model options and configurations
                                opt = OptInit().initialize()
                                print('  Loading dataset Bleed4...')
                                
                                # Load and prepare data
                                dl = dataloader()
                                fea, labels, ids, R, L = dl.DataLoader(region)
                                x = labels
                                
                                # Set up cross-validation
                                n_folds = 10
                                cv_splits = dl.data_split(n_folds)
                                
                                # Initialize arrays to store evaluation metrics
                                corrects = np.zeros(n_folds, dtype=np.int32)
                                accs = np.zeros(n_folds, dtype=np.float32)
                                aucs = np.zeros(n_folds, dtype=np.float32)
                                prfs = np.zeros([n_folds, 3], dtype=np.float32)
                                sens = np.zeros(n_folds, dtype=np.float32)
                                spes = np.zeros(n_folds, dtype=np.float32)
                                loss_test = np.zeros(5, dtype=np.float32)
                                loss_train = np.zeros(5, dtype=np.float32)
                                acc_test_tu = np.zeros(5, dtype=np.float32)
                                kas = np.zeros(n_folds, dtype=np.float32)
                                iccs = np.zeros(n_folds, dtype=np.float32)
                                
                                # Cross-validation fold loop
                                for fold in range(n_folds):
                                    print("\r\n========================== Fold {},{} ==========================".format(fold, region))
                                    
                                    # Split data into training and testing sets
                                    train_ind = cv_splits[fold][0]
                                    test_ind1 = cv_splits[fold][1]
                                    test_ids = ids[test_ind1]
                                    train_ids = ids[train_ind]
                                    
                                    print('  Constructing graph data...')
                                    # Extract node features and prepare graph data
                                    node_ftr_dis, node_ftr_hea, node_ftr_all, y, train_ind, test_ind = dl.get_node_features_load(k, region, da, fold)
                                    X = dl.X_make(train_ind, region=region)
                                    
                                    # Identify positive and negative samples
                                    post_ind, nega_ind = dl.post_nega(train_ind, y)
                                    
                                    # Build graph structure (edges and adjacency matrix)
                                    edge_index, edgenet_input, edge_labels, train_labels, edge_mask_numpy = dl.get_PAE_inputs(region, train_ind)
                                    num_edge = edge_index.shape[1]
                                    
                                    # Normalize edge features using standardization
                                    edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
                                    
                                    # Initialize the PKHG-GCN model with specified parameters
                                    model = PKHG(node_ftr_dis.shape[1], opt.num_classes, dropout[region],
                                                 edge_dropout=edropout[region], hgc=opt.hgc,
                                                 lg=opt.lg, edgenet_input_dim=edgenet_input.shape[1] // 2, lg1=opt.lg1,
                                                 gl=asdw, yuan_input=node_ftr_dis.shape[1])
                                    model = model.to(opt.device)  # Move model to appropriate device (CPU/GPU)
                                    
                                    # Calculate class weights for imbalanced data handling
                                    a = sum(y[train_ind])
                                    b = len(y[train_ind])
                                    c = (b - a) / a
                                    weights = torch.tensor([1, c], dtype=torch.float).to(opt.device)
                                    
                                    # Define loss functions based on data augmentation flag
                                    if da == 0:
                                        loss_fn = torch.nn.CrossEntropyLoss()
                                        loss_fn1 = torch.nn.CrossEntropyLoss()
                                    else:
                                        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)  # Weighted loss for class imbalance
                                        loss_fn1 = torch.nn.CrossEntropyLoss()
                                    
                                    # Set up optimizer with learning rate and weight decay
                                    optimizer = torch.optim.Adam(model.parameters(), lr=lr[region], weight_decay=opt.wd)
                                    
                                    # Convert all data to PyTorch tensors and move to appropriate device
                                    features_cuda_dis = torch.tensor(node_ftr_dis, dtype=torch.float32).to(opt.device)
                                    features_cuda_hea = torch.tensor(node_ftr_hea, dtype=torch.float32).to(opt.device)
                                    features_cuda = torch.tensor(X, dtype=torch.float32).to(opt.device)
                                    edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
                                    edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
                                    labels = torch.tensor(y, dtype=torch.long).to(opt.device)
                                    edge_mask = torch.tensor(edge_mask_numpy, dtype=torch.long).to(opt.device)
                                    edge_labels = torch.tensor(edge_labels, dtype=torch.long).to(opt.device)
                                    one_shot = F.one_hot(edge_labels)  # Convert to one-hot encoding for edge classification
                                    
                                    # Define model save path with fold and region information
                                    fold_model_path = opt.ckpt_path + "/fold{}_region{}_method_{}__{}.pth".format(fold, region, k, da, asdw)
                                    n_samples = X.shape[0] // region_number
                                    
                                    # Prepare masks for correlated samples in contrastive learning
                                    mask_pos, mask_neg = dl.mask_correlated_samples_pos(R, L, node_ftr_dis.shape[0], train_ind, opt)

                                    # Training function definition
                                    def train():
                                        print("  Number of training samples %d" % len(train_ind))
                                        print("  Start training...\r\n")
                                        
                                        # Initialize tracking variables for monitoring training progress
                                        f1 = 0
                                        loss_train_list = []
                                        loss_test_list = []
                                        epoch_list = []
                                        test_acc_list = []
                                        c = 0
                                        
                                        # Training loop over specified number of iterations
                                        for epoch in range(num_iter[region]):
                                            model.train()  # Set model to training mode
                                            optimizer.zero_grad()  # Clear gradients from previous iteration
                                            
                                            # Forward pass with gradient computation enabled
                                            with torch.set_grad_enabled(True):
                                                # Model forward pass - compute node logits, edge weights, and other outputs
                                                node_logits, _, edge_weights, LR_logit, val, l, chayi_L, chayi_R, edge_index1 = model(
                                                    features_cuda_dis, features_cuda_hea, edge_index,
                                                    edgenet_input, edge_mask, features_cuda, region=region)
                                                
                                                # Identify top-k samples for focused training (hard example mining)
                                                proportion1 = 0.1
                                                num_size = int(len(train_ind) * proportion1)
                                                topk_index = dl.find_k_largest_indices(num_size, train_ind, test_ind, [-1], val)
                                                
                                                # Calculate classification loss with emphasis on challenging samples
                                                loss_cls = 0.9 * loss_fn(node_logits[train_ind], labels[train_ind]) + \
                                                           0.1 * loss_fn(node_logits[topk_index], labels[topk_index])
                                                
                                                # Combined loss function with edge-aware regularization
                                                loss = 0.9 * loss_cls + \
                                                       0.1 * dl.nn_loss_k(edge_weights, one_shot, edge_mask_numpy[topk_index], train_labels)
                                                
                                                # Backward pass and optimization step
                                                loss.backward()
                                                optimizer.step()
                                            
                                            # Calculate training accuracy and metrics
                                            correct_train, acc_train, spe, sen, spe_num, sen_num = accuracy(
                                                node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                                            
                                            # Evaluation mode - no gradient computation
                                            model.eval()
                                            with torch.set_grad_enabled(False):
                                                # Forward pass without gradient computation for validation
                                                node_logits, _, edge_weights, LR_logit, _, _, _, _, _ = model(
                                                    features_cuda_dis, features_cuda_hea, edge_index,
                                                    edgenet_input, edge_mask, features_cuda, region=region)
                                                
                                                # Track metrics at regular intervals for monitoring
                                                if epoch % 5 == 0 and epoch != 0:
                                                    loss_test_list.append(np.mean(loss_test))
                                                    loss_train_list.append(np.mean(loss_train))
                                                    test_acc_list.append(np.mean(acc_test_tu))
                                                    epoch_list.append(c)
                                                    c += 1
                                                
                                                # Calculate test accuracy on current mini-batch
                                                a = epoch % 5
                                                logits_test = node_logits[test_ind].detach().cpu().numpy()
                                                correct_test, acc_test, spe, sen, spe_num, sen_num = accuracy(
                                                    logits_test, y[test_ind])
                                                acc_test_tu[a] = acc_test
                                            
                                            # Calculate additional evaluation metrics
                                            auc_test = auc(logits_test, y[test_ind])  # Area Under Curve
                                            prf_test = prf(logits_test, y[test_ind])  # Precision, Recall, F1-score
                                            icc = to_icc(logits_test, y[test_ind])  # Intraclass Correlation Coefficient
                                            ka = kappa(logits_test, y[test_ind])  # Cohen's Kappa
                                            
                                            # Print training progress and metrics
                                            print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f},\ttest acc: {:.5f},\tspe: {:.5f},\tsne: {:.5f}".format(
                                                epoch, loss.item(), acc_train.item(), acc_test.item(), spe, sen))
                                            
                                            # Save model if performance improves (early stopping criteria)
                                            if (prf_test[2] > f1 and epoch > 9) or (prf_test[2] == f1 and aucs[fold] < auc_test and epoch > 9):
                                                print("save!")
                                                f1 = prf_test[2]  # Update best F1-score
                                                acc = acc_test
                                                aucs[fold] = auc_test
                                                corrects[fold] = correct_test
                                                accs[fold] = acc_test
                                                prfs[fold] = prf_test
                                                spes[fold] = spe
                                                sens[fold] = sen
                                                kas[fold] = ka
                                                iccs[fold] = icc
                                                
                                                # Save model checkpoint if path is specified
                                                if opt.ckpt_path != '':
                                                    if not os.path.exists(opt.ckpt_path):
                                                        os.makedirs(opt.ckpt_path)
                                                    torch.save(model.state_dict(), fold_model_path)
                                        
                                        # Print fold results after training completion
                                        n = len(epoch_list)
                                        print("\r\n => Fold {} test accuacry {:.5f},auc {:.5f},f1 {:.5f} ".format(fold, accs[fold], aucs[fold], f1))
                                    
                                    # Evaluation function definition
                                    def evaluate():
                                        print("  Number of testing samples %d" % len(test_ind))
                                        print('  Start testing...')
                                        
                                        # Load saved model weights from checkpoint
                                        model.load_state_dict(torch.load(fold_model_path))
                                        model.eval()  # Set model to evaluation mode
                                        
                                        # Forward pass for testing
                                        node_logits, res, edge_weights, LR_logit, _, _, _, _, _ = model(
                                            features_cuda_dis, features_cuda_hea, edge_index,
                                            edgenet_input, edge_mask, features_cuda, region=region)
                                        
                                        # Prepare predictions for analysis and saving
                                        logits_test = node_logits[test_ind].detach().cpu().numpy()
                                        res_test = res[test_ind].detach().cpu().numpy()
                                        se = y[test_ind]
                                        sad = np.column_stack((logits_test, se))
                                        sad1 = np.column_stack((res_test, se))
                                        sad1 = np.column_stack((sad1, se))
                                        
                                        # Calculate comprehensive evaluation metrics
                                        corrects[fold], accs[fold], spe, sen, spe_num, sen_num = accuracy(
                                            node_logits[test_ind].detach().cpu().numpy(), y[test_ind])
                                        save(logits_test, y[test_ind], local=ids[test_ind], named=named, region=region, asdw=lx)
                                        kas[fold] = kappa(logits_test, y[test_ind])
                                        sens[fold] = sen
                                        spes[fold] = spe
                                        aucs[fold] = auc(logits_test, y[test_ind])
                                        prfs[fold] = prf(logits_test, y[test_ind])
                                        
                                        print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))
                                    
                                    # Execute training or evaluation based on configuration
                                    if opt.train == 1:
                                        train()
                                    elif opt.train == 0:
                                        evaluate()
                                
                                # Calculate and print execution time for performance monitoring
                                end_time = time.time()
                                execution_time = end_time - start_time
                                print(f"Code running time: {execution_time / 10:.4f} seconds")
                                
                                # Print comprehensive results with standard deviations for current region
                                print("\r\n========================== Finish ==========================", region)
                                print("acc:std", interval1_std(accs))
                                print("auc:std", interval1_std(aucs))
                                print("sen:std", interval1_std(sens))
                                print("spe:std", interval1_std(spes))
                                print("kappa:std", interval1_std(kas))
                                print("f1:std", interval1_std(prfs[:, 2]))
                                
                                # Calculate and print overall performance metrics across all folds
                                n_samples = X.shape[0] // region_number
                                acc_nfold = np.sum(corrects) / (n_samples)
                                print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
                                print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
                                se, sp, f1 = np.mean(prfs, axis=0)
                                se_std, sp_std, f1_std = np.std(prfs, axis=0)
                                print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(
                                    np.mean(sens), np.mean(spes), f1))
                                print("=> Average kappa sensitivity {:.4f}".format(np.mean(kas)))
