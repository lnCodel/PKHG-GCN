# PKHG-GCN

This repository provides the official PyTorch implementation and data processing pipeline for the paper:

**PKHG-GCN: Prior Knowledge and Homophily Graph Approximation Guided Graph Convolutional Network for Automated ASPECTS Scoring on Non-Contrast CT**

The code is fully open-sourced to ensure the reproducibility of all experimental results reported in the paper.

---

## Requirements

- CUDA 11.7  
- Python 3.10.13  
- PyTorch 2.0.0  
- Torchvision 0.15.0  
- PyTorch Geometric (PyG) 1.1.8  
- SimpleITK 2.3.0  
- SciPy 1.11.3  

---

## Usage

### 0. Installation

Clone this repository:

```bash
git clone https://github.com/lnCodel/PKHG-GCN.git


1. Data Preprocessing Pipeline
1.1 Skull Stripping

After converting the DICOM files of the dataset to NIfTI format, skull stripping is performed following the instructions provided in the following repository:

https://github.com/WuChanada/StripSkullCT

1.2 Registration

A template with manually traced ASPECTS contours is applied to all NCCT scans using non-linear registration.
The implementation details are provided in Reg.py.

1.3 Radiomics Feature Extraction

After registration, radiomics features are extracted from each ASPECTS region.
The detailed implementation can be found in batchprocessing.py.

1.4 Feature Selection

A Random Forest (RF) algorithm is employed for feature selection to enhance the performance of the proposed model.
The implementation details are provided in dataloader.py.

Different Random Forest hyperparameters can be configured in dataloader.py to generate different feature subsets, which enables validation of the stability of the Random Forest–based feature selection.
In addition, the script analyze_common_top_features.py is provided to analyze and compare the features selected under different Random Forest configurations.

1.5 Model Training and Evaluation

The features selected by the Random Forest are used for training and testing the proposed PKHG-GCN model.
The complete training and evaluation pipeline is implemented in train_eval_PKIG.py.

Acknowledgements

We sincerely thank the authors of EV-GCN for their valuable contributions, which inspired parts of our implementation.
