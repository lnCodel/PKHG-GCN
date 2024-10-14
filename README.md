# PKHG-GCN
The data and code for the paper "PKHG-GCN: Prior Knowledge and Homophily Graph Approximation Guided Graph Convolutional Network for Automated ASPECTS Scoring on Non-Contrast CT" <br />

## Requirements
CUDA 11.7<br />
Python 3.10.13<br /> 
Pytorch 2.0.0<br />
Torchvision 0.15.0<br />
PYG 11.8<br />
SimpleITK 2.3.0 <br />
scipy 1.11.3 <br />

## Usage

### 0. Installation
* Install our PKHG as below
  
```
git clone https://github.com/lnCodel/PKIG.git

```

### 1.1 Skull-stripping
After converting the DICOM files of the dataset to NIfTI format, perform skull stripping according to the instructions at https://github.com/WuChanada/StripSkullCT.  <br />

### 1.2 Registration
We applyed a template of manually traced ASPECTS contours for non-linear registration onto all NCCTs. "Reg. py" shows the details.   <br />

### 1.3 Pre-processing
After registration is completed, we extracted radiomics features from each region. "batchprocessing.py" shows the details.  <br />

### 1.3 Feature selection
We have employed a Random Forest algorithm for feature selection to enhance the performance of our model. "dataloader.py" shows the details.

### 1.4 Train and Test
Ultimately, we utilized the features selected by the Random Forest for training and testing our model. The complete code can be accessed in the file "train_eval_PKIG.py".

## Acknowledgements
We are immensely appreciative of the invaluable support and assistance provided by EV-GCN. <br />

