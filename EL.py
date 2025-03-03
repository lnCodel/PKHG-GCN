import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch.nn.functional as F
from torch import nn
from opt import *
opt = OptInit().initialize()
class EL(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.3, hidden=512):  #1
        super(EL, self).__init__()
        self.parser = nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True),
                )
        self.matrix = nn.Parameter(torch.randn(1, input_dim)).to(opt.device)
        self.binary_matrix = torch.round(torch.sigmoid(self.matrix))#MLP也可以
        ##伯努利分布生成掩码也可以
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()
    def forward(self, x):
        x1 = x[:, 0:self.input_dim]
        x2 = x[:, self.input_dim:]
        h1 = self.binary_matrix * x1
        h2 = self.binary_matrix * x2
        p = (self.cos(h1, h2) + 1) * 0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
