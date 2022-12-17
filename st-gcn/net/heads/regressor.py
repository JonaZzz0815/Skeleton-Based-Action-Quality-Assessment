import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim,dropout_ratio=0.):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)
         

    def forward(self, x):
        x = self.activation(self.dropout(self.layer1(x)))
        x = self.activation(self.dropout(self.layer2(x)))
        output = self.softmax(self.layer3(x))
        return output


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim,model_type='single', num_judges=None,dropout_ratio=0.):
        super(Evaluator, self).__init__()
        self.model_type = model_type

        if model_type == 'single':
            self.evaluator = MLP_block(feature_dim=feature_dim,output_dim=output_dim,dropout_ratio=dropout_ratio)
        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.ModuleList([MLP_block(feature_dim=feature_dim,output_dim=output_dim,dropout_ratio=dropout_ratio) 
                                            for _ in range(num_judges)])

    def forward(self, feats_avg):  # data: NCTHW ??? 

        if self.model_type == 'single':
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs
